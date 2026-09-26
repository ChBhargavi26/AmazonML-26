from pathlib import Path
import sqlite3
import re
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = OUTPUT_DIR / "blocking_index.db"
OUTPUT_FILE = OUTPUT_DIR / "candidate_pairs.tsv"


# ============================================================
# SETTINGS
# ============================================================

STOPWORDS = {
    "the", "and", "of", "for", "a", "an",
    "inc", "llc", "ltd", "limited",
    "corp", "corporation", "company", "co",
    "group", "center", "centre"
}

MAX_TOKEN_BLOCK = 5000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def get_tokens(text):
    return {
        token
        for token in text.split()
        if len(token) >= 2
        and token not in STOPWORDS
    }


# ============================================================
# CREATE SQLITE DATABASE
# ============================================================

print("Creating blocking index...")

if DB_FILE.exists():
    DB_FILE.unlink()

conn = sqlite3.connect(DB_FILE)

conn.execute("""
CREATE TABLE records (
    source TEXT,
    entity_id TEXT,
    country TEXT,
    name TEXT,
    address TEXT
)
""")

conn.execute("""
CREATE TABLE token_index (
    country TEXT,
    token TEXT,
    source TEXT,
    entity_id TEXT
)
""")

conn.execute("""
CREATE INDEX idx_token
ON token_index(country, token)
""")

conn.execute("""
CREATE INDEX idx_name
ON records(country, name)
""")

conn.execute("""
CREATE INDEX idx_address
ON records(country, address)
""")


# ============================================================
# LOAD S2/S3 INTO SQLITE
# ============================================================

def index_source(filename, source_name):

    print(f"Indexing {source_name}...")

    for chunk in pd.read_csv(
        TRAIN_DIR / filename,
        sep="\t",
        chunksize=100_000,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ):

        rows = []
        token_rows = []

        for row in chunk.itertuples(index=False):

            entity_id = str(row.entity_id)
            country = normalize_text(row.country)
            name = normalize_text(row.business_name)
            address = normalize_text(row.business_address)

            rows.append(
                (
                    source_name,
                    entity_id,
                    country,
                    name,
                    address
                )
            )

            for token in get_tokens(name):

                token_rows.append(
                    (
                        country,
                        token,
                        source_name,
                        entity_id
                    )
                )

        conn.executemany(
            """
            INSERT INTO records
            VALUES (?, ?, ?, ?, ?)
            """,
            rows
        )

        conn.executemany(
            """
            INSERT INTO token_index
            VALUES (?, ?, ?, ?)
            """,
            token_rows
        )

        conn.commit()

    print(f"{source_name} indexed.")


index_source(
    "train_source2.tsv",
    "source2"
)

index_source(
    "train_source3.tsv",
    "source3"
)


# ============================================================
# REMOVE VERY COMMON TOKENS
# ============================================================

print("Removing oversized token blocks...")

conn.execute("""
DELETE FROM token_index
WHERE rowid IN (
    SELECT ti.rowid
    FROM token_index ti
    JOIN (
        SELECT country, token
        FROM token_index
        GROUP BY country, token
        HAVING COUNT(*) > ?
    ) big
    ON ti.country = big.country
    AND ti.token = big.token
)
""", (MAX_TOKEN_BLOCK,))

conn.commit()

print("Blocking index ready.")


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print("Generating candidates...")

# Start fresh
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(
        "source1_entity_id\t"
        "candidate_entity_id\t"
        "candidate_source\n"
    )


def generate_for_s1_chunk(chunk):

    output_rows = []

    for row in chunk.itertuples(index=False):

        s1_id = str(row.entity_id)
        country = normalize_text(row.country)
        name = normalize_text(row.business_name)
        address = normalize_text(row.business_address)

        candidate_set = set()

        # ----------------------------------------------------
        # 1. Name-token blocking
        # ----------------------------------------------------

        for token in get_tokens(name):

            results = conn.execute(
                """
                SELECT source, entity_id
                FROM token_index
                WHERE country = ?
                AND token = ?
                """,
                (country, token)
            )

            for source, entity_id in results:
                candidate_set.add(
                    (entity_id, source)
                )

        # ----------------------------------------------------
        # 2. Exact normalized name
        # ----------------------------------------------------

        results = conn.execute(
            """
            SELECT source, entity_id
            FROM records
            WHERE country = ?
            AND name = ?
            AND name != ''
            """,
            (country, name)
        )

        for source, entity_id in results:
            candidate_set.add(
                (entity_id, source)
            )

        # ----------------------------------------------------
        # 3. Exact normalized address
        # ----------------------------------------------------

        if address:

            results = conn.execute(
                """
                SELECT source, entity_id
                FROM records
                WHERE country = ?
                AND address = ?
                AND address != ''
                """,
                (country, address)
            )

            for source, entity_id in results:
                candidate_set.add(
                    (entity_id, source)
                )

        for entity_id, source in candidate_set:

            output_rows.append(
                (
                    s1_id,
                    entity_id,
                    source
                )
            )

    return output_rows


# Process Source 1 in chunks

for chunk_number, chunk in enumerate(
    pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
        chunksize=50_000,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ),
    start=1
):

    rows = generate_for_s1_chunk(chunk)

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        for s1_id, candidate_id, source in rows:

            f.write(
                f"{s1_id}\t"
                f"{candidate_id}\t"
                f"{source}\n"
            )

    print(
        f"Processed S1 chunk {chunk_number}"
    )


# ============================================================
# CLEAN UP
# ============================================================

conn.close()

print("\nCandidate generation complete.")
print("Output:")
print(OUTPUT_FILE)