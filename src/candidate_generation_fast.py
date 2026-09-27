from pathlib import Path
import sqlite3
import pandas as pd
import re
import time
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DB_FILE = PROCESSED_DIR / "blocking_index.db"
OUTPUT_FILE = PROCESSED_DIR / "candidate_pairs.tsv"

CHUNK_SIZE = 10_000

STOPWORDS = {
    "the", "and", "for", "with", "from", "that",
    "this", "inc", "llc", "ltd", "co", "company"
}


def normalize_text(value):
    if value is None:
        return ""

    value = str(value).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def tokenize(value):
    value = normalize_text(value)

    if not value:
        return []

    return [
        token
        for token in value.split()
        if len(token) >= 2 and token not in STOPWORDS
    ]


if not DB_FILE.exists():
    raise FileNotFoundError(
        f"Blocking database not found: {DB_FILE}"
    )


print("=" * 70)
print("FINAL CANDIDATE GENERATION")
print("=" * 70)

print(f"Database: {DB_FILE}")
print(
    f"Database size: "
    f"{DB_FILE.stat().st_size / (1024 ** 3):.2f} GB"
)
print(f"Chunk size: {CHUNK_SIZE:,}")
print(f"Output: {OUTPUT_FILE}")
print()


# ------------------------------------------------------------
# CONNECT TO DATABASE
# ------------------------------------------------------------

conn = sqlite3.connect(str(DB_FILE))

conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
conn.execute("PRAGMA temp_store=FILE")

cursor = conn.cursor()


# ------------------------------------------------------------
# CHECK DATABASE
# ------------------------------------------------------------

print("Checking database...")

record_count = cursor.execute(
    "SELECT COUNT(*) FROM records"
).fetchone()[0]

token_count = cursor.execute(
    "SELECT COUNT(*) FROM token_index"
).fetchone()[0]

print(f"Records:    {record_count:,}")
print(f"Token rows: {token_count:,}")
print()


# ------------------------------------------------------------
# SOURCE1
# ------------------------------------------------------------

source1_file = TRAIN_DIR / "train_source1.tsv"

total_source1 = sum(
    1
    for _ in open(
        source1_file,
        "r",
        encoding="utf-8",
        errors="ignore"
    )
) - 1

total_chunks = (
    total_source1 + CHUNK_SIZE - 1
) // CHUNK_SIZE

print(f"Source1 rows: {total_source1:,}")
print(f"Total chunks: {total_chunks:,}")
print()


# ------------------------------------------------------------
# RESUME SUPPORT
# ------------------------------------------------------------

# If output already exists, determine how many Source1 rows
# have already been completely processed.
#
# Each chunk is written as a temporary file first.
# After successful completion, it is appended to the final file.
#
# We use a progress file to know exactly where to resume.

PROGRESS_FILE = PROCESSED_DIR / "candidate_generation_progress.txt"

if PROGRESS_FILE.exists():
    try:
        start_chunk = int(
            PROGRESS_FILE.read_text().strip()
        )
    except Exception:
        start_chunk = 0
else:
    start_chunk = 0


# ------------------------------------------------------------
# INITIALIZE OUTPUT
# ------------------------------------------------------------

if start_chunk == 0:

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "source1_id\tcandidate_id\tsource\n"
        )

    print("Starting candidate generation from chunk 1.")

else:

    print(
        f"Resuming from chunk "
        f"{start_chunk + 1:,} / {total_chunks:,}"
    )

print()


# ------------------------------------------------------------
# PROCESS SOURCE1 IN CHUNKS
# ------------------------------------------------------------

global_start = time.time()

reader = pd.read_csv(
    source1_file,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK_SIZE
)


for chunk_number, df in enumerate(reader):

    # Skip chunks already completed.
    if chunk_number < start_chunk:
        continue

    chunk_start = time.time()

    print(
        "=" * 70
    )
    print(
        f"CHUNK {chunk_number + 1:,} / "
        f"{total_chunks:,}"
    )

    first_row = chunk_number * CHUNK_SIZE + 1
    last_row = min(
        (chunk_number + 1) * CHUNK_SIZE,
        total_source1
    )

    print(
        f"Source1 rows: "
        f"{first_row:,} - {last_row:,}"
    )

    # --------------------------------------------------------
    # NORMALIZE SOURCE1
    # --------------------------------------------------------

    df["country_norm"] = (
        df["country"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["name_norm"] = (
        df["business_name"]
        .map(normalize_text)
    )

    df["address_norm"] = (
        df["business_address"]
        .map(normalize_text)
    )

    # --------------------------------------------------------
    # COLLECT UNIQUE NAME TOKEN REQUESTS
    # --------------------------------------------------------

    name_requests = set()

    for _, row in df.iterrows():

        country = row["country_norm"]

        for token in tokenize(
            row["business_name"]
        ):
            name_requests.add(
                (country, token)
            )

    print(
        f"Unique name/token requests: "
        f"{len(name_requests):,}"
    )

    # --------------------------------------------------------
    # NAME TOKEN LOOKUPS
    # --------------------------------------------------------

    token_candidates = {}

    for country, token in name_requests:

        rows = cursor.execute(
            """
            SELECT entity_id, source
            FROM token_index
            WHERE country = ?
              AND token = ?
            """,
            (country, token)
        ).fetchall()

        token_candidates[
            (country, token)
        ] = rows

    # --------------------------------------------------------
    # EXACT NAME LOOKUPS
    # --------------------------------------------------------

    name_exact = {}

    unique_names = set(
        (
            r["country_norm"],
            r["name_norm"]
        )
        for _, r in df.iterrows()
        if r["name_norm"]
    )

    for country, name in unique_names:

        rows = cursor.execute(
            """
            SELECT entity_id, source
            FROM records
            WHERE country = ?
              AND name = ?
            """,
            (country, name)
        ).fetchall()

        name_exact[
            (country, name)
        ] = rows

    # --------------------------------------------------------
    # EXACT ADDRESS LOOKUPS
    # --------------------------------------------------------

    address_exact = {}

    unique_addresses = set(
        (
            r["country_norm"],
            r["address_norm"]
        )
        for _, r in df.iterrows()
        if r["address_norm"]
    )

    for country, address in unique_addresses:

        rows = cursor.execute(
            """
            SELECT entity_id, source
            FROM records
            WHERE country = ?
              AND address = ?
            """,
            (country, address)
        ).fetchall()

        address_exact[
            (country, address)
        ] = rows

    # --------------------------------------------------------
    # WRITE CHUNK TO TEMP FILE
    # --------------------------------------------------------

    temp_file = (
        PROCESSED_DIR
        / f"candidate_chunk_{chunk_number:05d}.tsv"
    )

    chunk_candidate_count = 0

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        for _, row in df.iterrows():

            s1_id = row["entity_id"]

            country = row["country_norm"]
            name = row["name_norm"]
            address = row["address_norm"]

            candidate_set = set()

            # -----------------------------------------------
            # NAME-TOKEN BLOCKING
            # -----------------------------------------------

            for token in tokenize(
                row["business_name"]
            ):

                for candidate_id, source in (
                    token_candidates.get(
                        (country, token),
                        []
                    )
                ):

                    candidate_set.add(
                        (candidate_id, source)
                    )

            # -----------------------------------------------
            # EXACT NAME
            # -----------------------------------------------

            for candidate_id, source in (
                name_exact.get(
                    (country, name),
                    []
                )
            ):

                candidate_set.add(
                    (candidate_id, source)
                )

            # -----------------------------------------------
            # EXACT ADDRESS
            # -----------------------------------------------

            for candidate_id, source in (
                address_exact.get(
                    (country, address),
                    []
                )
            ):

                candidate_set.add(
                    (candidate_id, source)
                )

            # -----------------------------------------------
            # WRITE CANDIDATES
            # -----------------------------------------------

            for candidate_id, source in candidate_set:

                f.write(
                    f"{s1_id}\t"
                    f"{candidate_id}\t"
                    f"{source}\n"
                )

                chunk_candidate_count += 1

    # --------------------------------------------------------
    # APPEND COMPLETED CHUNK TO FINAL OUTPUT
    # --------------------------------------------------------

    with open(
        temp_file,
        "r",
        encoding="utf-8"
    ) as source_file:

        with open(
            OUTPUT_FILE,
            "a",
            encoding="utf-8"
        ) as output_file:

            for line in source_file:
                output_file.write(line)

    # Remove temporary chunk
    temp_file.unlink()

    # Mark chunk completed
    PROGRESS_FILE.write_text(
        str(chunk_number + 1)
    )

    chunk_elapsed = (
        time.time() - chunk_start
    )

    total_elapsed = (
        time.time() - global_start
    )

    completed = chunk_number + 1

    remaining = (
        total_chunks - completed
    )

    if completed:
        avg_chunk_time = (
            total_elapsed / completed
        )
    else:
        avg_chunk_time = 0

    estimated_remaining = (
        remaining * avg_chunk_time
    )

    print()
    print(
        f"Candidates this chunk: "
        f"{chunk_candidate_count:,}"
    )

    print(
        f"Chunk time: "
        f"{chunk_elapsed / 60:.2f} minutes"
    )

    print(
        f"Total elapsed: "
        f"{total_elapsed / 60:.2f} minutes"
    )

    print(
        f"Estimated remaining: "
        f"{estimated_remaining / 60:.2f} minutes"
    )

    print(
        f"Progress: "
        f"{completed:,}/{total_chunks:,}"
    )

    print()


# ------------------------------------------------------------
# FINISHED
# ------------------------------------------------------------

conn.close()

elapsed = time.time() - global_start

print("=" * 70)
print("CANDIDATE GENERATION COMPLETE")
print("=" * 70)

print(
    f"Source1 rows: "
    f"{total_source1:,}"
)

print(
    f"Total time: "
    f"{elapsed / 3600:.2f} hours"
)

print(
    f"Output: "
    f"{OUTPUT_FILE}"
)

print("=" * 70)