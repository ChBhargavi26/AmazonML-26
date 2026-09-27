from pathlib import Path
import pandas as pd
import re
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = PROJECT_ROOT / "data" / "raw" / "test"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "submission.csv"

CHUNK_SIZE = 50_000

STOPWORDS = {
    "the", "and", "for", "with", "from", "that",
    "this", "inc", "llc", "ltd", "co", "company"
}


def normalize_text(value):
    if value is None:
        return ""

    value = str(value).lower()

    value = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        value
    )

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


print("=" * 70)
print("QUICK TEST PREDICTION - EXACT NAME + ADDRESS")
print("=" * 70)

start_time = time.time()

# ------------------------------------------------------------
# LOAD TEST SOURCE2 + SOURCE3
# ------------------------------------------------------------

print()
print("Loading test reference data...")

s2 = pd.read_csv(
    TEST_DIR / "test_source2.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s3 = pd.read_csv(
    TEST_DIR / "test_source3.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

reference = pd.concat(
    [s2, s3],
    ignore_index=True
)

reference["country_norm"] = (
    reference["country"]
    .astype(str)
    .str.lower()
    .str.strip()
)

reference["name_norm"] = (
    reference["business_name"]
    .map(normalize_text)
)

reference["address_norm"] = (
    reference["business_address"]
    .map(normalize_text)
)

print(f"Reference rows: {len(reference):,}")

# ------------------------------------------------------------
# BUILD EXACT NAME INDEX
# ------------------------------------------------------------

print()
print("Building exact-name index...")

name_index = {}

for pos, row in enumerate(
    reference[
        ["country_norm", "name_norm"]
    ].itertuples(index=False)
):
    country, name = row

    if name:
        key = (country, name)

        name_index.setdefault(
            key,
            []
        ).append(pos)

print(
    f"Exact-name keys: {len(name_index):,}"
)

# ------------------------------------------------------------
# BUILD EXACT ADDRESS INDEX
# ------------------------------------------------------------

print("Building exact-address index...")

address_index = {}

for pos, row in enumerate(
    reference[
        ["country_norm", "address_norm"]
    ].itertuples(index=False)
):
    country, address = row

    if address:
        key = (country, address)

        address_index.setdefault(
            key,
            []
        ).append(pos)

print(
    f"Exact-address keys: {len(address_index):,}"
)

# ------------------------------------------------------------
# PROCESS TEST SOURCE1
# ------------------------------------------------------------

source1_file = TEST_DIR / "test_source1.tsv"

reader = pd.read_csv(
    source1_file,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK_SIZE
)

results = []

total_rows = 0
total_candidates = 0
total_matches = 0

for chunk_number, df in enumerate(reader, start=1):

    for _, row in df.iterrows():

        s1_id = row["entity_id"]

        country = (
            str(row["country"])
            .lower()
            .strip()
        )

        name = normalize_text(
            row["business_name"]
        )

        address = normalize_text(
            row["business_address"]
        )

        candidate_positions = set()

        # Exact normalized name
        candidate_positions.update(
            name_index.get(
                (country, name),
                []
            )
        )

        # Exact normalized address
        candidate_positions.update(
            address_index.get(
                (country, address),
                []
            )
        )

        total_candidates += len(candidate_positions)

        for pos in candidate_positions:

            candidate = reference.iloc[pos]

            results.append({
                "source1_entity_id": s1_id,
                "matched_entity_id": candidate["entity_id"]
            })

            total_matches += 1

    total_rows += len(df)

    elapsed = time.time() - start_time

    print(
        f"Chunk {chunk_number:,} | "
        f"Source1 rows: {total_rows:,} | "
        f"Candidates: {total_candidates:,} | "
        f"Matches: {total_matches:,} | "
        f"Elapsed: {elapsed / 60:.2f} min"
    )

# ------------------------------------------------------------
# CREATE SUBMISSION
# ------------------------------------------------------------

print()
print("Creating submission...")

matches = pd.DataFrame(results)

# ------------------------------------------------------------
# LOAD ALL TEST SOURCE1 IDS
# ------------------------------------------------------------

test_source1 = pd.read_csv(
    source1_file,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    usecols=["entity_id"]
)

test_source1 = test_source1.rename(
    columns={
        "entity_id": "source1_entity_id"
    }
)

# ------------------------------------------------------------
# GROUP MATCHES
# ------------------------------------------------------------

if len(matches) > 0:

    grouped = (
        matches
        .groupby("source1_entity_id")["matched_entity_id"]
        .apply(
            lambda x: ",".join(
                dict.fromkeys(
                    x.astype(str)
                )
            )
        )
        .reset_index()
    )

    grouped.columns = [
        "source1_entity_id",
        "matched_entity_ids"
    ]

else:

    grouped = pd.DataFrame(
        columns=[
            "source1_entity_id",
            "matched_entity_ids"
        ]
    )

# ------------------------------------------------------------
# MERGE ALL SOURCE1 IDS
# ------------------------------------------------------------

submission = test_source1.merge(
    grouped,
    on="source1_entity_id",
    how="left"
)

submission["matched_entity_ids"] = (
    submission["matched_entity_ids"]
    .fillna("")
)

# ------------------------------------------------------------
# WRITE SUBMISSION
# ------------------------------------------------------------

submission.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# FINAL REPORT
# ------------------------------------------------------------

print()
print("=" * 70)
print("QUICK PREDICTION COMPLETE")
print("=" * 70)

print(
    f"Test Source1 rows: "
    f"{len(submission):,}"
)

print(
    f"Candidates checked: "
    f"{total_candidates:,}"
)

print(
    f"Matches: "
    f"{total_matches:,}"
)

print(
    f"Source1 with matches: "
    f"{(submission['matched_entity_ids'] != '').sum():,}"
)

print(
    f"Submission: "
    f"{OUTPUT_FILE}"
)

print(
    f"Total time: "
    f"{(time.time() - start_time) / 60:.2f} minutes"
)

print("=" * 70)