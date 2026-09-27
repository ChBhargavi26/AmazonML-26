from pathlib import Path
import pandas as pd
import re
import time
from difflib import SequenceMatcher


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = PROJECT_ROOT / "data" / "raw" / "test"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "submission.csv"

CHUNK_SIZE = 50_000

# Composite address keys with more than this many records
# are ignored to reduce false positives.
MAX_COMPOSITE_CANDIDATES = 20

STOPWORDS = {
    "the", "and", "for", "with", "from",
    "that", "this", "inc", "llc", "ltd",
    "co", "company", "road", "rd",
    "street", "st", "avenue", "ave",
    "boulevard", "blvd", "drive", "dr",
    "lane", "ln", "way", "highway", "hwy",
    "floor", "unit", "building", "bldg"
}


# ============================================================
# NORMALIZATION
# ============================================================

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


def tokenize(value):
    value = normalize_text(value)

    if not value:
        return []

    return [
        token
        for token in value.split()
        if len(token) >= 2
        and token not in STOPWORDS
    ]


# ============================================================
# ADDRESS BLOCK KEY
# ============================================================

def address_block_key(value):
    """
    Create a conservative address key:

        (first number, longest meaningful word)

    Example:
        4303 Elkins Avenue Unit B Nashville TN

    becomes approximately:

        ('4303', 'nashville')
    """

    normalized = normalize_text(value)

    if not normalized:
        return None

    tokens = normalized.split()

    numbers = [
        token
        for token in tokens
        if token.isdigit()
    ]

    words = [
        token
        for token in tokens
        if token.isalpha()
        and len(token) >= 5
        and token not in STOPWORDS
    ]

    if not numbers or not words:
        return None

    number = numbers[0]

    # Longest meaningful word.
    word = max(
        words,
        key=len
    )

    return number, word


# ============================================================
# SIMILARITY
# ============================================================

def sequence_similarity(a, b):
    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


def token_similarity(a, b):
    ta = set(tokenize(a))
    tb = set(tokenize(b))

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FAST HYBRID TEST PREDICTION")
print("=" * 70)

start_time = time.time()


# ============================================================
# LOAD SOURCE2 + SOURCE3
# ============================================================

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

print(
    f"Reference rows: {len(reference):,}"
)


# ============================================================
# NORMALIZE REFERENCE
# ============================================================

print("Normalizing reference data...")

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


# ============================================================
# BUILD EXACT NAME INDEX
# ============================================================

print("Building exact-name index...")

name_index = {}

for pos, row in enumerate(
    reference[
        ["country_norm", "name_norm"]
    ].itertuples(index=False)
):

    country, name = row

    if not name:
        continue

    key = (
        country,
        name
    )

    name_index.setdefault(
        key,
        []
    ).append(pos)


print(
    f"Exact-name keys: {len(name_index):,}"
)


# ============================================================
# BUILD EXACT ADDRESS INDEX
# ============================================================

print("Building exact-address index...")

address_index = {}

for pos, row in enumerate(
    reference[
        ["country_norm", "address_norm"]
    ].itertuples(index=False)
):

    country, address = row

    if not address:
        continue

    key = (
        country,
        address
    )

    address_index.setdefault(
        key,
        []
    ).append(pos)


print(
    f"Exact-address keys: {len(address_index):,}"
)


# ============================================================
# BUILD SELECTIVE ADDRESS COMPOSITE INDEX
# ============================================================

print("Building address composite index...")

address_block_index = {}

for pos, row in enumerate(
    reference[
        ["country_norm", "address_norm"]
    ].itertuples(index=False)
):

    country, address = row

    key = address_block_key(address)

    if key is None:
        continue

    number, word = key

    full_key = (
        country,
        number,
        word
    )

    address_block_index.setdefault(
        full_key,
        []
    ).append(pos)


# Remove very large blocks.
address_block_index = {
    key: positions
    for key, positions
    in address_block_index.items()
    if len(positions) <= MAX_COMPOSITE_CANDIDATES
}

print(
    f"Usable composite keys: "
    f"{len(address_block_index):,}"
)


# ============================================================
# LOAD SOURCE1 IDS
# ============================================================

source1_file = (
    TEST_DIR / "test_source1.tsv"
)

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


# ============================================================
# PROCESS SOURCE1
# ============================================================

print()
print("Processing Source1...")

reader = pd.read_csv(
    source1_file,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK_SIZE
)

results = []

total_rows = 0
total_exact_name = 0
total_exact_address = 0
total_composite = 0

chunk_start_time = time.time()


for chunk_number, df in enumerate(
    reader,
    start=1
):

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

        matched_ids = set()


        # ----------------------------------------------------
        # 1. EXACT NAME
        # ----------------------------------------------------

        exact_name_positions = name_index.get(
            (
                country,
                name
            ),
            []
        )

        if exact_name_positions:

            for pos in exact_name_positions:

                matched_ids.add(
                    reference.iloc[pos]["entity_id"]
                )

            total_exact_name += len(
                exact_name_positions
            )


        # ----------------------------------------------------
        # 2. EXACT ADDRESS
        # ----------------------------------------------------

        exact_address_positions = address_index.get(
            (
                country,
                address
            ),
            []
        )

        if exact_address_positions:

            for pos in exact_address_positions:

                matched_ids.add(
                    reference.iloc[pos]["entity_id"]
                )

            total_exact_address += len(
                exact_address_positions
            )


        # ----------------------------------------------------
        # 3. SELECTIVE ADDRESS COMPOSITE
        # ----------------------------------------------------

        block_key = address_block_key(
            address
        )

        if block_key is not None:

            number, word = block_key

            positions = address_block_index.get(
                (
                    country,
                    number,
                    word
                ),
                []
            )

            for pos in positions:

                candidate = reference.iloc[pos]

                candidate_name = (
                    candidate["name_norm"]
                )

                candidate_address = (
                    candidate["address_norm"]
                )

                # Address similarity
                address_ratio = sequence_similarity(
                    address,
                    candidate_address
                )

                address_token_ratio = token_similarity(
                    address,
                    candidate_address
                )

                # Name similarity
                name_ratio = sequence_similarity(
                    name,
                    candidate_name
                )

                name_token_ratio = token_similarity(
                    name,
                    candidate_name
                )

                # Conservative acceptance.
                #
                # The composite block already shares:
                # country + address number + word.
                #
                # We additionally require reasonable
                # address OR name similarity.
                if (
                    address_ratio >= 0.45
                    or address_token_ratio >= 0.30
                    or name_ratio >= 0.55
                    or name_token_ratio >= 0.30
                ):

                    matched_ids.add(
                        candidate["entity_id"]
                    )

                    total_composite += 1


        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        if matched_ids:

            results.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(
                    sorted(matched_ids)
                )
            })

        else:

            results.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ""
            })


    total_rows += len(df)

    elapsed = (
        time.time() - chunk_start_time
    )

    print(
        f"Chunk {chunk_number:02d} | "
        f"Source1 processed: {total_rows:,} | "
        f"Elapsed: {elapsed / 60:.2f} min"
    )


# ============================================================
# CREATE SUBMISSION
# ============================================================

print()
print("Creating submission...")

submission = pd.DataFrame(
    results
)

# Ensure every Source1 row appears exactly once.
submission = test_source1.merge(
    submission,
    on="source1_entity_id",
    how="left"
)

submission["matched_entity_ids"] = (
    submission["matched_entity_ids"]
    .fillna("")
)


# ============================================================
# WRITE
# ============================================================

submission.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

nonempty = (
    submission["matched_entity_ids"]
    .astype(str)
    .str.len()
    .gt(0)
    .sum()
)

empty = (
    len(submission) - nonempty
)

print()
print("=" * 70)
print("FAST HYBRID TEST PREDICTION COMPLETE")
print("=" * 70)

print(
    f"Test Source1 rows: {len(submission):,}"
)

print(
    f"Rows with matches: {nonempty:,}"
)

print(
    f"Rows without matches: {empty:,}"
)

print(
    f"Exact-name matches: {total_exact_name:,}"
)

print(
    f"Exact-address matches: {total_exact_address:,}"
)

print(
    f"Composite matches: {total_composite:,}"
)

print(
    f"Output: {OUTPUT_FILE}"
)

print(
    f"Total time: "
    f"{(time.time() - start_time) / 60:.2f} minutes"
)

print("=" * 70)