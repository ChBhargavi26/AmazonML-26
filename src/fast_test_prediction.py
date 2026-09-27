from pathlib import Path
import pandas as pd
import numpy as np
import re
import joblib
import time
from difflib import SequenceMatcher


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = PROJECT_ROOT / "data" / "raw" / "test"
MODEL_FILE = PROJECT_ROOT / "models" / "random_forest_model.joblib"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "submission.csv"

CHUNK_SIZE = 50_000
PREDICTION_BATCH_SIZE = 25_000
THRESHOLD = 0.85

MAX_ADDRESS_BLOCK = 50

STOPWORDS = {
    "the", "and", "for", "with", "from",
    "that", "this", "inc", "llc", "ltd",
    "co", "company"
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
        x
        for x in value.split()
        if len(x) >= 2
        and x not in STOPWORDS
    ]


def similarity(a, b):
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


def address_block_key(value):
    """
    Fast address block:
    first number + first meaningful long word.
    """

    if not value:
        return None

    tokens = str(value).split()

    number = None
    word = None

    for token in tokens:

        if number is None and token.isdigit():
            number = token
            continue

        if (
            word is None
            and token.isalpha()
            and len(token) >= 5
            and token not in STOPWORDS
        ):
            word = token

        if number is not None and word is not None:
            break

    if number is None or word is None:
        return None

    return number, word


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FAST RF TEST PREDICTION - OPTIMIZED")
print("=" * 70)

start_time = time.time()


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading Random Forest model...")

bundle = joblib.load(
    MODEL_FILE
)

model = bundle["model"]

if "threshold" in bundle:
    THRESHOLD = float(
        bundle["threshold"]
    )

try:
    model.set_params(
        n_jobs=-1
    )
except Exception:
    pass

print("Model loaded.")
print(
    f"Threshold: {THRESHOLD}"
)
print(
    f"Trees: {model.n_estimators}"
)


# ============================================================
# LOAD TEST SOURCE2 + SOURCE3
# ============================================================

print()
print("Loading test Source2 + Source3...")

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

print()
print("Normalizing reference...")

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

print("Normalization complete.")


# ============================================================
# FAST EXACT NAME INDEX
# ============================================================

print()
print("Building exact-name index...")

name_index = (
    reference
    .reset_index()
    .groupby(
        [
            "country_norm",
            "name_norm"
        ],
        sort=False
    )["index"]
    .agg(list)
    .to_dict()
)

# Remove empty names
name_index = {
    key: value
    for key, value in name_index.items()
    if key[1]
}

print(
    f"Exact-name keys: {len(name_index):,}"
)


# ============================================================
# FAST FIRST-TOKEN INDEX
# ============================================================

print("Building name-token index...")

# Only use first meaningful token.
# This keeps the index compact and fast.

first_tokens = []

for value in reference["name_norm"].values:

    tokens = [
        x
        for x in value.split()
        if len(x) >= 2
        and x not in STOPWORDS
    ]

    if tokens:
        first_tokens.append(tokens[0])
    else:
        first_tokens.append("")


reference["first_name_token"] = first_tokens

token_index = (
    reference
    .reset_index()
    .groupby(
        [
            "country_norm",
            "first_name_token"
        ],
        sort=False
    )["index"]
    .agg(list)
    .to_dict()
)

token_index = {
    key: value
    for key, value in token_index.items()
    if key[1]
}

print(
    f"Token keys: {len(token_index):,}"
)


# ============================================================
# FAST EXACT ADDRESS INDEX
# ============================================================

print("Building exact-address index...")

address_index = (
    reference
    .reset_index()
    .groupby(
        [
            "country_norm",
            "address_norm"
        ],
        sort=False
    )["index"]
    .agg(list)
    .to_dict()
)

address_index = {
    key: value
    for key, value in address_index.items()
    if key[1]
}

print(
    f"Exact-address keys: {len(address_index):,}"
)


# ============================================================
# ADDRESS BLOCK INDEX
# ============================================================

print("Building address block index...")

address_keys = (
    reference["address_norm"]
    .map(address_block_key)
)

reference["address_number"] = (
    address_keys.map(
        lambda x: x[0]
        if x is not None
        else ""
    )
)

reference["address_word"] = (
    address_keys.map(
        lambda x: x[1]
        if x is not None
        else ""
    )
)

address_block_index = (
    reference
    .reset_index()
    .groupby(
        [
            "country_norm",
            "address_number",
            "address_word"
        ],
        sort=False
    )["index"]
    .agg(list)
    .to_dict()
)

# Keep only selective blocks.
address_block_index = {
    key: value
    for key, value in address_block_index.items()
    if key[1]
    and key[2]
    and len(value) <= MAX_ADDRESS_BLOCK
}

print(
    f"Usable address blocks: "
    f"{len(address_block_index):,}"
)


# ============================================================
# LOAD SOURCE1
# ============================================================

source1_file = (
    TEST_DIR / "test_source1.tsv"
)

reader = pd.read_csv(
    source1_file,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK_SIZE
)


# ============================================================
# PROCESS SOURCE1
# ============================================================

results = []

total_rows = 0
total_candidates = 0
total_matches = 0

prediction_start = time.time()


for chunk_number, df in enumerate(
    reader,
    start=1
):

    feature_rows = []
    candidate_info = []

    # --------------------------------------------------------
    # NORMALIZE SOURCE1 CHUNK
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
    # PROCESS ROWS
    # --------------------------------------------------------

    for row in df.itertuples(
        index=False
    ):

        s1_id = row.entity_id

        country = row.country_norm
        name = row.name_norm
        address = row.address_norm

        candidate_positions = set()

        # ----------------------------------------------------
        # EXACT NAME
        # ----------------------------------------------------

        candidate_positions.update(
            name_index.get(
                (
                    country,
                    name
                ),
                []
            )
        )

        # ----------------------------------------------------
        # FIRST NAME TOKEN
        # ----------------------------------------------------

        tokens = [
            x
            for x in name.split()
            if len(x) >= 2
            and x not in STOPWORDS
        ]

        if tokens:

            candidate_positions.update(
                token_index.get(
                    (
                        country,
                        tokens[0]
                    ),
                    []
                )
            )

        # ----------------------------------------------------
        # EXACT ADDRESS
        # ----------------------------------------------------

        candidate_positions.update(
            address_index.get(
                (
                    country,
                    address
                ),
                []
            )
        )

        # ----------------------------------------------------
        # ADDRESS BLOCK
        # ----------------------------------------------------

        block = address_block_key(
            address
        )

        if block is not None:

            number, word = block

            candidate_positions.update(
                address_block_index.get(
                    (
                        country,
                        number,
                        word
                    ),
                    []
                )
            )

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        for pos in candidate_positions:

            candidate = reference.iloc[pos]

            candidate_name = (
                candidate["name_norm"]
            )

            candidate_address = (
                candidate["address_norm"]
            )

            name_ratio = similarity(
                name,
                candidate_name
            )

            name_token_ratio = (
                token_similarity(
                    name,
                    candidate_name
                )
            )

            address_ratio = similarity(
                address,
                candidate_address
            )

            address_token_ratio = (
                token_similarity(
                    address,
                    candidate_address
                )
            )

            name_exact = int(
                bool(name)
                and name == candidate_name
            )

            address_exact = int(
                bool(address)
                and address == candidate_address
            )

            country_match = int(
                country
                == candidate["country_norm"]
            )

            feature_rows.append([
                name_ratio,
                name_token_ratio,
                address_ratio,
                address_token_ratio,
                name_exact,
                address_exact,
                country_match
            ])

            candidate_info.append([
                s1_id,
                candidate["entity_id"]
            ])

    # --------------------------------------------------------
    # RF PREDICTION
    # --------------------------------------------------------

    candidate_count = len(
        feature_rows
    )

    total_candidates += candidate_count

    if candidate_count:

        features = np.asarray(
            feature_rows,
            dtype=np.float32
        )

        info = np.asarray(
            candidate_info,
            dtype=object
        )

        for batch_start in range(
            0,
            candidate_count,
            PREDICTION_BATCH_SIZE
        ):

            batch_end = min(
                batch_start
                + PREDICTION_BATCH_SIZE,
                candidate_count
            )

            probabilities = (
                model.predict_proba(
                    features[
                        batch_start:batch_end
                    ]
                )[:, 1]
            )

            matched_indices = (
                np.flatnonzero(
                    probabilities
                    >= THRESHOLD
                )
            )

            for idx in matched_indices:

                actual_idx = (
                    batch_start + idx
                )

                results.append({
                    "source1_entity_id":
                        info[
                            actual_idx,
                            0
                        ],

                    "matched_entity_id":
                        info[
                            actual_idx,
                            1
                        ]
                })

                total_matches += 1

    total_rows += len(df)

    elapsed = (
        time.time()
        - prediction_start
    )

    print(
        f"Chunk {chunk_number:02d} | "
        f"Source1 rows: {total_rows:,} | "
        f"Candidates: {total_candidates:,} | "
        f"Matches: {total_matches:,} | "
        f"Elapsed: {elapsed / 60:.2f} min"
    )


# ============================================================
# CREATE SUBMISSION
# ============================================================

print()
print("Creating submission...")

test_ids = pd.read_csv(
    source1_file,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    usecols=["entity_id"]
)

test_ids = test_ids.rename(
    columns={
        "entity_id":
            "source1_entity_id"
    }
)

matches = pd.DataFrame(
    results
)

if len(matches) > 0:

    grouped = (
        matches
        .groupby(
            "source1_entity_id"
        )["matched_entity_id"]
        .apply(
            lambda x:
                ",".join(
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


submission = test_ids.merge(
    grouped,
    on="source1_entity_id",
    how="left"
)

submission["matched_entity_ids"] = (
    submission["matched_entity_ids"]
    .fillna("")
)


# ============================================================
# WRITE OUTPUT
# ============================================================

submission.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

nonempty = (
    submission[
        "matched_entity_ids"
    ]
    .astype(str)
    .str.len()
    .gt(0)
    .sum()
)

empty = (
    len(submission)
    - nonempty
)

print()
print("=" * 70)
print("FAST RF TEST PREDICTION COMPLETE")
print("=" * 70)

print(
    f"Source1 rows: {len(submission):,}"
)

print(
    f"Candidates: {total_candidates:,}"
)

print(
    f"Matches: {total_matches:,}"
)

print(
    f"S1 with matches: {nonempty:,}"
)

print(
    f"S1 empty: {empty:,}"
)

print(
    f"Output: {OUTPUT_FILE}"
)

print(
    f"Total time: "
    f"{(time.time() - start_time) / 60:.2f} minutes"
)

print("=" * 70)