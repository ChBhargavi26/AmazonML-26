from pathlib import Path
import pandas as pd
import numpy as np
import time
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CANDIDATE_FILE = PROCESSED_DIR / "candidate_pairs.tsv"
OUTPUT_FILE = PROCESSED_DIR / "candidate_features.tsv"

CHUNK_SIZE = 50_000

SOURCE_FILES = {
    "source1": TRAIN_DIR / "train_source1_cleaned.tsv",
    "source2": TRAIN_DIR / "train_source2_cleaned.tsv",
    "source3": TRAIN_DIR / "train_source3_cleaned.tsv",
}


def normalize(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def token_set(value):
    value = normalize(value)
    return set(value.split()) if value else set()


def ratio(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def token_ratio(a, b):
    ta = token_set(a)
    tb = token_set(b)

    if not ta or not tb:
        return 0.0

    sa = " ".join(sorted(ta))
    sb = " ".join(sorted(tb))

    return SequenceMatcher(None, sa, sb).ratio()


print("=" * 70)
print("FEATURE GENERATION")
print("=" * 70)

if not CANDIDATE_FILE.exists():
    raise FileNotFoundError(
        f"Candidate file not found: {CANDIDATE_FILE}"
    )

print("Loading source data...")

source_data = {}

for source, path in SOURCE_FILES.items():
    print(f"Loading {source}: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name_normalized",
            "business_address_normalized",
            "country",
        ],
    )

    df["country"] = df["country"].map(normalize)
    df = df.set_index("entity_id")

    source_data[source] = df

    print(f"  Records: {len(df):,}")

print()
print("Starting candidate feature generation...")
print(f"Candidate file: {CANDIDATE_FILE}")
print(f"Output file:    {OUTPUT_FILE}")
print(f"Chunk size:     {CHUNK_SIZE:,}")
print()

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

first_write = True
total_rows = 0
start_time = time.time()

for chunk_number, candidates in enumerate(
    pd.read_csv(
        CANDIDATE_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):

    chunk_start = time.time()

    # Get Source1 records for this candidate chunk.
    s1 = source_data["source1"].reindex(
        candidates["source1_id"]
    )

    s2_mask = candidates["source"].eq("source2")
    s3_mask = candidates["source"].eq("source3")

    features = pd.DataFrame(index=candidates.index)

    features["name_ratio"] = 0.0
    features["name_token_ratio"] = 0.0
    features["address_ratio"] = 0.0
    features["address_token_ratio"] = 0.0
    features["name_exact"] = 0
    features["address_exact"] = 0
    features["country_match"] = 0

    for source, mask in [
        ("source2", s2_mask),
        ("source3", s3_mask),
    ]:

        if not mask.any():
            continue

        # Use integer positions rather than boolean .loc
        # to avoid the pandas indexing error.
        positions = np.flatnonzero(mask.to_numpy())

        s1_chunk = s1.iloc[positions]

        candidate_ids = candidates.iloc[positions]["candidate_id"]

        target = source_data[source].reindex(candidate_ids)

        s1_names = s1_chunk["business_name_normalized"].tolist()
        s1_addresses = s1_chunk[
            "business_address_normalized"
        ].tolist()
        s1_countries = s1_chunk["country"].tolist()

        target_names = target[
            "business_name_normalized"
        ].tolist()
        target_addresses = target[
            "business_address_normalized"
        ].tolist()
        target_countries = target["country"].tolist()

        name_ratios = []
        name_token_ratios = []
        address_ratios = []
        address_token_ratios = []
        name_exact = []
        address_exact = []
        country_match = []

        for (
            n1,
            n2,
            a1,
            a2,
            c1,
            c2,
        ) in zip(
            s1_names,
            target_names,
            s1_addresses,
            target_addresses,
            s1_countries,
            target_countries,
        ):

            name_ratios.append(ratio(n1, n2))
            name_token_ratios.append(token_ratio(n1, n2))

            address_ratios.append(ratio(a1, a2))
            address_token_ratios.append(token_ratio(a1, a2))

            name_exact.append(
                int(n1 == n2 and n1 != "")
            )

            address_exact.append(
                int(a1 == a2 and a1 != "")
            )

            country_match.append(
                int(c1 == c2 and c1 != "")
            )

        features.iloc[positions, features.columns.get_loc("name_ratio")] = name_ratios
        features.iloc[positions, features.columns.get_loc("name_token_ratio")] = name_token_ratios
        features.iloc[positions, features.columns.get_loc("address_ratio")] = address_ratios
        features.iloc[positions, features.columns.get_loc("address_token_ratio")] = address_token_ratios
        features.iloc[positions, features.columns.get_loc("name_exact")] = name_exact
        features.iloc[positions, features.columns.get_loc("address_exact")] = address_exact
        features.iloc[positions, features.columns.get_loc("country_match")] = country_match

    output = pd.concat(
        [
            candidates[
                ["source1_id", "candidate_id", "source"]
            ],
            features,
        ],
        axis=1,
    )

    output.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        mode="w" if first_write else "a",
        header=first_write,
    )

    first_write = False
    total_rows += len(output)

    elapsed = time.time() - start_time
    chunk_time = time.time() - chunk_start

    print(
        f"Chunk {chunk_number:,} | "
        f"Rows: {len(output):,} | "
        f"Total: {total_rows:,} | "
        f"Chunk time: {chunk_time / 60:.2f} min | "
        f"Elapsed: {elapsed / 60:.2f} min"
    )

print()
print("=" * 70)
print("FEATURE GENERATION COMPLETE")
print("=" * 70)
print(f"Total feature rows: {total_rows:,}")
print(f"Output: {OUTPUT_FILE}")
print(
    f"Total time: {(time.time() - start_time) / 60:.2f} minutes"
)