from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FEATURE_FILE = PROCESSED_DIR / "candidate_features.tsv"
MODEL_FILE = MODEL_DIR / "random_forest_model.joblib"
OUTPUT_FILE = OUTPUT_DIR / "scored_candidates.tsv"

CHUNK_SIZE = 100_000
THRESHOLD = 0.85

FEATURES = [
    "name_ratio",
    "name_token_ratio",
    "address_ratio",
    "address_token_ratio",
    "name_exact",
    "address_exact",
    "country_match",
]

print("=" * 70)
print("RANDOM FOREST CANDIDATE SCORING")
print("=" * 70)

if not FEATURE_FILE.exists():
    raise FileNotFoundError(FEATURE_FILE)

if not MODEL_FILE.exists():
    raise FileNotFoundError(MODEL_FILE)

OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading model...")
bundle = joblib.load(MODEL_FILE)

model = bundle["model"]
features = bundle["features"]
threshold = bundle.get("threshold", THRESHOLD)

print(f"Model: {MODEL_FILE}")
print(f"Threshold: {threshold}")
print(f"Features: {features}")

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

total_rows = 0
total_matches = 0
start_time = time.time()
first_write = True

print()
print("Scoring candidates...")
print(f"Feature file: {FEATURE_FILE}")
print(f"Output file:  {OUTPUT_FILE}")
print(f"Chunk size:   {CHUNK_SIZE:,}")
print()

for chunk_number, chunk in enumerate(
    pd.read_csv(
        FEATURE_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
        usecols=[
            "source1_id",
            "candidate_id",
            "source",
            *FEATURES,
        ],
    ),
    start=1,
):

    chunk_start = time.time()

    X = chunk[FEATURES].astype(np.float32)

    probabilities = model.predict_proba(X)[:, 1]

    chunk["match_probability"] = probabilities

    # Keep only predictions meeting the required threshold.
    matches = chunk[
        chunk["match_probability"] >= threshold
    ].copy()

    if len(matches) > 0:
        matches = matches[
            [
                "source1_id",
                "candidate_id",
                "source",
                "match_probability",
            ]
        ]

        matches.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False
        total_matches += len(matches)

    total_rows += len(chunk)

    elapsed = time.time() - start_time
    chunk_time = time.time() - chunk_start

    print(
        f"Chunk {chunk_number:,} | "
        f"Scored: {len(chunk):,} | "
        f"Matches: {len(matches):,} | "
        f"Total scored: {total_rows:,} | "
        f"Total matches: {total_matches:,} | "
        f"Chunk: {chunk_time:.2f}s | "
        f"Elapsed: {elapsed / 60:.2f} min"
    )

print()
print("=" * 70)
print("SCORING COMPLETE")
print("=" * 70)
print(f"Total candidates scored: {total_rows:,}")
print(f"Matches at threshold {threshold}: {total_matches:,}")
print(f"Output: {OUTPUT_FILE}")
print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")
print("=" * 70)