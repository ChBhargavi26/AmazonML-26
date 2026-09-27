from pathlib import Path
import pandas as pd
import numpy as np
import time
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"
MODEL_DIR = PROJECT_ROOT / "models"

FEATURE_FILE = PROCESSED_DIR / "candidate_features.tsv"
GROUND_TRUTH_FILE = RAW_TRAIN_DIR / "train_ground_truth.tsv"
MODEL_FILE = MODEL_DIR / "random_forest_model.joblib"

FEATURES = [
    "name_ratio",
    "name_token_ratio",
    "address_ratio",
    "address_token_ratio",
    "name_exact",
    "address_exact",
    "country_match",
]

RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_TRAIN_ROWS = 2_000_000


print("=" * 70)
print("RANDOM FOREST MODEL TRAINING")
print("=" * 70)

if not FEATURE_FILE.exists():
    raise FileNotFoundError(FEATURE_FILE)

if not GROUND_TRUTH_FILE.exists():
    raise FileNotFoundError(GROUND_TRUTH_FILE)

MODEL_DIR.mkdir(exist_ok=True)

print("Loading ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

print(f"Ground-truth rows: {len(gt):,}")

# Convert comma-separated matched IDs into a lookup set.
truth = {}

for _, row in gt.iterrows():
    source1_id = row["source1_entity_id"]
    matched = row["matched_entity_ids"]

    if matched:
        truth[source1_id] = set(
            x.strip() for x in matched.split(",") if x.strip()
        )
    else:
        truth[source1_id] = set()

print(f"Ground-truth entities: {len(truth):,}")

print()
print("Loading candidate features for training...")

train_parts = []
loaded_rows = 0

for chunk in pd.read_csv(
    FEATURE_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    usecols=[
        "source1_id",
        "candidate_id",
        "source",
        *FEATURES,
    ],
    chunksize=100_000,
):

    # Create positive/negative labels.
    labels = []

    for s1, candidate in zip(
        chunk["source1_id"],
        chunk["candidate_id"],
    ):
        labels.append(
            int(
                candidate
                in truth.get(s1, set())
            )
        )

    chunk["label"] = labels

    train_parts.append(chunk)
    loaded_rows += len(chunk)

    if loaded_rows >= MAX_TRAIN_ROWS:
        break

print(f"Training candidate rows loaded: {loaded_rows:,}")

train_df = pd.concat(
    train_parts,
    ignore_index=True,
)

positive = int(train_df["label"].sum())
negative = len(train_df) - positive

print(f"Positive matches: {positive:,}")
print(f"Negative matches: {negative:,}")

if positive == 0:
    raise RuntimeError(
        "No positive training examples were found."
    )

X = train_df[FEATURES].astype(np.float32)
y = train_df["label"].astype(np.int8)

print()
print("Splitting training data...")

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training rows:   {len(X_train):,}")
print(f"Validation rows: {len(X_valid):,}")

print()
print("Training Random Forest...")
print(f"Trees: {N_ESTIMATORS}")

start = time.time()

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced",
    max_features="sqrt",
)

model.fit(X_train, y_train)

elapsed = time.time() - start

print(f"Training time: {elapsed / 60:.2f} minutes")

print()
print("Validation...")

pred = model.predict(X_valid)

print(classification_report(
    y_valid,
    pred,
    digits=4,
))

print("Confusion matrix:")
print(confusion_matrix(y_valid, pred))

print()
print("Saving model...")

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "threshold": 0.85,
    },
    MODEL_FILE,
)

print(f"Model saved: {MODEL_FILE}")

print()
print("=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)