
"""Train the baseline CFPB complaint routing model."""

from pathlib import Path

import joblib
import pandas as pd

from features import build_pipeline


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "clean.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

TEXT_COL = "consumer_complaint_narrative"
LABEL_COL = "product"
SPLIT_COL = "split"


def train_model():
    """Train the model using only the training split."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {DATA_PATH}\n"
            "Run src/deduplicate_split.py first."
        )

    df = pd.read_parquet(DATA_PATH)

    required = {TEXT_COL, LABEL_COL, SPLIT_COL}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Clean dataset is empty.")

    if not df[SPLIT_COL].isin(["train", "test"]).all():
        raise ValueError("Split column must contain only 'train' and 'test'.")

    train = df.loc[df[SPLIT_COL] == "train"].copy()

    if train.empty:
        raise ValueError("Training split is empty.")

    if train[TEXT_COL].isna().any():
        raise ValueError("Training data contains missing narratives.")

    if train[LABEL_COL].isna().any():
        raise ValueError("Training data contains missing product labels.")

    X_train = train[TEXT_COL].astype(str)
    y_train = train[LABEL_COL].astype(str)

    if y_train.nunique() < 2:
        raise ValueError("Training data must contain at least two classes.")

    model = build_pipeline()

    print(f"Training samples: {len(train):,}")
    print(f"Product classes: {y_train.nunique()}")
    print("Training model...")

    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model saved successfully.")
    print(f"Location: {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_model()

