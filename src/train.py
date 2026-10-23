from pathlib import Path
import sys
import json
import pandas as pd
import joblib
from features import build_pipeline
PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data/processed/clean.parquet"
SPLIT_PATH = PROJECT / "data/processed/split_ids.csv"
MODEL_PATH = PROJECT / "models/model.joblib"

TEXT_COL = "consumer_complaint_narrative"
LABEL_COL = "product"
ID_COL = "complaint_id"
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Missing clean data: {DATA_PATH}"
    )

if not SPLIT_PATH.exists():
    raise FileNotFoundError(
        f"Missing split file: {SPLIT_PATH}"
    )

df = pd.read_parquet(DATA_PATH)
splits = pd.read_csv(SPLIT_PATH)

required = {ID_COL, TEXT_COL, LABEL_COL}

if not required.issubset(df.columns):
    raise ValueError("Missing required data columns")

if not {ID_COL, "split"}.issubset(splits.columns):
    raise ValueError(
        "Split file must contain complaint_id and split columns"
    )

if df[ID_COL].duplicated().any():
    raise ValueError("Duplicate complaint IDs in clean data")

if splits[ID_COL].duplicated().any():
    raise ValueError("Duplicate complaint IDs in split file")

df[ID_COL] = df[ID_COL].astype(str)
splits[ID_COL] = splits[ID_COL].astype(str)

if not splits["split"].isin(["train", "test"]).all():
    raise ValueError("Expected train/test split values")

if set(df[ID_COL]) != set(splits[ID_COL]):
    raise ValueError("Split IDs do not match clean data IDs")

df = df.merge(
    splits[[ID_COL, "split"]],
    on=ID_COL,
    how="inner",
    validate="one_to_one"
)

train = df[df["split"] == "train"]

if train.empty:
    raise ValueError("Training set is empty")

if train[TEXT_COL].isna().any():
    raise ValueError("Missing training texts")

if train[LABEL_COL].isna().any():
    raise ValueError("Missing training labels")

X_train = train[TEXT_COL].astype(str)
y_train = train[LABEL_COL]

model = build_pipeline()

model.set_params(
    classifier__class_weight="balanced"
)

print("Training samples:", len(train))
print("Training model...")

model.fit(X_train, y_train)

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(model, MODEL_PATH)

print("Model saved successfully!")
print("Location:", MODEL_PATH)
    train_model()
# Model training script
