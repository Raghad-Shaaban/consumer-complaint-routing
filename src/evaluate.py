from pathlib import Path import json import pandas as pd import joblib
from sklearn.metrics import ( accuracy_score, f1_score, classification_report, confusion_matrix )
PROJECT = Path(file).resolve().parents[1]
DATA_PATH = PROJECT / "data/processed/clean.parquet" SPLIT_PATH = PROJECT / "data/processed/split_ids.csv" MODEL_PATH = PROJECT / "models/model.joblib" REPORT_PATH = PROJECT / "reports/evaluation_report.json"
ID_COL = "complaint_id" TEXT_COL = "consumer_complaint_narrative" LABEL_COL = "product"
def evaluate_model():
if not DATA_PATH.exists():
    raise FileNotFoundError("clean.parquet is missing")

if not SPLIT_PATH.exists():
    raise FileNotFoundError("split_ids.csv is missing")

if not MODEL_PATH.exists():
    raise FileNotFoundError("model.joblib is missing")

df = pd.read_parquet(DATA_PATH)
splits = pd.read_csv(SPLIT_PATH)

required_data = {ID_COL, TEXT_COL, LABEL_COL}
required_splits = {ID_COL, "split"}

if not required_data.issubset(df.columns):
    raise ValueError("Missing required data columns")

if not required_splits.issubset(splits.columns):
    raise ValueError("Missing required split columns")

df[ID_COL] = df[ID_COL].astype(str)
splits[ID_COL] = splits[ID_COL].astype(str)

if df[ID_COL].duplicated().any():
    raise ValueError("Duplicate complaint IDs")

if splits[ID_COL].duplicated().any():
    raise ValueError("Duplicate split IDs")

if set(df[ID_COL]) != set(splits[ID_COL]):
    raise ValueError("Split IDs do not match clean data")

if not splits["split"].isin(["train", "test"]).all():
    raise ValueError("Invalid split values")

df = df.merge(
    splits[[ID_COL, "split"]],
    on=ID_COL,
    how="inner",
    validate="one_to_one"
)

test = df[df["split"] == "test"]

if test.empty:
    raise ValueError("Test set is empty")

model = joblib.load(MODEL_PATH)

X_test = test[TEXT_COL].astype(str)
y_test = test[LABEL_COL]

predictions = model.predict(X_test)

labels = sorted(set(y_test) | set(predictions))

report = {
    "test_samples": len(test),
    "accuracy": accuracy_score(y_test, predictions),
    "macro_f1": f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    ),
    "classification_report": classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0
    ),
    "labels": labels,
    "confusion_matrix": confusion_matrix(
        y_test,
        predictions,
        labels=labels
    ).tolist()
}

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("Accuracy:", report["accuracy"])
print("Macro-F1:", report["macro_f1"])
print("Evaluation report saved!")
if name == "main": evaluate_model()
#Model evaluation script
