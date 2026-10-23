
"""Evaluate the trained CFPB complaint routing model."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "clean.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "evaluation_report.json"

TEXT_COL = "consumer_complaint_narrative"
LABEL_COL = "product"
SPLIT_COL = "split"


def evaluate_model():
    """Evaluate the saved model on the held-out test split."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run src/train.py first."
        )

    df = pd.read_parquet(DATA_PATH)

    required = {TEXT_COL, LABEL_COL, SPLIT_COL}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    if not df[SPLIT_COL].isin(["train", "test"]).all():
        raise ValueError("Split column must contain only 'train' and 'test'.")

    test = df.loc[df[SPLIT_COL] == "test"].copy()

    if test.empty:
        raise ValueError("Test split is empty.")

    if test[TEXT_COL].isna().any():
        raise ValueError("Test data contains missing narratives.")

    if test[LABEL_COL].isna().any():
        raise ValueError("Test data contains missing product labels.")

    X_test = test[TEXT_COL].astype(str)
    y_test = test[LABEL_COL].astype(str)

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    # Include every class present in either the true or predicted labels.
    labels = sorted(set(y_test) | set(predictions))

    report = {
        "test_samples": int(len(test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "labels": labels,
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        ).tolist(),
        "evaluation_split": "test",
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"Test samples: {report['test_samples']:,}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro-F1: {report['macro_f1']:.4f}")
    print(f"Evaluation report saved to: {REPORT_PATH}")

    return report


if __name__ == "__main__":
    evaluate_model()
