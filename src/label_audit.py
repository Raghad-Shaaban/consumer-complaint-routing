
"""Audit product labels in the cleaned CFPB complaint dataset."""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Supports running without a graphical display.
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "clean.parquet"
REPORT_DIR = PROJECT_ROOT / "reports"

TEXT_COL = "consumer_complaint_narrative"
TARGET_COL = "product"
ID_COL = "complaint_id"


# ---------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------

def normalize_text(value):
    """Normalize text to identify exact-text label conflicts."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def json_default(value):
    """Convert NumPy/Pandas scalar values to JSON-compatible values."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)


# ---------------------------------------------------------
# 3. Audit
# ---------------------------------------------------------

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run src/deduplicate_split.py first."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_FILE)

    required = {TEXT_COL, TARGET_COL}
    missing_columns = required - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Input dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("The cleaned dataset is empty.")

    print(f"Loaded {len(df):,} rows from {INPUT_FILE.relative_to(PROJECT_ROOT)}")

    # -----------------------------------------------------
    # Label distribution
    # -----------------------------------------------------

    labels = df[TARGET_COL].astype("string").str.strip()
    missing_label_mask = labels.isna() | labels.eq("")

    label_counts = labels[~missing_label_mask].value_counts()
    label_percentages = (
        labels[~missing_label_mask].value_counts(normalize=True) * 100
    )

    audit_summary = pd.DataFrame({
        TARGET_COL: label_counts.index,
        "count": label_counts.values,
        "percentage": [
            round(float(label_percentages[label]), 2)
            for label in label_counts.index
        ],
    })

    audit_summary.to_csv(
        REPORT_DIR / "label_audit_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Exact-text conflicting labels
    # -----------------------------------------------------

    working = df[[TEXT_COL, TARGET_COL]].copy()
    working["_normalized_text"] = working[TEXT_COL].map(normalize_text)
    working["_normalized_label"] = (
        working[TARGET_COL].astype("string").str.strip()
    )

    usable = working[
        working["_normalized_text"].ne("")
        & working["_normalized_label"].notna()
        & working["_normalized_label"].ne("")
    ]

    labels_per_text = usable.groupby("_normalized_text")[
        "_normalized_label"
    ].nunique()

    conflicting_texts = set(
        labels_per_text[labels_per_text > 1].index
    )

    conflict_rows = df.loc[
        working["_normalized_text"].isin(conflicting_texts)
    ].copy()

    # Avoid exporting the full complaint narratives into a report.
    conflict_columns = [c for c in [ID_COL, TARGET_COL] if c in df.columns]
    conflict_output = conflict_rows[conflict_columns].copy()

    conflict_output.insert(
        0,
        "normalized_text_group",
        conflict_rows[TEXT_COL].map(normalize_text).map(
            lambda text: str(abs(hash(text)))
        ).values,
    )

    # Python's hash can vary between processes; use a stable group key instead.
    import hashlib

    conflict_output["normalized_text_group"] = (
        conflict_rows[TEXT_COL]
        .map(normalize_text)
        .map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])
        .values
    )

    conflict_output.to_csv(
        REPORT_DIR / "conflicting_label_groups.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Missing labels and duplicate statistics
    # -----------------------------------------------------

    normalized_text = df[TEXT_COL].map(normalize_text)
    duplicate_text_rows = int(
        normalized_text[normalized_text.ne("")].duplicated().sum()
    )

    # -----------------------------------------------------
    # Chart
    # -----------------------------------------------------

    if not audit_summary.empty:
        fig_height = max(5, 0.45 * len(audit_summary))
        fig, ax = plt.subplots(figsize=(12, fig_height))

        ax.barh(
            audit_summary[TARGET_COL],
            audit_summary["count"],
        )
        ax.invert_yaxis()
        ax.set_title("Distribution of Complaint Product Labels")
        ax.set_xlabel("Number of complaints")
        ax.set_ylabel("Product category")

        fig.tight_layout()
        fig.savefig(
            REPORT_DIR / "label_distribution.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    report = {
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "total_rows": int(len(df)),
        "unique_labels": int(labels[~missing_label_mask].nunique()),
        "missing_or_empty_labels": int(missing_label_mask.sum()),
        "empty_or_missing_narratives": int(
            df[TEXT_COL].isna().sum()
            + df[TEXT_COL].fillna("").astype(str).str.strip().eq("").sum()
            - (df[TEXT_COL].isna() & df[TEXT_COL].fillna("").astype(str).str.strip().eq("")).sum()
        ),
        "exact_duplicate_narrative_rows": duplicate_text_rows,
        "conflicting_exact_narrative_groups": int(len(conflicting_texts)),
        "rows_in_conflicting_groups": int(len(conflict_rows)),
        "label_distribution": audit_summary.to_dict(orient="records"),
        "historical_label_mapping_applied": False,
        "notes": [
            "This script audits labels; it does not rename or harmonize categories.",
            "Conflicting exact-text groups require investigation before any label correction.",
            "Near-duplicate semantic narratives are not checked by this audit.",
        ],
    }

    (REPORT_DIR / "label_audit_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )

    print("\nLabel audit completed.")
    print(f"Rows: {len(df):,}")
    print(f"Distinct labels: {report['unique_labels']}")
    print(f"Missing/empty labels: {report['missing_or_empty_labels']}")
    print(f"Conflicting exact-text groups: {report['conflicting_exact_narrative_groups']}")
    print(f"Report directory: {REPORT_DIR}")


if __name__ == "__main__":
    main()

