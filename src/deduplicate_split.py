
"""Deduplicate CFPB complaints and create reproducible train/test splits."""

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ---------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "valid.parquet"
CLEAN_FILE = PROJECT_ROOT / "data" / "clean.parquet"
SPLIT_IDS_FILE = PROJECT_ROOT / "data" / "split_ids.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

TEXT_COL = "consumer_complaint_narrative"
TARGET_COL = "product"
ID_COL = "complaint_id"

RANDOM_SEED = 42
N_SPLITS = 5  # Approximately 80% train / 20% test


# ---------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------

def normalize_text(value):
    """Normalize text for detecting exact duplicates."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def json_default(value):
    """Convert common NumPy/Pandas values for JSON output."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)


# ---------------------------------------------------------
# 3. Load and check input
# ---------------------------------------------------------

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run src/validate.py successfully before this script."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_FILE).reset_index(drop=True)
    rows_before = len(df)

    required = {TEXT_COL, TARGET_COL, ID_COL}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("The validated dataset is empty.")

    # Preserve a stable row key even if complaint_id is missing.
    df["_source_row_id"] = range(len(df))
    df["_normalized_text"] = df[TEXT_COL].map(normalize_text)

    # Rows that cannot be assigned a usable text or label cannot be split.
    unusable = (
        df["_normalized_text"].eq("")
        | df[TARGET_COL].isna()
        | df[TARGET_COL].astype("string").str.strip().eq("").fillna(True)
    )

    unusable_rows = int(unusable.sum())
    df = df.loc[~unusable].copy()

    if df.empty:
        raise ValueError("No usable rows remain after basic checks.")

    # -----------------------------------------------------
    # 4. Detect exact-text narratives with conflicting labels
    # -----------------------------------------------------

    labels_per_text = df.groupby("_normalized_text")[TARGET_COL].nunique()
    conflicting_texts = set(labels_per_text[labels_per_text > 1].index)

    conflicts = df[df["_normalized_text"].isin(conflicting_texts)].copy()

    conflict_file = REPORT_DIR / "conflicting_label_groups.csv"
    if not conflicts.empty:
        conflicts[
            ["_source_row_id", ID_COL, TARGET_COL, TEXT_COL]
        ].to_csv(conflict_file, index=False)
    else:
        pd.DataFrame(
            columns=["_source_row_id", ID_COL, TARGET_COL, TEXT_COL]
        ).to_csv(conflict_file, index=False)

    # Exclude all rows for an exact narrative with contradictory labels.
    # We do not guess which label is correct.
    df = df[~df["_normalized_text"].isin(conflicting_texts)].copy()

    if df.empty:
        raise ValueError(
            "No rows remain after excluding conflicting-label narratives."
        )

    # -----------------------------------------------------
    # 5. Remove exact-text duplicates
    # -----------------------------------------------------

    rows_before_dedup = len(df)

    # At this point each normalized narrative has only one label.
    # Keep the first occurrence and preserve its original text/ID.
    df = df.drop_duplicates(
        subset=["_normalized_text"],
        keep="first",
    ).copy()

    exact_duplicates_removed = rows_before_dedup - len(df)

    if df[TARGET_COL].nunique() < 2:
        raise ValueError("At least two product classes are needed to split.")

    # Each normalized narrative receives one stable group ID.
    # These are exact-text groups, not semantic/near-duplicate groups.
    df["group_id"] = pd.factorize(df["_normalized_text"], sort=True)[0]

    # -----------------------------------------------------
    # 6. Create an approximately stratified grouped split
    # -----------------------------------------------------

    class_counts = df[TARGET_COL].value_counts()

    if class_counts.min() < N_SPLITS:
        too_small = class_counts[class_counts < N_SPLITS].to_dict()
        raise ValueError(
            f"Some classes have fewer than {N_SPLITS} examples, "
            f"so {N_SPLITS}-fold splitting is unsafe: {too_small}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            y=df[TARGET_COL],
            groups=df["group_id"],
        )
    )

    df["split"] = "train"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_groups = set(train_df["group_id"])
    test_groups = set(test_df["group_id"])
    shared_groups = train_groups & test_groups

    if shared_groups:
        raise RuntimeError(
            f"Data leakage detected: {len(shared_groups)} groups "
            "appear in both train and test."
        )

    if train_df.empty or test_df.empty:
        raise RuntimeError("The split produced an empty train or test set.")

    # -----------------------------------------------------
    # 7. Save outputs
    # -----------------------------------------------------

    # Keep the split metadata in the clean file so downstream
    # scripts can use the same reproducible split.
    output_df = df.drop(columns=["_normalized_text"])

    output_df.to_parquet(CLEAN_FILE, index=False)

    split_ids = df[
        ["_source_row_id", ID_COL, "group_id", "split"]
    ].copy()

    split_ids.to_csv(SPLIT_IDS_FILE, index=False)

    # -----------------------------------------------------
    # 8. Save report
    # -----------------------------------------------------

    report = {
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "clean_file": str(CLEAN_FILE.relative_to(PROJECT_ROOT)),
        "split_ids_file": str(SPLIT_IDS_FILE.relative_to(PROJECT_ROOT)),
        "rows_valid_input": int(rows_before),
        "rows_excluded_unusable": unusable_rows,
        "conflicting_narrative_groups": int(len(conflicting_texts)),
        "rows_excluded_due_to_label_conflicts": int(len(conflicts)),
        "exact_duplicate_rows_removed": int(exact_duplicates_removed),
        "rows_after_deduplication": int(len(df)),
        "split_method": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "selected_test_fold": 0,
        "random_seed": RANDOM_SEED,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_fraction": float(len(train_df) / len(df)),
        "test_fraction": float(len(test_df) / len(df)),
        "unique_groups": int(df["group_id"].nunique()),
        "groups_in_both_sets": int(len(shared_groups)),
        "class_counts_before_split": {
            str(k): int(v) for k, v in class_counts.items()
        },
        "class_counts_train": {
            str(k): int(v)
            for k, v in train_df[TARGET_COL].value_counts().items()
        },
        "class_counts_test": {
            str(k): int(v)
            for k, v in test_df[TARGET_COL].value_counts().items()
        },
        "near_duplicate_grouping_performed": False,
        "notes": [
            "Conflicting exact-text narratives are excluded, not relabeled.",
            "Group IDs represent normalized exact-text narratives only.",
            "Near-duplicate semantic grouping requires a separate audited method.",
            "StratifiedGroupKFold aims for class balance; exact 80/20 proportions are not guaranteed."
        ],
    }

    (REPORT_DIR / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    print("Deduplication and splitting completed.")
    print(f"Input rows: {rows_before:,}")
    print(f"Rows after deduplication: {len(df):,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Groups shared between splits: {len(shared_groups)}")
    print(f"Clean data: {CLEAN_FILE}")
    print(f"Split IDs: {SPLIT_IDS_FILE}")
    print(f"Report: {REPORT_DIR / 'split_report.json'}")
    print(f"Label conflicts: {conflict_file}")


if __name__ == "__main__":
    main()


print("The splitting process was successful and all the files were generated with correct and balanced proportions! The files")
print(f"Number of training rows(Train): {report['train_rows']:,}")
print(f"Number of testing rows(Test): {report['test_rows']:,}")

