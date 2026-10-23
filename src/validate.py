
"""Validate the CFPB candidate dataset and generate quality reports."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

IN_FILE = PROJECT_DIR / "data" / "interim" / "complaints_candidate.parquet"
AUDIT_FILE = PROJECT_DIR / "reports" / "source_audit_report.json"
OUT_FILE = PROJECT_DIR / "data" / "interim" / "valid.parquet"
REPORT_FILE = PROJECT_DIR / "reports" / "validation_report.json"
REJECT_FILE = PROJECT_DIR / "reports" / "validation_rejects.csv"

REQUIRED_COLUMNS = [
    "complaint_id",
    "date_received",
    "consumer_complaint_narrative",
    "product",
]
ID, DATE, TEXT, LABEL = REQUIRED_COLUMNS

MIN_WORDS = 3
MIN_CHARS = 10
MAX_CHARS_SOFT = 20_000
DATE_MIN = pd.Timestamp("2011-01-01", tz="UTC")
RARE_CLASS_MIN = 50
MAX_REJECT_RATE = 0.05

# Informational only. These are NOT applied as label mappings.
PRODUCT_FAMILIES = {
    "credit_reporting": [
        "Credit reporting",
        "Credit reporting or other personal consumer reports",
        "Credit reporting, credit repair services, or other personal consumer reports",
    ],
    "credit_card": [
        "Credit card",
        "Credit card or prepaid card",
        "Prepaid card",
    ],
    "bank_account": [
        "Bank account or service",
        "Checking or savings account",
    ],
    "payday_loan": [
        "Payday loan",
        "Payday loan, title loan, or personal loan",
        "Payday loan, title loan, personal loan, or advance loan",
    ],
    "money_transfer": [
        "Money transfers",
        "Money transfer, virtual currency, or money service",
        "Virtual currency",
    ],
    "consumer_loan": [
        "Consumer Loan",
        "Vehicle loan or lease",
    ],
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _is_str(value):
    return isinstance(value, str)


def _clean_for_json(value):
    """Convert common NumPy and pandas values to JSON-safe values."""
    if isinstance(value, dict):
        return {
            str(key): _clean_for_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_clean_for_json(item) for item in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if value is pd.NaT:
        return None

    if isinstance(value, np.ndarray):
        return [_clean_for_json(item) for item in value.tolist()]

    return value


def expected_labels_from_audit():
    """Read known product labels from the ingestion audit, if available."""
    if not AUDIT_FILE.is_file():
        return None

    report = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    distribution = report.get("usable_product_distribution")

    if not isinstance(distribution, dict):
        return None

    return set(distribution.keys())


def prepare_dates(df):
    """Convert dates to UTC; unparseable values become NaT."""
    result = df.copy()
    result[DATE] = pd.to_datetime(
        result[DATE],
        errors="coerce",
        utc=True,
    )
    return result


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate(df, expected_labels=None):
    """Validate a DataFrame and return valid rows, rejects, and a report."""
    report = {
        "input_rows": int(len(df)),
        "checks": {},
    }
    checks = report["checks"]

    # 1. Schema
    missing_cols = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]
    extra_cols = [
        column for column in df.columns
        if column not in REQUIRED_COLUMNS
    ]

    checks["schema"] = {
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "passed": not missing_cols and not extra_cols,
    }

    if missing_cols:
        report.update({
            "valid_rows": 0,
            "rejected_rows": int(len(df)),
            "reject_rate": 1.0 if len(df) else 0.0,
            "passed": False,
            "error": "Required columns are missing.",
        })
        rejects = pd.DataFrame({
            ID: df[ID] if ID in df.columns else [None] * len(df),
            "reason": ["schema_missing_columns"] * len(df),
        })
        return None, rejects, report

    # Parse dates consistently.
    df = prepare_dates(df)

    # 2. Data types
    non_str = {
        column: int(
            (~df[column].dropna().map(_is_str)).sum()
        )
        for column in (ID, TEXT, LABEL)
    }

    date_is_dt = pd.api.types.is_datetime64_any_dtype(df[DATE])
    date_tz = getattr(df[DATE].dt, "tz", None) if date_is_dt else None

    checks["dtypes"] = {
        "declared": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "non_string_values": non_str,
        "date_is_datetime": bool(date_is_dt),
        "date_is_timezone_aware": bool(date_tz is not None),
        "complaint_id_not_all_digits": int(
            (~df[ID].astype("string").str.strip().str.fullmatch(r"\d+"))
            .sum()
        ),
        "passed": bool(date_is_dt and sum(non_str.values()) == 0),
    }

    # 3. Missing values
    missing_counts = {
        column: int(df[column].isna().sum())
        for column in REQUIRED_COLUMNS
    }

    blank_counts = {
        column: int(
            df[column].dropna().map(
                lambda value: _is_str(value) and not value.strip()
            ).sum()
        )
        for column in (ID, TEXT, LABEL)
    }

    checks["missing_values"] = {
        "null_counts": missing_counts,
        "null_rate": {
            column: round(count / max(len(df), 1), 5)
            for column, count in missing_counts.items()
        },
        "blank_string_counts": blank_counts,
    }

    # 4. Row-level hard rules
    reasons = pd.Series("", index=df.index, dtype=object)

    def flag(mask, reason):
        mask = mask.fillna(False).astype(bool)
        reasons.loc[mask] = reasons.loc[mask].map(
            lambda current: (
                f"{current};{reason}" if current else reason
            )
        )
        return int(mask.sum())

    id_s = df[ID].astype("string").str.strip()
    txt_s = df[TEXT].astype("string").str.strip()
    lab_s = df[LABEL].astype("string").str.strip()

    hard = {}

    hard["id_missing_or_blank"] = flag(
        id_s.isna() | (id_s == ""),
        "id_missing_or_blank",
    )

    hard["id_duplicate"] = flag(
        id_s.duplicated(keep="first") & id_s.notna(),
        "id_duplicate",
    )

    hard["narrative_missing_or_blank"] = flag(
        txt_s.isna() | (txt_s == ""),
        "narrative_missing_or_blank",
    )

    words = txt_s.str.split().str.len()

    hard["narrative_too_short"] = flag(
        txt_s.notna()
        & (txt_s != "")
        & ((words < MIN_WORDS) | (txt_s.str.len() < MIN_CHARS)),
        "narrative_too_short",
    )

    hard["narrative_no_letters"] = flag(
        txt_s.notna()
        & (txt_s != "")
        & ~txt_s.str.contains(r"[A-Za-z]", regex=True, na=False),
        "narrative_no_letters",
    )

    hard["label_missing_or_blank"] = flag(
        lab_s.isna() | (lab_s == ""),
        "label_missing_or_blank",
    )

    if expected_labels is not None:
        hard["label_not_in_known_set"] = flag(
            lab_s.notna()
            & (lab_s != "")
            & ~lab_s.isin(expected_labels),
            "label_not_in_known_set",
        )

    # Missing dates are allowed; invalid nonmissing dates are NaT after parsing.
    # Preserve the original missing-date distinction for reporting.
    original_dates = pd.Series(df[DATE], index=df.index)
    invalid_date_mask = original_dates.isna()

    # Since prepare_dates has already run, NaT includes both missing and
    # unparseable values. Report this separately; do not reject automatically.
    checks["hard_rule_violations"] = hard

    rejected_mask = reasons != ""
    rejects = pd.DataFrame({
        ID: df.loc[rejected_mask, ID].values,
        "reason": reasons.loc[rejected_mask].values,
    })

    valid = df.loc[~rejected_mask].copy()

    if valid.empty:
        reject_rate = len(rejects) / max(len(df), 1)
        report.update({
            "valid_rows": 0,
            "rejected_rows": int(len(rejects)),
            "reject_rate": round(reject_rate, 5),
            "rejects_by_reason": {
                key: value for key, value in hard.items() if value
            },
            "passed": False,
            "error": "No valid rows remain after validation.",
            "thresholds": {
                "MIN_WORDS": MIN_WORDS,
                "MIN_CHARS": MIN_CHARS,
                "MAX_REJECT_RATE": MAX_REJECT_RATE,
                "RARE_CLASS_MIN": RARE_CLASS_MIN,
            },
        })
        return valid, rejects, report

    # Strip whitespace from accepted narratives and labels.
    ws_changes = int(
        (valid[TEXT] != valid[TEXT].str.strip()).sum()
        + (valid[LABEL] != valid[LABEL].str.strip()).sum()
    )

    valid[TEXT] = valid[TEXT].str.strip()
    valid[LABEL] = valid[LABEL].str.strip()
    checks["whitespace_stripped_values"] = ws_changes

    # 5. Dates (soft checks)
    d = valid[DATE]

    checks["dates"] = {
        "missing_or_unparseable": int(d.isna().sum()),
        "missing_or_unparseable_rate": round(float(d.isna().mean()), 5),
        "min": d.min(),
        "max": d.max(),
        "by_year": {
            str(year): int(count)
            for year, count in d.dt.year.value_counts().sort_index().items()
        },
        "out_of_range_before_2011": int(
            (d.notna() & (d < DATE_MIN)).sum()
        ),
        "note": (
            "Missing or unparseable dates are retained as metadata, "
            "but cannot be used for chronological splitting."
        ),
    }

    # 6. Narrative quality (soft checks)
    t = valid[TEXT]
    nchar = t.str.len()
    nword = t.str.split().str.len()

    redaction = t.str.contains(r"X{2,}", regex=True, na=False)
    non_ascii = t.str.contains(r"[^\x00-\x7f]", regex=True, na=False)
    url_email = t.str.contains(
        r"https?://|www\.|\S+@\S+\.\S+",
        regex=True,
        na=False,
    )

    def mostly_uppercase(text):
        letters = [char for char in text if char.isalpha()]
        return bool(
            letters
            and sum(char.isupper() for char in letters) / len(letters) > 0.8
        )

    checks["narrative_quality"] = {
        "chars": {
            "min": int(nchar.min()),
            "p5": float(nchar.quantile(0.05)),
            "median": float(nchar.median()),
            "p95": float(nchar.quantile(0.95)),
            "max": int(nchar.max()),
        },
        "words": {
            "min": int(nword.min()),
            "p5": float(nword.quantile(0.05)),
            "median": float(nword.median()),
            "p95": float(nword.quantile(0.95)),
            "max": int(nword.max()),
        },
        "very_short_lt_20_words": int((nword < 20).sum()),
        f"very_long_gt_{MAX_CHARS_SOFT}_chars": int(
            (nchar > MAX_CHARS_SOFT).sum()
        ),
        "contains_xxxx_redaction": int(redaction.sum()),
        "contains_xxxx_redaction_rate": round(float(redaction.mean()), 4),
        "contains_non_ascii": int(non_ascii.sum()),
        "contains_url_or_email": int(url_email.sum()),
        "mostly_uppercase_gt_80pct_letters": int(
            t.map(mostly_uppercase).sum()
        ),
    }

    # 7. Duplicate checks (report only; Person 3 handles deduplication)
    norm = (
        valid[TEXT]
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    dup_extra = int(len(valid) - norm.nunique())
    grouped = valid.assign(_normalized_text=norm).groupby(
        "_normalized_text"
    )[LABEL].agg(["size", "nunique"])

    repeated_groups = grouped[grouped["size"] > 1]
    conflicting_groups = repeated_groups[
        repeated_groups["nunique"] > 1
    ]

    checks["duplicates"] = {
        "duplicate_complaint_ids": int(valid[ID].duplicated().sum()),
        "repeated_narrative_extra_rows": dup_extra,
        "repeated_narrative_groups": int(len(repeated_groups)),
        "repeated_groups_with_conflicting_labels": int(
            len(conflicting_groups)
        ),
        "rows_in_conflicting_groups": int(
            conflicting_groups["size"].sum()
        ),
        "largest_repeated_group_size": (
            int(repeated_groups["size"].max())
            if not repeated_groups.empty else 0
        ),
        "note": (
            "Duplicates are reported, not removed. The deduplication "
            "stage handles duplicate resolution and group splitting."
        ),
    }

    # 8. Label consistency
    labs = valid[LABEL]
    canonical_names = labs.str.casefold().str.replace(
        r"\s+", " ", regex=True
    )
    case_variants = (
        valid.assign(_canonical_name=canonical_names)
        .groupby("_canonical_name")[LABEL]
        .nunique()
    )

    first_last = valid.assign(
        _year=valid[DATE].dt.year
    ).groupby(LABEL)["_year"].agg(["min", "max"])

    present = set(labs.unique())

    checks["label_consistency"] = {
        "n_classes": int(labs.nunique()),
        "classes_with_spelling_or_case_variants": [
            label for label, count in case_variants.items()
            if count > 1
        ],
        "unexpected_labels": (
            sorted(present - expected_labels)
            if expected_labels is not None
            else "audit file not found"
        ),
        "expected_labels_absent_from_sample": (
            sorted(expected_labels - present)
            if expected_labels is not None
            else None
        ),
        "possible_overlapping_name_families": {
            family: [name for name in names if name in present]
            for family, names in PRODUCT_FAMILIES.items()
            if sum(name in present for name in names) > 1
        },
        "first_last_year_seen_per_class": {
            label: {
                "first": (
                    None if pd.isna(row["min"]) else int(row["min"])
                ),
                "last": (
                    None if pd.isna(row["max"]) else int(row["max"])
                ),
            }
            for label, row in first_last.iterrows()
        },
        "note": (
            "Potential historical renames are reported, not merged. "
            "Verify CFPB mappings before harmonizing labels."
        ),
    }

    # 9. Class distribution
    value_counts = labs.value_counts()

    checks["class_distribution"] = {
        "counts": value_counts.to_dict(),
        "share": (value_counts / value_counts.sum()).round(4).to_dict(),
        "majority_class": value_counts.index[0],
        "minority_class": value_counts.index[-1],
        "imbalance_ratio_max_over_min": round(
            float(value_counts.iloc[0] / value_counts.iloc[-1]), 1
        ),
        f"rare_classes_lt_{RARE_CLASS_MIN}_rows": (
            value_counts[value_counts < RARE_CLASS_MIN].to_dict()
        ),
        "classes_with_lt_10_rows": (
            value_counts[value_counts < 10].to_dict()
        ),
        "note": (
            "Use suitable stratified or group-aware evaluation and "
            "report macro-F1 for imbalanced classes."
        ),
    }

    # Summary and validation gate
    reject_rate = len(rejects) / max(len(df), 1)

    report.update({
        "valid_rows": int(len(valid)),
        "rejected_rows": int(len(rejects)),
        "reject_rate": round(reject_rate, 5),
        "rejects_by_reason": {
            key: value for key, value in hard.items() if value
        },
        "passed": bool(
            checks["schema"]["passed"]
            and checks["dtypes"]["passed"]
            and reject_rate <= MAX_REJECT_RATE
        ),
        "thresholds": {
            "MIN_WORDS": MIN_WORDS,
            "MIN_CHARS": MIN_CHARS,
            "MAX_CHARS_SOFT": MAX_CHARS_SOFT,
            "MAX_REJECT_RATE": MAX_REJECT_RATE,
            "RARE_CLASS_MIN": RARE_CLASS_MIN,
        },
    })

    return valid, rejects, report


# --------------------------------------------------
# Script entry point
# --------------------------------------------------

def main():
    if not IN_FILE.is_file():
        sys.exit(
            f"Input not found: {IN_FILE}\n"
            "Make the candidate Parquet file available at this "
            "repository-relative path before running validation."
        )

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REJECT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(IN_FILE)

    valid, rejects, report = validate(
        df,
        expected_labels_from_audit(),
    )

    REPORT_FILE.write_text(
        json.dumps(
            _clean_for_json(report),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rejects.to_csv(REJECT_FILE, index=False)

    print(f"Input rows : {report['input_rows']:,}")
    print(f"Valid rows : {report.get('valid_rows', 0):,}")
    print(
        f"Rejected   : {report.get('rejected_rows', 0):,} "
        f"{report.get('rejects_by_reason', {})}"
    )
    print(f"Passed gate: {report['passed']}")

    if not report["passed"]:
        sys.exit(
            "FAIL: validation gate did not pass. "
            f"See {REPORT_FILE}"
        )

    if valid is None:
        sys.exit("FAIL: no valid dataset was returned.")

    valid.to_parquet(OUT_FILE, index=False)
    print(f"Saved validated data: {OUT_FILE}")
    print(f"Saved validation report: {REPORT_FILE}")
    print(f"Saved rejects: {REJECT_FILE}")


if __name__ == "__main__":
    main()
