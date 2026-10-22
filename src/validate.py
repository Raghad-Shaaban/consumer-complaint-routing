import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------
PROJECT_DIR = Path(os.environ.get("CFPB_PROJECT_DIR", "/content/drive/MyDrive/CFPB_Project"))
IN_FILE = PROJECT_DIR / "data" / "interim" / "complaints_candidate.parquet"
AUDIT_FILE = PROJECT_DIR / "reports" / "source_audit_report.json"
OUT_FILE = PROJECT_DIR / "data" / "interim" / "valid.parquet"
REPORT_FILE = PROJECT_DIR / "reports" / "validation_report.json"
REJECT_FILE = PROJECT_DIR / "reports" / "validation_rejects.csv"

REQUIRED_COLUMNS = ["complaint_id", "date_received", "consumer_complaint_narrative", "product"]
ID, DATE, TEXT, LABEL = REQUIRED_COLUMNS

MIN_WORDS = 3                      # narratives shorter than this are not usable text
MIN_CHARS = 10
MAX_CHARS_SOFT = 20_000            # very long narratives are only flagged
DATE_MIN = pd.Timestamp("2011-01-01", tz="UTC")   # CFPB complaint database starts in 2011
DATE_MAX = pd.Timestamp.now(tz="UTC")
RARE_CLASS_MIN = 50                # classes with fewer sample rows are flagged as rare
MAX_REJECT_RATE = 0.05             # gate: stop the pipeline if more than 5% of rows are rejected

# Product names from the audit that are known to overlap / have been renamed by the CFPB over time.
# (informational only - P3 decides how to treat them)
PRODUCT_FAMILIES = {
    "credit_reporting": ["Credit reporting",
                         "Credit reporting or other personal consumer reports",
                         "Credit reporting, credit repair services, or other personal consumer reports"],
    "credit_card": ["Credit card", "Credit card or prepaid card", "Prepaid card"],
    "bank_account": ["Bank account or service", "Checking or savings account"],
    "payday_loan": ["Payday loan", "Payday loan, title loan, or personal loan",
                    "Payday loan, title loan, personal loan, or advance loan"],
    "money_transfer": ["Money transfers", "Money transfer, virtual currency, or money service",
                       "Virtual currency"],
    "consumer_loan": ["Consumer Loan", "Vehicle loan or lease"],
}


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
def _is_str(x):
    return isinstance(x, str)


def _clean_for_json(o):
    if isinstance(o, dict):
        return {str(k): _clean_for_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_for_json(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


def expected_labels_from_audit():
    """The 20 product names Person 1 found in the full source (usable records)."""
    if AUDIT_FILE.exists():
        rep = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
        return set(rep.get("usable_product_distribution", {}).keys())
    return None


# ----------------------------------------------------------------------------------------
# Core: pure function (no file I/O) so it can be tested on any DataFrame
# ----------------------------------------------------------------------------------------
def validate(df, expected_labels=None):
    report = {"input_rows": int(len(df)), "checks": {}}
    checks = report["checks"]

    # ---------------- 1. SCHEMA (dataset level, hard) ----------------
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    checks["schema"] = {"missing_columns": missing_cols, "extra_columns": extra_cols,
                        "passed": not missing_cols and not extra_cols}
    if missing_cols:
        report["passed"] = False
        return None, pd.DataFrame(columns=[ID, "reason"]), report

    # ---------------- 2. DATA TYPES ----------------
    non_str = {c: int((~df[c].dropna().map(_is_str)).sum()) for c in (ID, TEXT, LABEL)}
    date_is_dt = pd.api.types.is_datetime64_any_dtype(df[DATE])
    checks["dtypes"] = {
        "declared": {c: str(t) for c, t in df.dtypes.items()},
        "non_string_values": non_str,
        "date_is_datetime": bool(date_is_dt),
        "date_is_timezone_aware": bool(date_is_dt and getattr(df[DATE].dt, "tz", None) is not None),
        "complaint_id_not_all_digits": int((~df[ID].astype(str).str.fullmatch(r"\d+")).sum()),
        "passed": bool(date_is_dt and sum(non_str.values()) == 0),
    }

    # ---------------- 3. MISSING VALUES ----------------
    miss = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS}
    blank = {c: int(df[c].dropna().map(lambda s: _is_str(s) and not s.strip()).sum())
             for c in (ID, TEXT, LABEL)}
    checks["missing_values"] = {"null_counts": miss,
                                "null_rate": {c: round(v / max(len(df), 1), 5) for c, v in miss.items()},
                                "blank_string_counts": blank}

    # ---------------- 4. ROW-LEVEL HARD RULES ----------------
    reasons = pd.Series("", index=df.index, dtype=object)

    def flag(mask, reason):
        mask = mask.fillna(False)
        reasons.loc[mask] = reasons.loc[mask].map(lambda r: (r + ";" if r else "") + reason)
        return int(mask.sum())

    id_s = df[ID].astype("string").str.strip()
    txt = df[TEXT].astype("string")
    txt_s = txt.str.strip()
    lab_s = df[LABEL].astype("string").str.strip()

    hard = {}
    hard["id_missing_or_blank"] = flag(id_s.isna() | (id_s == ""), "id_missing_or_blank")
    hard["id_duplicate"] = flag(id_s.duplicated(keep="first") & id_s.notna(), "id_duplicate")
    hard["narrative_missing_or_blank"] = flag(txt_s.isna() | (txt_s == ""), "narrative_missing_or_blank")
    words = txt_s.str.split().str.len()
    hard["narrative_too_short"] = flag(
        txt_s.notna() & (txt_s != "") & ((words < MIN_WORDS) | (txt_s.str.len() < MIN_CHARS)),
        "narrative_too_short")
    hard["narrative_no_letters"] = flag(
        txt_s.notna() & (txt_s != "") & ~txt_s.str.contains(r"[A-Za-z]", regex=True), "narrative_no_letters")
    hard["label_missing_or_blank"] = flag(lab_s.isna() | (lab_s == ""), "label_missing_or_blank")
    if expected_labels:
        hard["label_not_in_known_set"] = flag(
            lab_s.notna() & (lab_s != "") & ~lab_s.isin(expected_labels), "label_not_in_known_set")
    if date_is_dt:
        d = df[DATE]
        if getattr(d.dt, "tz", None) is None:
            d = d.dt.tz_localize("UTC")
        hard["date_out_of_range"] = flag(d.notna() & ((d < DATE_MIN) | (d > DATE_MAX)), "date_out_of_range")
    checks["hard_rule_violations"] = hard

    rejected_mask = reasons != ""
    rejects = pd.DataFrame({ID: df.loc[rejected_mask, ID].values, "reason": reasons[rejected_mask].values})
    valid = df.loc[~rejected_mask].copy()
    # tidy: strip whitespace on the kept rows (reported, not silent)
    ws_changes = int((valid[TEXT] != valid[TEXT].str.strip()).sum()
                     + (valid[LABEL] != valid[LABEL].str.strip()).sum())
    valid[TEXT] = valid[TEXT].str.strip()
    valid[LABEL] = valid[LABEL].str.strip()
    checks["whitespace_stripped_values"] = ws_changes

    # ---------------- 5. DATES (soft) ----------------
    if date_is_dt:
        d = df[DATE]
        checks["dates"] = {
            "missing": int(d.isna().sum()), "missing_rate": round(float(d.isna().mean()), 5),
            "min": d.min(), "max": d.max(),
            "by_year": {str(k): int(v) for k, v in d.dt.year.value_counts().sort_index().items()},
            "note": "Missing dates are kept (date is metadata, not a model feature) but they cannot be "
                    "used for a chronological split.",
        }

    # ---------------- 6. NARRATIVE QUALITY (soft) ----------------
    t = valid[TEXT]
    nchar, nword = t.str.len(), t.str.split().str.len()
    checks["narrative_quality"] = {
        "chars": {"min": int(nchar.min()), "p5": float(nchar.quantile(.05)), "median": float(nchar.median()),
                  "p95": float(nchar.quantile(.95)), "max": int(nchar.max())},
        "words": {"min": int(nword.min()), "p5": float(nword.quantile(.05)), "median": float(nword.median()),
                  "p95": float(nword.quantile(.95)), "max": int(nword.max())},
        "very_short_lt_20_words": int((nword < 20).sum()),
        "very_long_gt_%d_chars" % MAX_CHARS_SOFT: int((nchar > MAX_CHARS_SOFT).sum()),
        "contains_xxxx_redaction": int(t.str.contains(r"X{2,}", regex=True).sum()),
        "contains_xxxx_redaction_rate": round(float(t.str.contains(r"X{2,}", regex=True).mean()), 4),
        "contains_non_ascii": int(t.str.contains(r"[^\x00-\x7f]", regex=True).sum()),
        "contains_url_or_email": int(t.str.contains(r"https?://|www\.|\S+@\S+\.\S+", regex=True).sum()),
        "mostly_uppercase_gt_80pct_letters": int(
            t.map(lambda s: (lambda L: len(L) > 0 and sum(ch.isupper() for ch in L) / len(L) > .8)(
                [ch for ch in s if ch.isalpha()])).sum()),
    }

    # ---------------- 7. DUPLICATES (soft: report only, P3 resolves) ----------------
    norm = valid[TEXT].str.lower().str.replace(r"\s+", " ", regex=True)
    dup_extra = int(len(valid) - norm.nunique())
    grp = valid.assign(_n=norm).groupby("_n")[LABEL].agg(["size", "nunique"])
    rep_groups = grp[grp["size"] > 1]
    checks["duplicates"] = {
        "duplicate_complaint_ids": int(valid[ID].duplicated().sum()),
        "repeated_narrative_extra_rows": dup_extra,
        "repeated_narrative_groups": int(len(rep_groups)),
        "repeated_groups_with_conflicting_labels": int((rep_groups["nunique"] > 1).sum()),
        "rows_in_conflicting_groups": int(rep_groups.loc[rep_groups["nunique"] > 1, "size"].sum()),
        "largest_repeated_group_size": int(rep_groups["size"].max()) if len(rep_groups) else 0,
        "note": "NOT removed here. Person 3 handles deduplication, conflict resolution and the group split.",
    }

    # ---------------- 8. LABEL CONSISTENCY ----------------
    labs = valid[LABEL]
    case_variants = labs.groupby(labs.str.casefold().str.replace(r"\s+", " ", regex=True)).nunique()
    first_last = valid.assign(_y=valid[DATE].dt.year if date_is_dt else np.nan).groupby(LABEL)["_y"].agg(["min", "max"])
    present = set(labs.unique())
    checks["label_consistency"] = {
        "n_classes": int(labs.nunique()),
        "classes_with_spelling_or_case_variants": [k for k, v in case_variants.items() if v > 1],
        "unexpected_labels": sorted(present - expected_labels) if expected_labels else "audit file not found",
        "expected_labels_absent_from_sample": sorted(expected_labels - present) if expected_labels else None,
        "possible_overlapping_name_families": {
            fam: [p for p in names if p in present] for fam, names in PRODUCT_FAMILIES.items()
            if sum(p in present for p in names) > 1},
        "first_last_year_seen_per_class": {k: {"first": None if pd.isna(r["min"]) else int(r["min"]),
                                               "last": None if pd.isna(r["max"]) else int(r["max"])}
                                           for k, r in first_last.iterrows()},
        "note": "Classes that stop/start in different years are renamed versions of the same product. "
                "Merging them is a modelling decision for later stages; it is NOT done here.",
    }

    # ---------------- 9. CLASS DISTRIBUTION ----------------
    vc = labs.value_counts()
    checks["class_distribution"] = {
        "counts": vc.to_dict(),
        "share": (vc / vc.sum()).round(4).to_dict(),
        "majority_class": vc.index[0], "minority_class": vc.index[-1],
        "imbalance_ratio_max_over_min": round(float(vc.iloc[0] / vc.iloc[-1]), 1),
        "rare_classes_lt_%d_rows" % RARE_CLASS_MIN: vc[vc < RARE_CLASS_MIN].to_dict(),
        "classes_with_lt_10_rows": vc[vc < 10].to_dict(),
        "note": "Sample is random, not balanced. Rare classes may be missing from test or val after splitting; "
                "use macro-F1 and stratify/group-stratify.",
    }

    # ---------------- Summary ----------------
    reject_rate = len(rejects) / max(len(df), 1)
    report.update({
        "valid_rows": int(len(valid)), "rejected_rows": int(len(rejects)),
        "reject_rate": round(reject_rate, 5),
        "rejects_by_reason": {k: v for k, v in hard.items() if v},
        "passed": bool(checks["schema"]["passed"] and checks["dtypes"]["passed"]
                       and reject_rate <= MAX_REJECT_RATE),
        "thresholds": {"MIN_WORDS": MIN_WORDS, "MIN_CHARS": MIN_CHARS, "MAX_REJECT_RATE": MAX_REJECT_RATE,
                       "RARE_CLASS_MIN": RARE_CLASS_MIN},
    })
    return valid, rejects, report


# ----------------------------------------------------------------------------------------
# Script entry point
# ----------------------------------------------------------------------------------------
def main():
    if not IN_FILE.exists():
        sys.exit(f"Input not found: {IN_FILE}\nSet CFPB_PROJECT_DIR or mount Google Drive.")
    df = pd.read_parquet(IN_FILE)
    valid, rejects, report = validate(df, expected_labels_from_audit())

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(_clean_for_json(report), indent=2, ensure_ascii=False), encoding="utf-8")
    rejects.to_csv(REJECT_FILE, index=False)

    print(f"input rows : {report['input_rows']:,}")
    print(f"valid rows : {report.get('valid_rows', 0):,}")
    print(f"rejected   : {report.get('rejected_rows', 0):,}  {report.get('rejects_by_reason', {})}")
    print(f"passed gate: {report['passed']}")
    if not report["passed"]:
        sys.exit("FAIL: validation gate did not pass - see reports/validation_report.json")
    valid.to_parquet(OUT_FILE, index=False)
    print("saved:", OUT_FILE)


if __name__ == "__main__":
    main()
#data validation script
