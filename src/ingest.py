
"""Stream the CFPB dataset, audit its source, and save a candidate sample."""

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_ID = "Mouwiya/cfpb-consumer-complaints"
DATASET_REVISION = None  # Pin to a verified Hugging Face commit before rerunning.
SAMPLE_SIZE = 20_000
RANDOM_SEED = 42

REQUIRED_COLUMNS = [
    "complaint_id",
    "date_received",
    "consumer_complaint_narrative",
    "product",
]


def find_project_root():
    """Find the repository root without assuming the script's depth."""
    script_path = Path(__file__).resolve()

    for parent in script_path.parents:
        if (parent / "dvc.yaml").is_file():
            return parent

    raise RuntimeError(
        "Could not find the project root: no dvc.yaml found "
        f"in the parent directories of {script_path}"
    )


PROJECT_ROOT = find_project_root()
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
SAMPLE_FILE = OUTPUT_DIR / "complaints_candidate.parquet"
REPORT_FILE = OUTPUT_DIR / "source_audit_report.json"


def get_year(value):
    """Return the year or a marker for an invalid/missing date."""
    if value is None:
        return "missing_or_invalid"

    parsed = pd.to_datetime(value, errors="coerce", utc=True)

    if pd.isna(parsed):
        return "missing_or_invalid"

    return str(parsed.year)


def run_source_audit():
    """Scan the dataset stream and save a reservoir sample."""
    if SAMPLE_SIZE <= 0:
        raise ValueError("SAMPLE_SIZE must be positive.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    load_args = {
        "path": DATASET_ID,
        "split": "train",
        "streaming": True,
    }

    if DATASET_REVISION is not None:
        load_args["revision"] = DATASET_REVISION

    dataset = load_dataset(**load_args)

    missing_columns = set(REQUIRED_COLUMNS) - set(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    rng = random.Random(RANDOM_SEED)
    reservoir = []

    rows_scanned = 0
    missing_narrative = 0
    missing_product = 0
    usable_seen = 0

    all_years = Counter()
    usable_years = Counter()
    usable_products = Counter()

    for row in dataset:
        rows_scanned += 1

        year = get_year(row.get("date_received"))
        all_years[year] += 1

        narrative = row.get("consumer_complaint_narrative")
        product = row.get("product")

        if not isinstance(narrative, str) or not narrative.strip():
            missing_narrative += 1
            continue

        if not isinstance(product, str) or not product.strip():
            missing_product += 1
            continue

        usable_seen += 1
        usable_years[year] += 1
        usable_products[product.strip()] += 1

        record = {
            "complaint_id": row.get("complaint_id"),
            "date_received": row.get("date_received"),
            "consumer_complaint_narrative": narrative.strip(),
            "product": product.strip(),
        }

        if len(reservoir) < SAMPLE_SIZE:
            reservoir.append(record)
        else:
            index = rng.randrange(usable_seen)
            if index < SAMPLE_SIZE:
                reservoir[index] = record

        if rows_scanned % 500_000 == 0:
            print(
                f"Scanned {rows_scanned:,} rows; "
                f"{usable_seen:,} usable records found."
            )

    if not reservoir:
        raise ValueError("No usable records found.")

    df_candidate = pd.DataFrame(reservoir, columns=REQUIRED_COLUMNS)
    df_candidate["date_received"] = pd.to_datetime(
        df_candidate["date_received"],
        errors="coerce",
        utc=True,
    )

    df_candidate.to_parquet(SAMPLE_FILE, index=False)

    report = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "random_seed": RANDOM_SEED,
        "requested_sample_size": SAMPLE_SIZE,
        "sampling_method": "reservoir sampling over the full stream",
        "rows_scanned": rows_scanned,
        "usable_records_seen": usable_seen,
        "sample_rows_saved": len(df_candidate),
        "rows_missing_or_empty_narrative": missing_narrative,
        "rows_missing_product_among_narrative_usable": missing_product,
        "all_source_year_distribution": dict(sorted(all_years.items())),
        "usable_year_distribution": dict(sorted(usable_years.items())),
        "usable_product_distribution": dict(sorted(usable_products.items())),
        "sample_file": str(SAMPLE_FILE.relative_to(PROJECT_ROOT)),
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Source audit complete: {rows_scanned:,} rows scanned.")
    print(f"Usable records found: {usable_seen:,}")
    print(f"Sample rows saved: {len(df_candidate):,}")
    print(f"Sample: {SAMPLE_FILE}")
    print(f"Report: {REPORT_FILE}")

    return df_candidate, report


if __name__ == "__main__":
    run_source_audit()
