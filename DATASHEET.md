# Datasheet — CFPB Consumer Complaint Routing

## 1. Dataset source
- Source: `Mouwiya/cfpb-consumer-complaints` on Hugging Face.
- Subject: Consumer complaints concerning financial products and services.
- Source rows scanned: 16,558,052.
- Eligible records found: 3,815,251.
- Sampling method: Reservoir sampling over the full source stream.
- Random seed: 42.
- Requested and saved sample size: 20,000 rows.

## 2. Prediction task
The goal is to predict the financial product category associated with
a complaint, using the consumer's written narrative.

- Input: `consumer_complaint_narrative`
- Target: `product`
- Identifier: `complaint_id`
- Date metadata: `date_received`

The complaint ID and receipt date are not model input features in the
initial text-classification baseline.

## 3. Eligibility criteria
Records were eligible for sampling when both the complaint narrative
and product label were nonempty.

The source audit found:
- 12,742,801 rows without a usable narrative.
- 0 records missing a product label among records with usable narratives.

## 4. Sampling and reproducibility
The sample was selected using reservoir sampling over the complete
stream, with random seed 42. This avoids restricting selection to the
first portion of the source dataset.

Reproducing the exact sample also depends on the source dataset version
and the order in which records are streamed.

## 5. Initial data checks
The saved candidate sample contains 20,000 rows and four columns:
`complaint_id`, `date_received`, `consumer_complaint_narrative`,
and `product`.

Initial checks found:
- No missing complaint IDs.
- No missing or blank narratives.
- No missing product labels.
- 256 missing receipt dates (1.28%).
- 0 duplicate complaint IDs.
- 2,976 additional repeated narrative rows, representing 616 distinct
  repeated narrative groups.
- 68 repeated narrative groups with conflicting product labels.

These duplicate and label-conflict findings are audit results. The
candidate sample has not been silently modified to resolve them.
Further cleaning and splitting decisions belong to subsequent stages.

## 6. Intended use
Educational research on classifying consumer complaint narratives
into financial product categories to support complaint routing.

## 7. Limitations
- The sample includes only records with usable narratives and product
  labels, so it does not represent all source complaints.
- The sample is a random sample of eligible records, not a balanced
  sample of product categories.
- Product categories are imbalanced; some categories may have very few
  examples or be absent from the sample.
- Product-category names may overlap or differ across time.
- Repeated or templated narratives may inflate evaluation scores if
  duplicates cross data partitions.
- Identical narratives with different labels create ambiguity for a
  model that only sees the narrative.
- Missing dates limit the use of affected records in chronological
  evaluation.
- Predictive performance does not guarantee that every prediction is
  correct. Predictions should be reviewed appropriately.

## 8. Privacy and responsible use
Review the source's terms and applicable privacy requirements.
Public availability does not eliminate privacy concerns. Avoid publishing
raw complaint narratives or potentially sensitive information in a
public repository. Treat predictions as decision support rather than
infallible decisions.

## 9. Reproducibility and artifacts
- Ingestion sample: `data/interim/complaints_candidate.parquet`
- Source audit: `reports/source_audit_report.json`
- Column inventory: `data/inventory.csv`
- Dataset documentation: `DATASHEET.md`

The candidate sample is the output of ingestion. Validation, conflict
handling, deduplication, and train/validation/test splitting are
separate project stages.
