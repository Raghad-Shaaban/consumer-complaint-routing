# Consumer Complaint Routing

## Six-field statement

| Field | Value |
|---|---|
| Input (x) | Consumer complaint narrative |
| Output (y) | Financial product category |
| Prediction time (t0) | When the complaint is received, before response-related information becomes available |
| Decision | Route the complaint to the predicted product team; uncertain cases are escalated to a human |
| Metric | Macro F1-score |
| Constraints | Avoid label leakage by excluding Issue, Sub-issue, Sub-product, and response-related fields; handle taxonomy drift and class imbalance |

## How to reproduce

(to be completed in W04: dvc repro)
