# Consumer Complaint Routing

## consumer-complaint-routing

| Field | Definition |
|---|---|
| 1. Input (x) | The consumer complaint narrative and information available at the time the complaint is received. |
| 2. Output (y) | The financial product category associated with the complaint. |
| 3. Prediction Time (t₀) | The prediction is made when the complaint is received, before routing or response-related information becomes available. |
| 4. Decision | Route the complaint to the predicted product team. If the model is uncertain, escalate the complaint to a human for review. |
| 5. Metric | Primary: macro-F1, to account for multi-class classification and class imbalance between product categories. Secondary: the share of complaints escalated to a human, and accuracy on the complaints the model routes automatically. |
| 6. Constraints | • Response time under 1 second per complaint.<br>• No personal data stored.<br>• Cost per 1,000 complaints reported. |
