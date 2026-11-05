# MedFed experiment protocol (draft)

## Primary question
How does federated learning perform across heterogeneous medical data modalities under IID and non-IID client distributions compared with local-only and centralised training?

## Tasks
1. PneumoniaMNIST: binary chest X-ray benchmark classification.
2. OrganAMNIST: axial abdominal CT organ classification.
3. BreastMNIST: benign/malignant breast ultrasound benchmark classification.
4. Diabetes: user-supplied documented labelled tabular dataset.
5. Blood tests: user-supplied documented labelled laboratory dataset.

These are five separate prediction tasks and separate model instances, not one universal diagnostic model.

## Experimental controls
- Record dataset version, source, licence, class mapping, preprocessing, random seed, client count, partition mode, Dirichlet alpha, local epochs, communication rounds, optimiser, learning rate and strategy.
- Keep official benchmark test sets untouched. For CSV data, use patient-level split where patient identifiers permit it; split before preprocessing.
- Fit imputation, encoding and scaling on training data only.
- Run at least five random seeds for reported comparisons when compute permits.
- Compare local-only, centralised and federated approaches under explicitly stated training budgets.
- Compare IID and non-IID (Dirichlet alpha 0.35) distributions, FedAvg and FedProx, client counts and rounds.

## Metrics
Report accuracy, balanced accuracy, macro-F1, per-class precision/recall, confusion matrix and ROC-AUC where mathematically valid. For binary tasks, report sensitivity and specificity. Report training time and communication volume. Include confidence intervals or mean ± standard deviation across seeds.

## Privacy claims
The current prototype does not implement differential privacy or secure aggregation. Do not make formal privacy-guarantee claims. Add a threat model and evaluate a validated DP mechanism before reporting epsilon/delta.

## Reporting discipline
Do not call benchmark scores clinical accuracy. State dataset limitations and intended task exactly. Avoid claims about diagnosis, clinical readiness or real hospital deployment.
