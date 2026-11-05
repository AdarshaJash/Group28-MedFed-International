# MedFed – Group 28

**A modular federated-learning research prototype for medical image and tabular classification.**

This project simulates multiple healthcare institutions (clients) that train locally and share model updates for aggregation. It currently supports three downloadable MedMNIST image benchmarks and two CSV-driven tabular task categories.

## Supported modules and exact scope

| Dashboard module | Dataset / input | Task | Important boundary |
|---|---|---|---|
| Pneumonia | PneumoniaMNIST | Normal vs pneumonia chest X-ray benchmark classification | Small 28×28 benchmark; not clinical diagnosis |
| CT scan | OrganAMNIST | Axial abdominal CT organ classification | Organ classification, not cancer/tumour detection |
| Cancer-related imaging | BreastMNIST | Benign vs malignant breast ultrasound benchmark classification | Ultrasound benchmark, not CT or general cancer diagnosis |
| Diabetes | User-supplied labelled CSV | Target column selected by user | Requires documented public/consented dataset and label definition |
| Blood tests | User-supplied labelled CSV | Target column selected by user | Lab values do not independently establish a diagnosis |

The five modules are separate tasks with separate models. They do **not** share one global model. The image datasets are downloaded from MedMNIST when selected. Diabetes and blood-test datasets are not fabricated or bundled; upload a documented CSV with a target column.

## Architecture

- `app.py`: Streamlit research dashboard.
- `fedhealth/data_unified.py`: MedMNIST and CSV ingestion, preprocessing, client partitioning.
- `fedhealth/models.py`: CNN and tabular MLP model factories.
- `fedhealth/engine_unified.py`: reusable in-process FedAvg and FedProx training, evaluation and baselines.
- `fedhealth/flower_adapter.py`: generic Flower NumPyClient and FedAvg strategy adapters for a separately launched Flower runtime.
- `fedhealth/model.py`, `data.py`, `engine.py`: original V2 pneumonia implementation retained for reference/backward compatibility.
- `fedhealth/flower_client.py`, `flower_server.py`: original Flower adapter/strategy examples retained; the current dashboard engine is an in-process simulator, not a Flower distributed runtime.
- `tests/`: offline smoke tests for aggregation, tabular ingestion, model training and evaluation.
- `research/`: experiment protocol and paper planning notes.

## Windows setup (Python 3.11 recommended)

Open PowerShell in the extracted `Group28_MedFed_International` folder (the folder containing `app.py` and `requirements.txt`):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If `.venv` already exists, skip the first command. First use of a MedMNIST image module downloads its dataset and therefore needs internet access.

## Running an experiment

1. Choose one of the five modules in the sidebar.
2. For image modules, set the maximum training/test subset sizes.
3. For Diabetes or Blood tests, upload a CSV and choose its target column. Ensure the target is a meaningful, documented label and remove any leakage-prone columns before upload.
4. Set client count, rounds, local epochs, IID/Non-IID distribution and FedAvg/FedProx.
5. Run the experiment. Compare the federated model with local-only and centralised baselines.
6. Export JSON results, round metrics CSV and model weights.

## Methodology notes

- MedMNIST uses its official train, validation and test splits.
- CSV data is split into train/validation/test subsets with stratification. Imputation, scaling and one-hot encoding are fitted on training data only.
- CSV datasets with repeated records per patient should be split by patient before use. The generic uploader cannot infer patient linkage safely.
- Non-IID partitioning uses a Dirichlet distribution (alpha=0.35). IID partitioning randomly distributes examples.
- FedAvg performs sample-count-weighted parameter averaging. FedProx adds a proximal penalty (mu=0.01) during local optimisation.
- Local-only and centralised baselines use `local_epochs × rounds` epochs; this is an approximate epoch-budget comparison, not a guarantee of equal wall-clock compute.
- Report multiple random seeds, class-wise metrics, confidence intervals where possible, and dataset provenance/licensing in any paper.

## Privacy and clinical limitations

This is a **single-machine simulation**. Raw records stay in separate client dataset partitions during the simulated training loop, but this is not a real hospital deployment. Model updates may leak information. This version does not implement formal differential privacy, secure aggregation, authentication, access control or regulatory compliance. Do not claim that federated learning alone guarantees privacy.

All outputs are research benchmark predictions, not medical diagnoses. No clinical validation or regulatory approval is claimed. Do not upload identifiable patient data.

## Research question

How does federated learning perform across heterogeneous medical data modalities under IID and non-IID client distributions, compared with local-only and centralised training?

Suggested experiments: FedAvg vs FedProx; IID vs non-IID; 2/3/5/10 clients; varying communication rounds/local epochs; repeated seeds; privacy–utility experiments only after a formal DP mechanism is implemented.
