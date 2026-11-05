import io
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from fedhealth.data_unified import load_csv_bundle
from fedhealth.engine_unified import run_federated

rng = np.random.default_rng(42)
results = []

# Synthetic diabetes-style dataset
n = 300
glucose = rng.normal(125, 25, n).clip(60, 250)
bmi = rng.normal(27, 5, n).clip(15, 50)
age = rng.integers(20, 75, n)
family = rng.choice(["yes", "no"], n)
risk = ((glucose > 135) & (bmi > 27)) | ((age > 55) & (family == "yes"))

diabetes = pd.DataFrame({
    "patient_id": range(n),
    "glucose": glucose,
    "bmi": bmi,
    "age": age,
    "family_history": family,
    "diabetes_label": np.where(risk, "positive", "negative")
})

# Synthetic blood-test-style dataset
n = 300
haemoglobin = rng.normal(13, 2, n).clip(5, 19)
wbc = rng.normal(7500, 2000, n).clip(1500, 16000)
platelets = rng.normal(250000, 50000, n).clip(50000, 500000)
age = rng.integers(18, 80, n)

abnormal = (haemoglobin < 10) | (wbc > 10500) | (platelets < 150000)

blood = pd.DataFrame({
    "patient_id": range(n),
    "haemoglobin": haemoglobin,
    "wbc": wbc,
    "platelets": platelets,
    "age": age,
    "blood_result": np.where(abnormal, "abnormal", "normal")
})

workflows = [
    ("Diabetes CSV", diabetes, "diabetes_label"),
    ("Blood Test CSV", blood, "blood_result"),
]

for name, df, target in workflows:
    print("\n" + "=" * 65)
    print("TESTING:", name)
    print("=" * 65)

    started = time.time()

    try:
        # Add a few missing feature values to test imputation
        feature_col = [c for c in df.columns if c not in [target, "patient_id"]][0]
        df.loc[df.index[:5], feature_col] = np.nan

        csv_bytes = df.to_csv(index=False).encode("utf-8")

        bundle = load_csv_bundle(
            file_bytes=csv_bytes,
            target_column=target,
            num_clients=2,
            partition="IID",
            batch_size=32,
            seed=42
        )

        print("Dataset:", bundle.dataset_name)
        print("Classes:", bundle.class_names)
        print("Features:", bundle.feature_names)
        print("Input dimensions:", bundle.input_dim)
        print("Hospital samples:", bundle.client_sizes)
        print("Test samples:", len(bundle.test_loader.dataset))

        result = run_federated(
            bundle,
            rounds=1,
            local_epochs=1,
            seed=42,
            strategy="FedAvg"
        )

        metrics = result["final"]

        row = {
            "module": name,
            "status": "PASS",
            "classes": bundle.class_names,
            "input_dimensions": bundle.input_dim,
            "hospital_samples": bundle.client_sizes,
            "test_samples": len(bundle.test_loader.dataset),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "sensitivity": metrics["recall_sensitivity"],
            "specificity": metrics["specificity"],
            "roc_auc": metrics["roc_auc"],
            "confusion_matrix": metrics["confusion_matrix"],
            "seconds": round(time.time() - started, 2)
        }

        print(json.dumps(row, indent=2))
        results.append(row)

    except Exception as e:
        print("FAILED:", str(e))
        results.append({
            "module": name,
            "status": "FAIL",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

# Test invalid target handling
print("\nTesting invalid CSV target handling...")
try:
    load_csv_bundle(
        diabetes.to_csv(index=False).encode("utf-8"),
        target_column="missing_target"
    )
    print("FAIL: Invalid target was accepted.")
except ValueError:
    print("PASS: Invalid target correctly rejected.")

output = Path("artifacts") / "tabular_module_test_results.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(results, indent=2), encoding="utf-8")

print("\n========== TABULAR TEST SUMMARY ==========")
for row in results:
    print(row["module"], ":", row["status"])
print("Report:", output.resolve())
