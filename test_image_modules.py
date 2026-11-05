import json
import time
import traceback
from pathlib import Path

from fedhealth.data_unified import load_medmnist_bundle
from fedhealth.engine_unified import run_federated

datasets = [
    ("Pneumonia", "pneumoniamnist"),
    ("CT Organ", "organamnist"),
    ("Breast Cancer", "breastmnist"),
]

results = []

for display_name, dataset_key in datasets:
    print("\n" + "=" * 65)
    print("TESTING:", display_name)
    print("=" * 65)

    started = time.time()

    try:
        bundle = load_medmnist_bundle(
            dataset_key=dataset_key,
            num_clients=2,
            partition="IID",
            max_train=300,
            max_test=120,
            batch_size=32,
            seed=42
        )

        print("Dataset loaded:", bundle.dataset_name)
        print("Classes:", bundle.class_names)
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
            "module": display_name,
            "status": "PASS",
            "classes": bundle.class_names,
            "hospital_samples": bundle.client_sizes,
            "test_samples": len(bundle.test_loader.dataset),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "sensitivity": metrics["recall_sensitivity"],
            "specificity": metrics["specificity"],
            "roc_auc": metrics["roc_auc"],
            "confusion_matrix": metrics["confusion_matrix"],
            "training_seconds": round(result["elapsed"], 2),
            "total_seconds": round(time.time() - started, 2),
            "device": result["device"]
        }

        print("RESULT:", json.dumps(row, indent=2))
        results.append(row)

    except Exception as e:
        row = {
            "module": display_name,
            "status": "FAIL",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print("FAILED:", e)
        results.append(row)

output = Path("artifacts") / "image_module_test_results.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(results, indent=2), encoding="utf-8")

print("\n" + "=" * 65)
print("BATCH TEST SUMMARY")
for row in results:
    print(row["module"], ":", row["status"])

print("Report saved:", output.resolve())
print("=" * 65)
