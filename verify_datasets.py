from pathlib import Path
import medmnist

root = Path.home() / ".medmnist"
root.mkdir(exist_ok=True)

datasets = {
    "PneumoniaMNIST": medmnist.PneumoniaMNIST,
    "OrganAMNIST": medmnist.OrganAMNIST,
    "BreastMNIST": medmnist.BreastMNIST,
}

for name, cls in datasets.items():
    print(f"\n===== {name} =====")
    for split in ["train", "val", "test"]:
        try:
            ds = cls(split=split, root=str(root), download=True)
            print(f"{split}: {len(ds)} samples | image shape: {ds.imgs.shape} | labels: {ds.labels.shape}")
        except Exception as e:
            print(f"{split}: FAILED — {type(e).__name__}: {e}")

print("\nDataset verification finished.")
