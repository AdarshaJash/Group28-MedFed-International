from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import medmnist
from medmnist import INFO

@dataclass
class DataBundle:
    client_loaders: list
    client_sizes: list
    test_loader: DataLoader
    validation_loader: DataLoader
    train_loader: DataLoader
    class_names: list
    client_label_counts: list
    train_labels: np.ndarray
    class_weights: torch.Tensor

def _stratified_limit(labels, limit, seed):
    """Select up to limit indices while approximately preserving class proportions."""
    labels = np.asarray(labels).reshape(-1)
    limit = min(int(limit), len(labels))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    exact = counts / counts.sum() * limit
    quotas = np.floor(exact).astype(int)
    # Distribute any remaining slots to the classes with the largest fractional remainders.
    remainder = limit - int(quotas.sum())
    if remainder > 0:
        order = np.argsort(-(exact - quotas))
        for i in order[:remainder]:
            quotas[i] += 1
    chosen = []
    for cls, quota in zip(classes, quotas):
        ids = np.where(labels == cls)[0]
        rng.shuffle(ids)
        chosen.extend(ids[:quota].tolist())
    rng.shuffle(chosen)
    return chosen

def _stratified_train_validation_split(selected_ids, all_labels, validation_fraction, seed):
    """Split selected dataset indices class-wise so both sets retain both labels."""
    rng = np.random.default_rng(seed)
    train_ids, val_ids = [], []
    selected_labels = np.asarray(all_labels).reshape(-1)
    for cls in np.unique(selected_labels[np.asarray(selected_ids, dtype=int)]):
        ids = np.asarray([idx for idx in selected_ids if selected_labels[idx] == cls], dtype=int)
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * validation_fraction))) if len(ids) > 1 else 0
        val_ids.extend(ids[:n_val].tolist())
        train_ids.extend(ids[n_val:].tolist())
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids

def load_data(num_clients=3, max_train=2400, max_test=1000, batch_size=64, seed=42, partition="Non-IID"):
    info = INFO["pneumoniamnist"]
    Dataset = getattr(medmnist, info["python_class"])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_full = Dataset(split="train", transform=transform, download=True)
    test_full = Dataset(split="test", transform=transform, download=True)
    train_labels_all = np.asarray(train_full.labels).reshape(-1).astype(int)
    test_labels_all = np.asarray(test_full.labels).reshape(-1).astype(int)

    selected_train_ids = _stratified_limit(train_labels_all, min(max_train, len(train_full)), seed)
    train_ids, validation_ids = _stratified_train_validation_split(
        selected_train_ids, train_labels_all, validation_fraction=0.15, seed=seed + 11
    )
    test_ids = _stratified_limit(test_labels_all, min(max_test, len(test_full)), seed + 1)
    y = train_labels_all[np.asarray(train_ids, dtype=int)]

    rng = np.random.default_rng(seed)
    if partition.upper() == "IID":
        chunks = [x.tolist() for x in np.array_split(rng.permutation(len(train_ids)), num_clients)]
    else:
        buckets = [[] for _ in range(num_clients)]
        for cls in np.unique(y):
            ids = rng.permutation(np.where(y == cls)[0])
            proportions = rng.dirichlet(np.repeat(0.35, num_clients))
            cuts = (np.cumsum(proportions) * len(ids)).astype(int)[:-1]
            for j, part in enumerate(np.split(ids, cuts)):
                buckets[j].extend(part.tolist())
        chunks = [rng.permutation(x).tolist() for x in buckets]

    train_set = Subset(train_full, train_ids)
    validation_set = Subset(train_full, validation_ids)
    test_set = Subset(test_full, test_ids)
    client_loaders, counts = [], []
    for p in chunks:
        client_loaders.append(DataLoader(Subset(train_set, p), batch_size=batch_size, shuffle=True, num_workers=0))
        counts.append({
            "Normal": int(np.sum(y[np.asarray(p)] == 0)) if p else 0,
            "Pneumonia": int(np.sum(y[np.asarray(p)] == 1)) if p else 0,
        })
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    class_counts = np.bincount(y, minlength=2).astype(np.float32)
    # Global inverse-frequency weights help avoid a majority-class-only solution.
    weights = class_counts.sum() / (2.0 * np.maximum(class_counts, 1.0))
    class_weights = torch.tensor(weights, dtype=torch.float32)
    return DataBundle(client_loaders, [len(x) for x in chunks], test_loader, validation_loader,
                      train_loader, ["Normal", "Pneumonia"], counts, y, class_weights)
