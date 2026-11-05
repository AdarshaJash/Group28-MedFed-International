"""Unified loaders for MedMNIST image benchmarks and user-supplied labelled CSV data.

CSV data is never fabricated. The user supplies a legally usable, documented dataset.
"""
from dataclasses import dataclass
from io import BytesIO
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms

@dataclass
class UnifiedBundle:
    client_loaders: list
    client_sizes: list
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    class_names: list
    client_label_counts: list
    class_weights: torch.Tensor
    task_type: str
    input_dim: int = 0
    dataset_name: str = ""
    feature_names: list | None = None

class _LabelledImageDataset(Dataset):
    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, list(indices), transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        image, label = self.base[self.indices[i]]
        if not isinstance(image, torch.Tensor): image = self.transform(image)
        label = int(np.asarray(label).reshape(-1)[0])
        return image, torch.tensor(label, dtype=torch.long)

def _partition_indices(labels, clients, mode, seed, alpha=0.35):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    if mode.upper() == "IID":
        chunks = [x.tolist() for x in np.array_split(rng.permutation(len(labels)), clients)]
    else:
        buckets = [[] for _ in range(clients)]
        for cls in np.unique(labels):
            ids = rng.permutation(np.where(labels == cls)[0])
            props = rng.dirichlet(np.repeat(alpha, clients))
            cuts = (np.cumsum(props) * len(ids)).astype(int)[:-1]
            for j, part in enumerate(np.split(ids, cuts)): buckets[j].extend(part.tolist())
        chunks = [rng.permutation(x).tolist() for x in buckets]
    # Avoid empty clients for very small datasets.
    for i in range(clients):
        if not chunks[i]:
            donor = max(range(clients), key=lambda j: len(chunks[j]))
            if len(chunks[donor]) > 1: chunks[i].append(chunks[donor].pop())
    return chunks

def _weights(labels, n_classes):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes).astype(np.float32)
    w = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float32)

def load_medmnist_bundle(dataset_key: str, num_clients=3, partition="Non-IID", max_train=3000,
                         max_test=1000, batch_size=64, seed=42, image_size=28):
    import medmnist
    from medmnist import INFO
    if dataset_key not in {"pneumoniamnist", "organamnist", "breastmnist"}:
        raise ValueError("Unsupported image dataset key.")
    info = INFO[dataset_key]
    DatasetClass = getattr(medmnist, info["python_class"])
    tfm = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                               transforms.Resize((image_size, image_size)),
                               transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_base = DatasetClass(split="train", transform=tfm, download=True)
    val_base = DatasetClass(split="val", transform=tfm, download=True)
    test_base = DatasetClass(split="test", transform=tfm, download=True)
    labels_all = np.asarray(train_base.labels).reshape(len(train_base), -1)[:, 0].astype(int)
    rng = np.random.default_rng(seed)
    chosen = []
    classes, counts = np.unique(labels_all, return_counts=True)
    limit = min(int(max_train), len(train_base))
    exact = counts / counts.sum() * limit
    quotas = np.floor(exact).astype(int)
    for j in np.argsort(-(exact - quotas))[:limit - int(quotas.sum())]: quotas[j] += 1
    for cls, quota in zip(classes, quotas):
        ids = np.where(labels_all == cls)[0]; rng.shuffle(ids)
        chosen.extend(ids[:int(quota)].tolist())
    rng.shuffle(chosen)
    labels = labels_all[chosen]
    chunks = _partition_indices(labels, num_clients, partition, seed)
    train_ds = _LabelledImageDataset(train_base, chosen, tfm)
    # Keep official validation/test splits separate from training.
    val_ds = _LabelledImageDataset(val_base, range(len(val_base)), tfm)
    test_ids = np.arange(len(test_base))
    if max_test and len(test_ids) > max_test:
        test_labels = np.asarray(test_base.labels).reshape(len(test_base), -1)[:, 0].astype(int)
        picked=[]
        for cls in np.unique(test_labels):
            ids=np.where(test_labels==cls)[0]; rng.shuffle(ids)
            picked.extend(ids[:max(1, round(max_test*len(ids)/len(test_base)))].tolist())
        test_ids=np.asarray(picked[:max_test], dtype=int)
    test_ds = _LabelledImageDataset(test_base, test_ids, tfm)
    loaders=[]; counts=[]
    for part in chunks:
        loaders.append(DataLoader(Subset(train_ds, part), batch_size=batch_size, shuffle=True, num_workers=0))
        counts.append({str(c): int(np.sum(labels[np.asarray(part, dtype=int)] == c)) for c in np.unique(labels)})
    names = info.get("label", {})
    class_names = [str(names.get(str(i), names.get(i, i))) for i in range(len(set(labels_all.tolist())))]
    return UnifiedBundle(loaders, [len(x) for x in chunks],
        DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0),
        DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=0),
        DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=0), class_names, counts,
        _weights(labels, len(class_names)), "image", dataset_name=info.get("python_class", dataset_key))

def load_csv_bundle(file_bytes, target_column: str, num_clients=3, partition="Non-IID", batch_size=64,
                    seed=42, test_fraction=0.2, validation_fraction=0.15):
    df = pd.read_csv(BytesIO(file_bytes))
    if target_column not in df.columns: raise ValueError("Target column was not found in the CSV.")
    df = df.dropna(subset=[target_column]).copy()
    y_raw = df.pop(target_column).astype(str)
    # Exclude obvious identifier columns; users must also review domain-specific leakage columns.
    id_like = [c for c in df.columns if c.lower() in {"id", "patient_id", "subject_id", "record_id"}]
    df = df.drop(columns=id_like, errors="ignore")
    if df.shape[1] == 0: raise ValueError("No feature columns remain after excluding identifiers.")
    classes = sorted(y_raw.unique().tolist())
    if len(classes) < 2: raise ValueError("The target must contain at least two classes.")
    mapping = {v:i for i,v in enumerate(classes)}
    y = y_raw.map(mapping).to_numpy(dtype=np.int64)
    if len(y) < max(30, num_clients * 5): raise ValueError("Please provide at least 30 labelled rows and enough examples for each client.")
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    indices = np.arange(len(y))
    try:
        train_val, test_idx = train_test_split(indices, test_size=test_fraction, random_state=seed, stratify=y)
        train_idx, val_idx = train_test_split(train_val, test_size=validation_fraction, random_state=seed+1, stratify=y[train_val])
    except ValueError as e:
        raise ValueError(f"Could not create stratified splits. Ensure each target class has enough rows. Details: {e}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    transformers=[]
    if numeric_cols:
        numeric_pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
        transformers.append(("numeric",numeric_pipe,numeric_cols))
    if categorical_cols:
        categorical_pipe=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
        transformers.append(("categorical",categorical_pipe,categorical_cols))
    preprocessor=ColumnTransformer(transformers,remainder="drop")
    x_train=preprocessor.fit_transform(df.iloc[train_idx]).astype(np.float32)
    x_val=preprocessor.transform(df.iloc[val_idx]).astype(np.float32)
    x_test=preprocessor.transform(df.iloc[test_idx]).astype(np.float32)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    if x_train.shape[1] == 0: raise ValueError("No usable feature columns remain after preprocessing.")
    chunks = _partition_indices(y_train, num_clients, partition, seed)
    def tensor_ds(x, yy): return TensorDataset(torch.tensor(x), torch.tensor(yy, dtype=torch.long))
    train_ds, val_ds, test_ds = tensor_ds(x_train,y_train), tensor_ds(x_val,y_val), tensor_ds(x_test,y_test)
    loaders=[]; counts=[]
    for part in chunks:
        loaders.append(DataLoader(Subset(train_ds,part),batch_size=batch_size,shuffle=True,num_workers=0))
        counts.append({classes[c]:int(np.sum(y_train[np.asarray(part)]==c)) for c in range(len(classes))})

    return UnifiedBundle(loaders,[len(x) for x in chunks],DataLoader(train_ds,batch_size=batch_size,shuffle=True),
        DataLoader(val_ds,batch_size=batch_size,shuffle=False),DataLoader(test_ds,batch_size=batch_size,shuffle=False),
        classes,counts,_weights(y_train,len(classes)),"tabular",input_dim=x_train.shape[1],
        dataset_name="User-supplied labelled CSV",feature_names=[str(c) for c in df.columns])
