import copy
import random
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score, classification_report,
                             balanced_accuracy_score)
from .model import SmallCNN

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _probabilities(model, loader, device):
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.reshape(-1).long().to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1]
            ys.extend(y.cpu().tolist())
            probs.extend(p.cpu().tolist())
    return np.asarray(ys, dtype=int), np.asarray(probs, dtype=float)

def choose_threshold(model, validation_loader, device):
    """Choose a binary decision threshold on held-out validation data, never on test data."""
    ys, probs = _probabilities(model, validation_loader, device)
    if len(np.unique(ys)) < 2:
        return 0.5
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probs)))
    best_threshold, best_score, best_tie = 0.5, -1.0, float("inf")
    for threshold in candidates:
        preds = (probs >= threshold).astype(int)
        score = balanced_accuracy_score(ys, preds)
        tie = abs(float(threshold) - 0.5)
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and tie < best_tie):
            best_threshold, best_score, best_tie = float(threshold), float(score), tie
    return best_threshold

def evaluate(model, loader, device, threshold=0.5):
    ys, probs = _probabilities(model, loader, device)
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(ys, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auc = float(roc_auc_score(ys, probs))
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(ys, preds)),
        "precision": float(precision_score(ys, preds, zero_division=0)),
        "recall_sensitivity": float(recall_score(ys, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(ys, preds)),
        "threshold": float(threshold),
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(ys, preds, labels=[0, 1], target_names=["Normal", "Pneumonia"], output_dict=True, zero_division=0),
        "y_true": ys.tolist(), "y_pred": preds.tolist(), "y_prob": probs.tolist(),
    }

def train_one(model, loader, device, epochs=1, lr=0.001, class_weights=None):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    weights = class_weights.to(device) if class_weights is not None else None
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.reshape(-1).long().to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    return model

def fedavg(states, sizes):
    if len(states) != len(sizes) or not states:
        raise ValueError("Invalid client updates.")
    total = float(sum(sizes))
    if total <= 0:
        raise ValueError("Cannot aggregate client updates with zero total samples.")
    out = copy.deepcopy(states[0])
    for k in out:
        if torch.is_floating_point(out[k]):
            out[k] = sum(s[k] * (n / total) for s, n in zip(states, sizes))
        else:
            out[k] = states[0][k].clone()
    return out

def run_federated(bundle, rounds=3, local_epochs=1, seed=42, progress=None):
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(2).to(device)
    history = []
    start_all = time.time()
    class_weights = bundle.class_weights
    for r in range(1, rounds + 1):
        start = time.time()
        states = []
        for loader in bundle.client_loaders:
            client = SmallCNN(2).to(device)
            client.load_state_dict(copy.deepcopy(model.state_dict()))
            train_one(client, loader, device, local_epochs, class_weights=class_weights)
            states.append({k: v.detach().cpu().clone() for k, v in client.state_dict().items()})
            del client
        model.load_state_dict(fedavg(states, bundle.client_sizes))
        threshold = choose_threshold(model, bundle.validation_loader, device)
        m = evaluate(model, bundle.test_loader, device, threshold=threshold)
        row = {"round": r, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
               "recall_sensitivity": m["recall_sensitivity"], "specificity": m["specificity"],
               "balanced_accuracy": m["balanced_accuracy"], "roc_auc": m["roc_auc"],
               "decision_threshold": threshold, "seconds": time.time() - start}
        history.append(row)
        if progress:
            progress(r, rounds, row)
    final_threshold = choose_threshold(model, bundle.validation_loader, device)
    final = evaluate(model, bundle.test_loader, device, threshold=final_threshold)
    return {"model": model, "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "history": history, "final": final, "threshold": final_threshold,
            "elapsed": time.time() - start_all, "device": str(device)}

def run_local_baselines(bundle, epochs=3, seed=42):
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, loader in enumerate(bundle.client_loaders, 1):
        seed_all(seed + i)
        model = SmallCNN(2).to(device)
        train_one(model, loader, device, epochs, class_weights=bundle.class_weights)
        threshold = choose_threshold(model, bundle.validation_loader, device)
        m = evaluate(model, bundle.test_loader, device, threshold=threshold)
        rows.append({"method": f"Local-only client {i}", "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "recall_sensitivity": m["recall_sensitivity"], "specificity": m["specificity"],
                     "balanced_accuracy": m["balanced_accuracy"], "roc_auc": m["roc_auc"],
                     "decision_threshold": threshold})
    return rows

def run_centralized(bundle, epochs=3, seed=42):
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(2).to(device)
    train_one(model, bundle.train_loader, device, epochs, class_weights=bundle.class_weights)
    threshold = choose_threshold(model, bundle.validation_loader, device)
    m = evaluate(model, bundle.test_loader, device, threshold=threshold)
    return {"method": "Centralised baseline", "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "recall_sensitivity": m["recall_sensitivity"], "specificity": m["specificity"],
            "balanced_accuracy": m["balanced_accuracy"], "roc_auc": m["roc_auc"],
            "decision_threshold": threshold}
