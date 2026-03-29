import json
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.chronowave_ellipticpp_actors import (
    ChronoWaveActorsConfig,
    ChronoWaveEllipticPPActorsDataset,
)
from src.models.chronowave_gnn import ChronoWaveGNN
from scripts.train_paper_utils import get_device


CONFIG = {
    "data": {
        "feature_path": "data/raw/ellipticpp/actors/wallets_features.csv",
        "class_path": "data/raw/ellipticpp/actors/wallets_classes.csv",
        "edge_path": "data/raw/ellipticpp/actors/AddrAddr_edgelist.csv",
        "train_start": 1,
        "train_end": 34,
        "test_start": 35,
        "test_end": 49,
        "filter_unknown": True,
        "wavelet": "haar",
        "wavelet_level": 2,
    },
    "model": {
        "hidden_dim": 256,
        "time_dim": 8,
        "heads": 2,
        "num_layers": 3,
        "dropout": 0.4,
        "out_dim": 2,
    },
    "training": {
        "epochs": 300,
        "lr": 5e-3,
        "weight_decay": 5e-4,
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
        "log_every": 10,
        "use_class_weights": True,
    },
    "save": {
        "save_dir": "models",
        "save_run": True,
        "prefix": "chronowave_gnn_ellipticpp_actors",
    },
}


def split_mask(data, split: str) -> torch.Tensor:
    attr = f"{split}_mask"
    if not hasattr(data, attr):
        raise ValueError(f"Missing {attr} in data.")
    return getattr(data, attr).bool() & (data.y != -1)


def safe_class_weights(y_train: torch.Tensor, device: torch.device):
    counts = torch.bincount(y_train, minlength=2).float()
    if (counts == 0).any():
        print(f"⚠ Missing class in TRAIN split (counts={counts.tolist()}), disabling class weights.")
        return None
    n = float(y_train.numel())
    w = n / (2.0 * counts)
    return w.to(device)


def forward_model(model: ChronoWaveGNN, data):
    return model(data.x, data.edge_index, data.time_step)


@torch.no_grad()
def eval_split(model: ChronoWaveGNN, data, split: str) -> dict:
    model.eval()
    mask = split_mask(data, split)
    logits = forward_model(model, data)[mask]
    y_true = data.y[mask].cpu()
    y_pred = logits.argmax(dim=1).cpu()

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fraud_pred = int((y_pred == 1).sum().item())

    return {
        "split": split,
        "n_labeled": int(mask.sum().item()),
        "acc": float(acc),
        "precision_pos": float(p[1]),
        "recall_pos": float(r[1]),
        "f1_pos": float(f1[1]),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "fraud_predictions": fraud_pred,
        "confusion_matrix_2x2": cm.tolist(),
    }


def print_epoch_metrics(epoch: int, loss: float, lr: float, train_metrics: dict, test_metrics: dict):
    print(
        f"Epoch {epoch:4d} | Loss={loss:.6f} | LR={lr:.6f} | "
        f"TRAIN: P={train_metrics['precision_pos']:.4f} R={train_metrics['recall_pos']:.4f} "
        f"F1={train_metrics['f1_pos']:.4f} (FraudPred={train_metrics['fraud_predictions']}) | "
        f"TEST: P={test_metrics['precision_pos']:.4f} R={test_metrics['recall_pos']:.4f} "
        f"F1={test_metrics['f1_pos']:.4f} (FraudPred={test_metrics['fraud_predictions']})"
    )


def print_final_metrics(title: str, metrics: dict):
    cm = metrics["confusion_matrix_2x2"]
    print(f"\n=== {title} Final Test Evaluation ===")
    print(f"[{metrics['split']}] #labeled={metrics['n_labeled']} | FraudPred={metrics['fraud_predictions']}")
    print(
        f"  Acc={metrics['acc']:.4f} | "
        f"P(pos)={metrics['precision_pos']:.4f} "
        f"R(pos)={metrics['recall_pos']:.4f} "
        f"F1(pos)={metrics['f1_pos']:.4f}"
    )
    print(
        f"  Macro: P={metrics['precision_macro']:.4f} "
        f"R={metrics['recall_macro']:.4f} "
        f"F1={metrics['f1_macro']:.4f}"
    )
    print(f"  CM [[TN,FP],[FN,TP]] = {cm}")


def save_run(model: ChronoWaveGNN, metrics: dict, config: dict):
    save_dir = config["save"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    if config["save"].get("save_run", True):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = config["save"]["prefix"]
        out_dir = os.path.join(save_dir, f"{prefix}_{run_id}")
        os.makedirs(out_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({"test": metrics}, f, indent=2)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Saved run to {out_dir}")


def main():
    device = get_device()

    dataset = ChronoWaveEllipticPPActorsDataset(
        ChronoWaveActorsConfig(**CONFIG["data"])
    )
    data = dataset.get_data().to(device)

    train_mask = split_mask(data, "train")
    y_train = data.y[train_mask]
    if y_train.numel() == 0:
        raise ValueError("Train split has 0 labeled nodes after unknown filtering.")

    class_weights = None
    if CONFIG["training"].get("use_class_weights", False):
        class_weights = safe_class_weights(y_train, device)
    print(f"Class weights: {class_weights}" if class_weights is not None else "Class weights: None")

    model = ChronoWaveGNN(
        in_dim=data.x.size(1),
        hidden_dim=CONFIG["model"]["hidden_dim"],
        out_dim=CONFIG["model"]["out_dim"],
        time_dim=CONFIG["model"]["time_dim"],
        heads=CONFIG["model"]["heads"],
        num_layers=CONFIG["model"]["num_layers"],
        dropout=CONFIG["model"]["dropout"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["training"]["epochs"])

    epochs = CONFIG["training"]["epochs"]
    log_every = CONFIG["training"].get("log_every", 10)
    grad_clip = CONFIG["training"].get("grad_clip", 1.0)
    label_smoothing = CONFIG["training"].get("label_smoothing", 0.1)

    print("\n=== Training ChronoWave-GNN on Elliptic++ Actors (train/test only) ===")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = forward_model(model, data)
        loss = F.cross_entropy(
            logits[train_mask],
            data.y[train_mask],
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}: {loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        if epoch == 1 or (epoch % log_every == 0) or epoch == epochs:
            train_metrics = eval_split(model, data, "train")
            test_metrics = eval_split(model, data, "test")
            current_lr = optimizer.param_groups[0]["lr"]
            print_epoch_metrics(epoch, float(loss.item()), float(current_lr), train_metrics, test_metrics)

    test_metrics = eval_split(model, data, "test")
    print_final_metrics("ChronoWave-GNN (Elliptic++ Actors)", test_metrics)
    save_run(model, test_metrics, CONFIG)


if __name__ == "__main__":
    main()
