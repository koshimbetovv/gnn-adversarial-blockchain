import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.optim import Adam
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.elliptic import EllipticDataset
from src.models.gat import GAT


CONFIG = {
    "model": {
        "hidden_dim": 128,
        "num_layers": 4,
        "heads": 4,
        "dropout": 0.2,
        "out_dim": 2,
        "use_norm": True,
    },
    "training": {
        "epochs": 1000,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "grad_clip": 1.0,
        "adam_eps": 1e-5,
        "log_every": 50,  # print train/test metrics every N epochs
    },
    "save": {
        "save_dir": "models",
        "save_run": True,     # saves models/gat_YYYYMMDD_HHMMSS/model.pt
        "filename": "gat.pt"  # used when save_run=False
    }
}


def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def split_mask(data, split: str):
    attr = f"{split}_mask"
    if not hasattr(data, attr):
        raise ValueError(f"Missing {attr} in data. Re-run preprocessing and ensure dataset loads masks.")
    return getattr(data, attr).bool() & (data.y != -1)


def safe_class_weights(y_train, device):
    counts = torch.bincount(y_train, minlength=2).float()
    if (counts == 0).any():
        print(f"⚠ Missing class in TRAIN split (counts={counts.tolist()}), disabling class weights.")
        return None
    n = float(y_train.numel())
    w = n / (2.0 * counts)
    return w.to(device)


@torch.no_grad()
def eval_split(model, data, split: str):
    model.eval()
    mask = split_mask(data, split)

    logits = model(data.x, data.edge_index)[mask]
    y_true = data.y[mask].cpu()
    y_pred = logits.argmax(dim=1).cpu()

    acc = accuracy_score(y_true, y_pred)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    pM, rM, f1M, _ = precision_recall_fscore_support(
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
        "precision_macro": float(pM),
        "recall_macro": float(rM),
        "f1_macro": float(f1M),
        "fraud_predictions": fraud_pred,
        "confusion_matrix_2x2": cm.tolist(),  # [[TN,FP],[FN,TP]]
    }


def main():
    device = get_device()

    data = EllipticDataset().get_data().to(device)
    data.x = data.x.float()
    data.edge_index = data.edge_index.long()

    for s in ["train", "test"]:
        _ = split_mask(data, s)

    model = GAT(
        in_dim=data.num_features,
        hidden_dim=CONFIG["model"]["hidden_dim"],
        out_dim=CONFIG["model"]["out_dim"],
        num_layers=CONFIG["model"]["num_layers"],
        heads=CONFIG["model"]["heads"],
        dropout=CONFIG["model"]["dropout"],
        use_norm=CONFIG["model"]["use_norm"],
    ).to(device)

    train_mask = split_mask(data, "train")
    y_train = data.y[train_mask]
    if y_train.numel() == 0:
        raise ValueError("Train split has 0 labeled nodes (y!=-1). Check preprocessing/masks.")

    class_weights = safe_class_weights(y_train, device)
    print(f"Class weights: {class_weights}" if class_weights is not None else "Class weights: None")

    opt = Adam(
        model.parameters(),
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
        eps=CONFIG["training"]["adam_eps"],
    )

    log_every = CONFIG["training"]["log_every"]

    print("\n=== Training (printing train/test P/R/F1 during training) ===")
    for epoch in range(1, CONFIG["training"]["epochs"] + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        out = model(data.x, data.edge_index)
        logits = out[train_mask]
        labels = data.y[train_mask]

        loss = F.cross_entropy(logits, labels, weight=class_weights)

        if not torch.isfinite(loss):
            print(f"Epoch {epoch}: Loss is not finite (loss={loss.item()}). Stopping.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["training"]["grad_clip"])
        opt.step()

        if epoch == 1 or (epoch % log_every == 0):
            train_metrics = eval_split(model, data, "train")
            test_metrics = eval_split(model, data, "test")

            print(
                f"Epoch {epoch:4d} | Loss={loss.item():.6f} | "
                f"TRAIN: P={train_metrics['precision_pos']:.4f} R={train_metrics['recall_pos']:.4f} F1={train_metrics['f1_pos']:.4f} "
                f"(FraudPred={train_metrics['fraud_predictions']}) | "
                f"TEST: P={test_metrics['precision_pos']:.4f} R={test_metrics['recall_pos']:.4f} F1={test_metrics['f1_pos']:.4f} "
                f"(FraudPred={test_metrics['fraud_predictions']})"
            )

    # Final test eval
    test_metrics = eval_split(model, data, "test")

    print("\n=== GAT Final Test Evaluation ===")
    cm = test_metrics["confusion_matrix_2x2"]
    print(f"[{test_metrics['split']}] #labeled={test_metrics['n_labeled']} | FraudPred={test_metrics['fraud_predictions']}")
    print(f"  Acc={test_metrics['acc']:.4f} | P(pos)={test_metrics['precision_pos']:.4f} R(pos)={test_metrics['recall_pos']:.4f} F1(pos)={test_metrics['f1_pos']:.4f}")
    print(f"  Macro: P={test_metrics['precision_macro']:.4f} R={test_metrics['recall_macro']:.4f} F1={test_metrics['f1_macro']:.4f}")
    print(f"  CM [[TN,FP],[FN,TP]] = {cm}")

    # Save
    save_dir = CONFIG["save"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    if CONFIG["save"]["save_run"]:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(save_dir, f"gat_{run_id}")
        os.makedirs(out_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({"test": test_metrics}, f, indent=2)
        
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(CONFIG, f, indent=2)

        print(f"\n✓ Saved run to {out_dir}")



if __name__ == "__main__":
    main()
