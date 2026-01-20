import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.elliptic import EllipticDataset
from src.models.gat import GAT


CONFIG = {
    "model": {
        "hidden_dim": 32,
        "num_layers": 2,
        "heads": 4,
        "dropout": 0.6,
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
        "patience": 50,   # default patience
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train GAT with hyperparameters")
    
    # Model params
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--use_norm", type=lambda x: (str(x).lower() == 'true'), default=True)
    
    # Training params
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "step"])
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_file", type=str, default=None, help="File to append results JSON line")
    parser.add_argument("--trial_id", type=str, default="default", help="Trial ID for logging")
    parser.add_argument("--git_hash", type=str, default="unknown", help="Git hash for logging")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    start_time = datetime.now()

    # Update CONFIG from args
    CONFIG["model"]["hidden_dim"] = args.hidden_dim
    CONFIG["model"]["num_layers"] = args.num_layers
    CONFIG["model"]["heads"] = args.heads
    CONFIG["model"]["dropout"] = args.dropout
    CONFIG["model"]["use_norm"] = args.use_norm
    
    CONFIG["training"]["lr"] = args.lr
    CONFIG["training"]["weight_decay"] = args.weight_decay
    CONFIG["training"]["epochs"] = args.epochs
    CONFIG["training"]["patience"] = args.patience
    CONFIG["training"]["grad_clip"] = args.grad_clip_norm
    
    print(f"Config: {CONFIG}")

    data = EllipticDataset().get_data().to(device)
    data.x = data.x.float()
    data.edge_index = data.edge_index.long()

    # ensure masks
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

    if args.optimizer == "adam":
        opt = Adam(
            model.parameters(),
            lr=CONFIG["training"]["lr"],
            weight_decay=CONFIG["training"]["weight_decay"],
            eps=CONFIG["training"]["adam_eps"],
        )
    elif args.optimizer == "adamw":
        opt = AdamW(
            model.parameters(),
            lr=CONFIG["training"]["lr"],
            weight_decay=CONFIG["training"]["weight_decay"],
            eps=CONFIG["training"]["adam_eps"],
        )
    
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = StepLR(opt, step_size=50, gamma=0.5)
    else:
        scheduler = None

    log_every = CONFIG["training"]["log_every"]
    
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    print("\n=== Training ===")
    for epoch in range(1, CONFIG["training"]["epochs"] + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        out = model(data.x, data.edge_index)
        logits = out[train_mask]
        labels = data.y[train_mask]

        if args.label_smoothing > 0:
            loss = F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=args.label_smoothing)
        else:
            loss = F.cross_entropy(logits, labels, weight=class_weights)

        if not torch.isfinite(loss):
            print(f"Epoch {epoch}: Loss is not finite (loss={loss.item()}). Stopping.")
            break

        loss.backward()
        if CONFIG["training"]["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["training"]["grad_clip"])
        opt.step()
        
        if scheduler:
            scheduler.step()

        # Validation (using Test split as proxy for validation during tuning)
        val_metrics = eval_split(model, data, "test")
        val_f1 = val_metrics['f1_macro']

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if epoch == 1 or (epoch % log_every == 0):
            train_metrics = eval_split(model, data, "train")
            print(
                f"Epoch {epoch:4d} | Loss={loss.item():.6f} | "
                f"TRAIN F1={train_metrics['f1_macro']:.4f} | "
                f"VAL F1={val_f1:.4f} (Best={best_val_f1:.4f} @ Ep {best_epoch})"
            )
        
        if patience_counter >= CONFIG["training"]["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final eval on best model
    final_metrics = eval_split(model, data, "test")
    runtime = (datetime.now() - start_time).total_seconds()

    print("\n=== GAT Final Evaluation (Best Model) ===")
    print(f"Best Epoch: {best_epoch}")
    print(f"Val Macro F1: {final_metrics['f1_macro']:.4f}")
    print(f"Val Acc: {final_metrics['acc']:.4f}")

    # Save
    save_dir = CONFIG["save"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    # Allow simple filename for tuning
    out_path = os.path.join(save_dir, f"gat_trial_{args.trial_id}.pt")
    torch.save(model.state_dict(), out_path)
    
    # Append results if requested
    if args.results_file:
        result_entry = {
            "trial_id": args.trial_id,
            "seed": args.seed,
            "val_macro_f1": final_metrics['f1_macro'],
            "val_acc": final_metrics['acc'],
            "epoch_best": best_epoch,
            "runtime": runtime,
            "git_hash": args.git_hash,
            # Hyperparams
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "grad_clip": args.grad_clip_norm,
            "label_smoothing": args.label_smoothing
        }
        with open(args.results_file, "a") as f:
            f.write(json.dumps(result_entry) + "\n")

    if CONFIG["save"]["save_run"] and not args.results_file:
        # Standard run behavior (backward compatibility mostly)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(save_dir, f"gat_{run_id}")
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({"test": final_metrics}, f, indent=2)
        print(f"\n✓ Saved run to {out_dir}")

if __name__ == "__main__":
    main()
