import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.elliptic import EllipticDataset
from src.models.gcn import GCN
from src.training.loss import FocalLoss


CONFIG = {
    "model": {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.4,
        "out_dim": 2,
        "use_norm": True,   # your GCN supports this
    },
    "training": {
        "epochs": 1500,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "grad_clip": 1.0,
        "adam_eps": 1e-5,
        "log_every": 50,

        # ---- loss options ----
        # "ce" or "focal"
        "loss": "ce",

        # CE: if True, use class weights computed on TRAIN_ONLY (train split minus dev subset)
        "use_class_weights": True,

        # Focal Loss params (used only if loss="focal")
        # alpha is per-class multiplier [alpha_legit, alpha_fraud]
        "focal_alpha": [1.0, 2.0],
        "focal_gamma": 2.0,

        # ---- threshold tuning ----
        # dev subset from train for tuning tau; set 0.0 to disable
        "dev_frac": 0.10,
        "seed": 0,
        # threshold grid for tuning tau (fraud probability threshold)
        "tau_min": 0.05,
        "tau_max": 0.95,
        "tau_steps": 91,   # 0.01 resolution over [0.05,0.95]
    },
    "save": {
        "save_dir": "models",
        "save_run": True,
        "filename": "gcn.pt"
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
    """
    Class weights computed on TRAIN_ONLY. If a class is missing, disable weights.
    """
    num_classes = 2
    counts = torch.bincount(y_train, minlength=num_classes).float()
    if (counts == 0).any():
        print(f"⚠ Missing class in TRAIN_ONLY (counts={counts.tolist()}), disabling class weights.")
        return None
    n = float(y_train.numel())
    w = n / (num_classes * counts)
    return w.to(device)


def make_train_dev_masks(data, train_mask, dev_frac=0.1, seed=0):
    """
    Split TRAIN into TRAIN_ONLY + DEV (stratified by class within labeled train).
    dev_frac=0 disables dev split (dev_mask all False, train_only=train_mask&labeled).
    """
    labeled_train = train_mask & (data.y != -1)

    if dev_frac <= 0.0:
        dev_mask = torch.zeros_like(labeled_train, dtype=torch.bool)
        train_only = labeled_train
        return train_only, dev_mask

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    idx = torch.where(labeled_train.cpu())[0]
    y = data.y.cpu()[idx]

    dev_idx_chunks = []
    for c in [0, 1]:
        c_idx = idx[y == c]
        if c_idx.numel() == 0:
            continue
        perm = c_idx[torch.randperm(c_idx.numel(), generator=g)]
        k = int(dev_frac * perm.numel())
        k = max(1, k) if perm.numel() > 0 else 0
        dev_idx_chunks.append(perm[:k])

    dev_idx = torch.cat(dev_idx_chunks) if len(dev_idx_chunks) else torch.tensor([], dtype=torch.long)

    dev_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    if dev_idx.numel() > 0:
        dev_mask[dev_idx] = True

    train_only = labeled_train.cpu() & (~dev_mask)
    return train_only.to(train_mask.device), dev_mask.to(train_mask.device)


@torch.no_grad()
def eval_mask_argmax(model, data, mask: torch.Tensor):
    model.eval()
    logits = model(data.x, data.edge_index)[mask]
    y_true = data.y[mask].cpu()
    y_pred = logits.argmax(dim=1).cpu()

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    pM, rM, f1M, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fraud_pred = int((y_pred == 1).sum().item())

    return {
        "n_labeled": int(mask.sum().item()),
        "acc": float(acc),
        "precision_pos": float(p[1]),
        "recall_pos": float(r[1]),
        "f1_pos": float(f1[1]),
        "precision_macro": float(pM),
        "recall_macro": float(rM),
        "f1_macro": float(f1M),
        "fraud_predictions": fraud_pred,
        "confusion_matrix_2x2": cm.tolist(),
    }


@torch.no_grad()
def eval_mask_threshold(model, data, mask: torch.Tensor, tau: float):
    model.eval()
    logits = model(data.x, data.edge_index)[mask]
    y_true = data.y[mask].cpu()

    prob_pos = torch.softmax(logits, dim=1)[:, 1].cpu()
    y_pred = (prob_pos >= tau).long()

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    pM, rM, f1M, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fraud_pred = int((y_pred == 1).sum().item())

    return {
        "tau": float(tau),
        "n_labeled": int(mask.sum().item()),
        "acc": float(acc),
        "precision_pos": float(p[1]),
        "recall_pos": float(r[1]),
        "f1_pos": float(f1[1]),
        "precision_macro": float(pM),
        "recall_macro": float(rM),
        "f1_macro": float(f1M),
        "fraud_predictions": fraud_pred,
        "confusion_matrix_2x2": cm.tolist(),
    }


@torch.no_grad()
def find_best_threshold(model, data, dev_mask: torch.Tensor, tau_min=0.05, tau_max=0.95, steps=91):
    grid = torch.linspace(float(tau_min), float(tau_max), int(steps)).tolist()
    best_tau, best_f1, best_metrics = 0.5, -1.0, None
    for tau in grid:
        m = eval_mask_threshold(model, data, dev_mask, tau=tau)
        if m["f1_pos"] > best_f1:
            best_f1 = m["f1_pos"]
            best_tau = tau
            best_metrics = m
    return best_tau, best_metrics


def main():
    device = get_device()

    data = EllipticDataset().get_data().to(device)
    data.x = data.x.float()
    data.edge_index = data.edge_index.long()

    # masks exist
    train_mask_full = split_mask(data, "train")
    test_mask = split_mask(data, "test")

    # train/dev split inside train
    train_only_mask, dev_mask = make_train_dev_masks(
        data, train_mask_full,
        dev_frac=CONFIG["training"]["dev_frac"],
        seed=CONFIG["training"]["seed"]
    )

    print(f"#train_full={int(train_mask_full.sum())} | #train_only={int(train_only_mask.sum())} | #dev={int(dev_mask.sum())} | #test={int(test_mask.sum())}")

    model = GCN(
        in_dim=data.num_features,
        hidden_dim=CONFIG["model"]["hidden_dim"],
        out_dim=CONFIG["model"]["out_dim"],
        num_layers=CONFIG["model"]["num_layers"],
        dropout=CONFIG["model"]["dropout"],
        use_norm=CONFIG["model"].get("use_norm", True),
    ).to(device)

    # loss setup
    loss_mode = CONFIG["training"]["loss"].lower()
    if loss_mode == "focal":
        criterion = FocalLoss(
            alpha=CONFIG["training"]["focal_alpha"],
            gamma=CONFIG["training"]["focal_gamma"],
        ).to(device)
        class_weights = None
        print(f"Loss: FOCAL (alpha={CONFIG['training']['focal_alpha']}, gamma={CONFIG['training']['focal_gamma']})")
    else:
        criterion = None
        class_weights = None
        if CONFIG["training"]["use_class_weights"]:
            y_train_only = data.y[train_only_mask]
            class_weights = safe_class_weights(y_train_only, device)
        print(f"Loss: CE (class_weights={class_weights})")

    opt = Adam(
        model.parameters(),
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
        eps=CONFIG["training"]["adam_eps"],
    )

    log_every = CONFIG["training"]["log_every"]

    print("\n=== Training (printing TRAIN+TEST P/R/F1 with argmax during training) ===")
    for epoch in range(1, CONFIG["training"]["epochs"] + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        out = model(data.x, data.edge_index)

        logits = out[train_only_mask]
        labels = data.y[train_only_mask]

        if loss_mode == "focal":
            loss = criterion(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels, weight=class_weights)

        if not torch.isfinite(loss):
            print(f"Epoch {epoch}: Loss is not finite (loss={loss.item()}). Stopping.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["training"]["grad_clip"])
        opt.step()

        if epoch == 1 or (epoch % log_every == 0):
            train_metrics = eval_mask_argmax(model, data, train_only_mask)
            test_metrics_mid = eval_mask_argmax(model, data, test_mask)

            print(
                f"Epoch {epoch:4d} | Loss={loss.item():.6f} | "
                f"TRAIN: P={train_metrics['precision_pos']:.4f} R={train_metrics['recall_pos']:.4f} F1={train_metrics['f1_pos']:.4f} "
                f"(FraudPred={train_metrics['fraud_predictions']}) | "
                f"TEST: P={test_metrics_mid['precision_pos']:.4f} R={test_metrics_mid['recall_pos']:.4f} F1={test_metrics_mid['f1_pos']:.4f} "
                f"(FraudPred={test_metrics_mid['fraud_predictions']})"
            )

    # ---- Final evaluation ----
    test_argmax = eval_mask_argmax(model, data, test_mask)

    # threshold tuning (on dev only)
    tuned = None
    best_tau = None
    if dev_mask.sum().item() > 0:
        best_tau, dev_best = find_best_threshold(
            model, data, dev_mask,
            tau_min=CONFIG["training"]["tau_min"],
            tau_max=CONFIG["training"]["tau_max"],
            steps=CONFIG["training"]["tau_steps"],
        )
        tuned = eval_mask_threshold(model, data, test_mask, tau=best_tau)

        print("\n=== Threshold tuning (DEV) ===")
        print(f"Best tau on DEV: {best_tau:.3f} | DEV F1_pos={dev_best['f1_pos']:.4f} P={dev_best['precision_pos']:.4f} R={dev_best['recall_pos']:.4f}")

    print("\n=== Final TEST Evaluation (argmax) ===")
    cm = test_argmax["confusion_matrix_2x2"]
    print(f"#labeled={test_argmax['n_labeled']} | FraudPred={test_argmax['fraud_predictions']}")
    print(f"Acc={test_argmax['acc']:.4f} | P(pos)={test_argmax['precision_pos']:.4f} R(pos)={test_argmax['recall_pos']:.4f} F1(pos)={test_argmax['f1_pos']:.4f}")
    print(f"Macro: P={test_argmax['precision_macro']:.4f} R={test_argmax['recall_macro']:.4f} F1={test_argmax['f1_macro']:.4f}")
    print(f"CM [[TN,FP],[FN,TP]] = {cm}")

    if tuned is not None:
        print("\n=== Final TEST Evaluation (tuned threshold) ===")
        cm2 = tuned["confusion_matrix_2x2"]
        print(f"tau={tuned['tau']:.3f} | #labeled={tuned['n_labeled']} | FraudPred={tuned['fraud_predictions']}")
        print(f"Acc={tuned['acc']:.4f} | P(pos)={tuned['precision_pos']:.4f} R(pos)={tuned['recall_pos']:.4f} F1(pos)={tuned['f1_pos']:.4f}")
        print(f"Macro: P={tuned['precision_macro']:.4f} R={tuned['recall_macro']:.4f} F1={tuned['f1_macro']:.4f}")
        print(f"CM [[TN,FP],[FN,TP]] = {cm2}")

    # ---- Save ----
    save_dir = CONFIG["save"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    if CONFIG["save"]["save_run"]:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(save_dir, f"gcn_{run_id}")
        os.makedirs(out_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

        payload = {"test_argmax": test_argmax}
        if tuned is not None:
            payload["test_tuned"] = tuned
            payload["best_tau"] = float(best_tau)

        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(payload, f, indent=2)

        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(CONFIG, f, indent=2)

        print(f"\n✓ Saved run to {out_dir}")


if __name__ == "__main__":
    main()
