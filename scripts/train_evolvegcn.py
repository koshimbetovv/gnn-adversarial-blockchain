import copy
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.evolvegcn_elliptic import EvolveGCNEllipticConfig, EvolveGCNEllipticDataset
from src.models.evolvegcn import (
    EvolveGCNClassifier,
    EvolveGCNH,
    EvolveGCNNodeClassifier,
    EvolveGCNO,
)


BASE_CONFIGS = {
    "h": {
        "data": {
            "feature_path": "data/raw/elliptic/elliptic_txs_features.csv",
            "class_path": "data/raw/elliptic/elliptic_txs_classes.csv",
            "edge_path": "data/raw/elliptic/elliptic_txs_edgelist.csv",
            "num_hist_steps": 5,
            "adj_mat_time_window": 1,
            "train_start": 1,
            "train_end": 34,
            "test_start": 35,
            "test_end": 49,
        },
        "model": {
            "variant": "h",
            "layer_1_feats": 76,
            "layer_2_feats": 76,
            "cls_feats": 510,
            "skipfeats": False,
        },
        "training": {
            "epochs": 1000,
            "lr": 1e-3,
            "class_weights": "auto",
            "log_every": 20,
        },
        "save": {
            "save_dir": "models",
            "save_run": True,
        },
    },
    "o": {
        "data": {
            "feature_path": "data/raw/elliptic/elliptic_txs_features.csv",
            "class_path": "data/raw/elliptic/elliptic_txs_classes.csv",
            "edge_path": "data/raw/elliptic/elliptic_txs_edgelist.csv",
            "num_hist_steps": 5,
            "adj_mat_time_window": 1,
            "train_start": 1,
            "train_end": 34,
            "test_start": 35,
            "test_end": 49,
        },
        "model": {
            "variant": "o",
            "layer_1_feats": 256,
            "layer_2_feats": 256,
            "cls_feats": 307,
            "skipfeats": False,
        },
        "training": {
            "epochs": 800,
            "lr": 1e-3,
            "class_weights": "auto",
            "log_every": 20,
        },
        "save": {
            "save_dir": "models",
            "save_run": True,
        },
    },
}

# Change this to "o" to train the exact IBM repo EvolveGCN-O variant.
CONFIG = copy.deepcopy(BASE_CONFIGS["h"])


class WeightedCrossEntropy(torch.nn.Module):
    """
    Exact weighted CE used by the IBM EvolveGCN repo.
    """

    def __init__(self, class_weights: list[float], device: torch.device):
        super().__init__()
        self.weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    @staticmethod
    def logsumexp(logits: torch.Tensor) -> torch.Tensor:
        m, _ = torch.max(logits, dim=1)
        m = m.view(-1, 1)
        sum_exp = torch.sum(torch.exp(logits - m), dim=1, keepdim=True)
        return m + torch.log(sum_exp)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1, 1)
        alpha = self.weights[labels].view(-1, 1)
        loss = alpha * (-logits.gather(-1, labels) + self.logsumexp(logits))
        return loss.mean()




def resolve_class_weights(train_samples, config: dict, device: torch.device) -> list[float]:
    cfg_weights = config["training"].get("class_weights", "auto")
    if cfg_weights != "auto" and cfg_weights is not None:
        return [float(cfg_weights[0]), float(cfg_weights[1])]

    y_train = torch.cat([sample.label_vals for sample in train_samples], dim=0).long()
    n0 = int((y_train == 0).sum().item())
    n1 = int((y_train == 1).sum().item())

    if n0 == 0 or n1 == 0:
        raise ValueError(f"Cannot compute automatic class weights because class counts are n0={n0}, n1={n1}.")

    total = n0 + n1
    w0 = total / (2.0 * n0)
    w1 = total / (2.0 * n1)

    print(f"Train labeled counts: licit(0)={n0}, illicit(1)={n1}")
    return [float(w0), float(w1)]

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    # Exact EvolveGCN here relies on sparse COO matmul; keep CPU fallback instead of MPS.
    print("Using CPU")
    return torch.device("cpu")


def move_sparse_tensor(sp: torch.Tensor, device: torch.device) -> torch.Tensor:
    sp = sp.coalesce()
    return torch.sparse_coo_tensor(
        sp.indices().to(device),
        sp.values().to(device),
        size=sp.size(),
        dtype=sp.dtype,
        device=device,
    ).coalesce()


def move_sample(sample, device: torch.device):
    hist_adj_list = [move_sparse_tensor(adj, device) for adj in sample.hist_adj_list]
    hist_ndFeats_list = [x.to(device) for x in sample.hist_ndFeats_list]
    node_mask_list = [m.to(device) for m in sample.node_mask_list]
    label_idx = sample.label_idx.to(device)
    label_vals = sample.label_vals.to(device)
    return hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx, label_vals


@torch.no_grad()
def evaluate(model, samples, loss_fn, device: torch.device):
    model.eval()
    losses = []
    all_true = []
    all_pred = []
    by_time = {}

    for sample in samples:
        hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx, label_vals = move_sample(sample, device)
        logits = model(hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx)
        loss = loss_fn(logits, label_vals)
        losses.append(float(loss.item()))

        y_true = label_vals.detach().cpu()
        y_pred = logits.argmax(dim=1).detach().cpu()
        all_true.append(y_true)
        all_pred.append(y_pred)

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        by_time[int(sample.current_time)] = {
            "precision_pos": float(p[1]),
            "recall_pos": float(r[1]),
            "f1_pos": float(f1[1]),
        }

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    pM, rM, f1M, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "n_labeled": int(len(y_true)),
        "acc": float(acc),
        "precision_pos": float(p[1]),
        "recall_pos": float(r[1]),
        "f1_pos": float(f1[1]),
        "precision_macro": float(pM),
        "recall_macro": float(rM),
        "f1_macro": float(f1M),
        "fraud_predictions": int((y_pred == 1).sum()),
        "confusion_matrix_2x2": cm.tolist(),
        "f1_pos_by_timestep": by_time,
    }


def print_epoch_metrics(epoch: int, train_metrics: dict, test_metrics: dict):
    print(
        f"Epoch {epoch:4d} | "
        f"TRAIN: Loss={train_metrics['loss']:.6f} P={train_metrics['precision_pos']:.4f} "
        f"R={train_metrics['recall_pos']:.4f} F1={train_metrics['f1_pos']:.4f} "
        f"(FraudPred={train_metrics['fraud_predictions']}) | "
        f"TEST: Loss={test_metrics['loss']:.6f} P={test_metrics['precision_pos']:.4f} "
        f"R={test_metrics['recall_pos']:.4f} F1={test_metrics['f1_pos']:.4f} "
        f"(FraudPred={test_metrics['fraud_predictions']})"
    )


def print_final_metrics(title: str, metrics: dict):
    cm = metrics["confusion_matrix_2x2"]
    print(f"\n=== {title} Final Test Evaluation ===")
    print(f"#labeled={metrics['n_labeled']} | FraudPred={metrics['fraud_predictions']}")
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


def save_run(model, metrics: dict, config: dict, prefix: str):
    save_dir = config["save"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    if config["save"].get("save_run", True):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(save_dir, f"{prefix}_{run_id}")
        os.makedirs(out_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({"test": metrics}, f, indent=2)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Saved run to {out_dir}")


def build_model(num_features: int, config: dict):
    variant = config["model"]["variant"].lower()
    activation = torch.nn.RReLU()

    if variant == "h":
        backbone = EvolveGCNH(
            in_dim=num_features,
            layer_1_feats=int(config["model"]["layer_1_feats"]),
            layer_2_feats=int(config["model"]["layer_2_feats"]),
            activation=activation,
            skipfeats=bool(config["model"].get("skipfeats", False)),
        )
    elif variant == "o":
        backbone = EvolveGCNO(
            in_dim=num_features,
            layer_1_feats=int(config["model"]["layer_1_feats"]),
            layer_2_feats=int(config["model"]["layer_2_feats"]),
            activation=activation,
            skipfeats=bool(config["model"].get("skipfeats", False)),
        )
    else:
        raise ValueError("CONFIG['model']['variant'] must be either 'h' or 'o'.")

    classifier_in = int(config["model"]["layer_2_feats"])
    if bool(config["model"].get("skipfeats", False)):
        classifier_in += num_features

    classifier = EvolveGCNClassifier(
        in_dim=classifier_in,
        hidden_dim=int(config["model"]["cls_feats"]),
        out_dim=2,
    )
    return EvolveGCNNodeClassifier(backbone=backbone, classifier=classifier)


def main():
    device = get_device()

    data_cfg = EvolveGCNEllipticConfig(**CONFIG["data"])
    dataset = EvolveGCNEllipticDataset(data_cfg)
    sequence = dataset.get_sequence()

    model = build_model(sequence.num_features, CONFIG).to(device)
    class_weights = resolve_class_weights(sequence.train_samples, CONFIG, device)
    loss_fn = WeightedCrossEntropy(class_weights, device)

    gcn_opt = torch.optim.Adam(model.backbone.parameters(), lr=CONFIG["training"]["lr"])
    cls_opt = torch.optim.Adam(model.classifier.parameters(), lr=CONFIG["training"]["lr"])

    print(f"Class weights: {class_weights}")
    print(
        f"\n=== Training exact EvolveGCN-{CONFIG['model']['variant'].upper()} ===\n"
        f"Train windows: {len(sequence.train_samples)} | Test windows: {len(sequence.test_samples)} | "
        f"num_nodes={sequence.num_nodes} | input_dim={sequence.num_features}"
    )

    epochs = int(CONFIG["training"]["epochs"])
    log_every = int(CONFIG["training"].get("log_every", 20))

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for sample in sequence.train_samples:
            hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx, label_vals = move_sample(sample, device)

            gcn_opt.zero_grad(set_to_none=True)
            cls_opt.zero_grad(set_to_none=True)

            logits = model(hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx)
            loss = loss_fn(logits, label_vals)
            loss.backward()

            gcn_opt.step()
            cls_opt.step()
            train_losses.append(float(loss.item()))

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            train_metrics = evaluate(model, sequence.train_samples, loss_fn, device)
            if train_losses:
                train_metrics["loss"] = float(np.mean(train_losses))
            test_metrics = evaluate(model, sequence.test_samples, loss_fn, device)
            print_epoch_metrics(epoch, train_metrics, test_metrics)

    test_metrics = evaluate(model, sequence.test_samples, loss_fn, device)
    print_final_metrics(f"EvolveGCN-{CONFIG['model']['variant'].upper()}", test_metrics)
    save_run(model, test_metrics, CONFIG, f"evolvegcn_{CONFIG['model']['variant'].lower()}")


if __name__ == "__main__":
    main()
