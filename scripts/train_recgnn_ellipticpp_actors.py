import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.optim import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_paper_utils import get_device
from src.datasets.recgnn_ellipticpp_actors import (
    RecGNNEllipticPPActorsConfig,
    RecGNNEllipticPPActorsDataset,
)
from src.models.recgnn import RecGNN


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
    },
    "model": {
        "hidden_dim": 50,
        "dropout": 0.5,
        "out_dim": 2,
        # state_rows is filled automatically from the largest timestep graph.
    },
    "training": {
        "epochs": 100,
        "lr": 1.5e-3,
        "weight_decay": 0.0,
        "log_every": 10,
    },
    "save": {
        "save_dir": "models",
        "save_run": True,
        "prefix": "recgnn_ellipticpp_actors",
    },
}


@torch.no_grad()
def _forward_sequence(model, graphs, device, prime_graphs=None):
    model.eval()
    model.reset_sequence_state(device)

    if prime_graphs is not None:
        for graph in prime_graphs:
            graph = graph.to(device)
            _ = model(graph.x.float(), graph.edge_index.long())
            model.detach_sequence_state()

    preds_all = []
    labels_all = []
    timestep_f1 = {}

    for graph in graphs:
        graph = graph.to(device)
        log_probs = model(graph.x.float(), graph.edge_index.long())
        model.detach_sequence_state()

        mask = (graph.y != -1)
        if int(mask.sum().item()) == 0:
            continue

        y_true = graph.y[mask].cpu()
        y_pred = log_probs[mask].argmax(dim=1).cpu()
        preds_all.append(y_pred)
        labels_all.append(y_true)

        p, r, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
        )
        timestep_f1[int(graph.graph_timestep)] = float(f1[1])

    if not labels_all:
        raise ValueError("No labeled nodes found while evaluating RecGNN sequence.")

    y_true = torch.cat(labels_all).numpy()
    y_pred = torch.cat(preds_all).numpy()

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "n_labeled": int(len(y_true)),
        "acc": float(acc),
        "precision_pos": float(p[1]),
        "recall_pos": float(r[1]),
        "f1_pos": float(f1[1]),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "fraud_predictions": int((y_pred == 1).sum()),
        "confusion_matrix_2x2": cm.tolist(),
        "f1_pos_by_timestep": timestep_f1,
    }


def print_epoch_metrics(epoch: int, loss: float, train_metrics: dict, test_metrics: dict):
    print(
        f"Epoch {epoch:4d} | AvgBatchLoss={loss:.6f} | "
        f"TRAIN: P={train_metrics['precision_pos']:.4f} R={train_metrics['recall_pos']:.4f} "
        f"F1={train_metrics['f1_pos']:.4f} (FraudPred={train_metrics['fraud_predictions']}) | "
        f"TEST: P={test_metrics['precision_pos']:.4f} R={test_metrics['recall_pos']:.4f} "
        f"F1={test_metrics['f1_pos']:.4f} (FraudPred={test_metrics['fraud_predictions']})"
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


def save_run(model, metrics: dict, config: dict):
    save_dir = config["save"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    if config["save"]["save_run"]:
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

    dataset = RecGNNEllipticPPActorsDataset(
        RecGNNEllipticPPActorsConfig(**CONFIG["data"])
    )
    sequence = dataset.get_sequence()
    CONFIG["model"]["state_rows"] = int(sequence.max_nodes)

    model = RecGNN(
        in_dim=sequence.num_features,
        hidden_dim=CONFIG["model"]["hidden_dim"],
        out_dim=CONFIG["model"]["out_dim"],
        state_rows=CONFIG["model"]["state_rows"],
        dropout=CONFIG["model"]["dropout"],
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"].get("weight_decay", 0.0),
    )

    print("Class weights: None")
    print(
        f"\n=== Training RecGNN on Elliptic++ Actors (sequence mode) ===\n"
        f"Train graphs: {len(sequence.train_graphs)} | Test graphs: {len(sequence.test_graphs)} | "
        f"state_rows={sequence.max_nodes} | input_dim={sequence.num_features}"
    )

    log_every = CONFIG["training"].get("log_every", 10)
    for epoch in range(1, CONFIG["training"]["epochs"] + 1):
        model.train()
        model.reset_sequence_state(device)

        batch_losses = []
        for graph in sequence.train_graphs:
            graph = graph.to(device)
            mask = (graph.y != -1)
            if int(mask.sum().item()) == 0:
                _ = model(graph.x.float(), graph.edge_index.long())
                model.detach_sequence_state()
                continue

            optimizer.zero_grad(set_to_none=True)
            log_probs = model(graph.x.float(), graph.edge_index.long())
            loss = F.nll_loss(log_probs[mask], graph.y[mask])
            loss.backward()
            optimizer.step()
            model.detach_sequence_state()
            batch_losses.append(float(loss.item()))

        avg_loss = float(np.mean(batch_losses)) if batch_losses else 0.0

        if epoch == 1 or epoch % log_every == 0 or epoch == CONFIG["training"]["epochs"]:
            train_metrics = _forward_sequence(model, sequence.train_graphs, device)
            test_metrics = _forward_sequence(model, sequence.test_graphs, device, prime_graphs=sequence.train_graphs)
            print_epoch_metrics(epoch, avg_loss, train_metrics, test_metrics)

    test_metrics = _forward_sequence(model, sequence.test_graphs, device, prime_graphs=sequence.train_graphs)
    print_final_metrics("RecGNN (Elliptic++ Actors)", test_metrics)
    save_run(model, test_metrics, CONFIG)


if __name__ == "__main__":
    main()
