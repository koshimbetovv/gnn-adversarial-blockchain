import yaml
import torch
from src.datasets.elliptic import EllipticDataset
from src.models.gcn import GCN
from src.training.trainer import Trainer
from src.training.evaluator import evaluate
from src.attacks.nettack_local import NettackLocalAttack

def main(cfg):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dataset = EllipticDataset()
    data = dataset.get_data()

    model = GCN(
        data.num_features,
        cfg["model"]["hidden_dim"],
        2
    )

    trainer = Trainer(model, data, device)
    trainer.train(cfg["training"]["epochs"], cfg["training"]["lr"])

    acc_clean = evaluate(model, data)

    attack = NettackLocalAttack(model, data, device)
    edge_index_adv = attack.attack(target_node=0, edge_index=data.edge_index)

    data.edge_index = edge_index_adv
    acc_adv = evaluate(model, data)

    print("Clean acc:", acc_clean)
    print("Attack acc:", acc_adv)

if __name__ == "__main__":
    with open("config/experiments/exp_elliptic_nettack_gcn.yaml") as f:
        cfg = yaml.safe_load(f)
    main(cfg)


# python src/main.py --config config/experiments/exp_attack_sweep.yaml
# experiments/{dataset}/{model}/{attack}/
