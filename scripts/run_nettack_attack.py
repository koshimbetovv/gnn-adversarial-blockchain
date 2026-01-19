import os
import sys
from collections import defaultdict
import torch
import torch.nn.functional as F


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.elliptic import EllipticDataset
from src.attacks.nettack_local import NettackLocalAttack
from src.utils.model_loader import load_model

# ---------- attack parameters ----------
MODEL_NAME = "gcn"
SPLIT = "test"
#N_TARGETS = 3
N_PERTURBATIONS = 5
SAMPLE_SIZE = 50
UNDIRECTED = False
ALLOW_REMOVALS = False
SEED = 0

# ---------- target selection controls ----------
ATTACK_ONLY_ILLICIT = True
ATTACK_FRACTION = 0.1  # Between 0 and 1.0

# Recommended for meaningful ASR (correct->wrong). You can still set False to sample broader targets.
ONLY_CLEAN_CORRECT = True
MAX_TARGETS = None

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def build_adj_list(edge_index, undirected):
    adj = defaultdict(set)
    for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj[u].add(v)
        if undirected:
            adj[v].add(u)
    return adj

def main():
    device = get_device()
    data = EllipticDataset().get_data().to(device)
    model = load_model(MODEL_NAME, data.num_features, 2, device=device)

    split_mask = getattr(data, f"{SPLIT}_mask", (data.y != -1)).bool() & (data.y != -1)
    candidates = torch.where(split_mask)[0]

    with torch.no_grad():
        logits_clean = model(data.x, data.edge_index)
    y_pred_clean = logits_clean.argmax(1)
    
    if ATTACK_ONLY_ILLICIT:
        candidates = candidates[data.y[candidates] == 1]
    if ONLY_CLEAN_CORRECT and candidates.numel() > 0:
        candidates = candidates[y_pred_clean[candidates] == data.y[candidates]]
    if candidates.numel() == 0:
        print("No eligible target nodes found for the chosen settings.")
        return

    n_targets = int(round(float(ATTACK_FRACTION) * float(candidates.numel())))
    n_targets = max(1, min(n_targets, int(candidates.numel())))
    if MAX_TARGETS is not None:
        n_targets = min(int(MAX_TARGETS), n_targets)

    g = torch.Generator(device="cpu"); g.manual_seed(int(SEED))
    cand_cpu = candidates.detach().cpu()
    perm = torch.randperm(cand_cpu.numel(), generator=g)
    targets = cand_cpu[perm[:n_targets]].to(device)

    print(f"frac={ATTACK_FRACTION} | n_targets={int(targets.numel())}")

    edge_index_base = data.edge_index.cpu()

    # IMPORTANT: don't reuse a shared mutable adj_list across targets.
    atk = NettackLocalAttack(model, data, device, adj_list=None,
                              undirected=UNDIRECTED, allow_removals=ALLOW_REMOVALS, seed=SEED)

    success = 0
    attempted = 0  # only clean-correct targets count as 'attempted' for ASR
    conf_drop_sum = 0.0
    conf_n = 0
    for t in targets.tolist():
        is_clean_correct = bool(int(y_pred_clean[t].item()) == int(data.y[t].item()))
        edge_adv = atk.attack(t, edge_index_base, n_perturbations=N_PERTURBATIONS, sample_size=SAMPLE_SIZE)

        with torch.no_grad():
            pred_adv = int(model(data.x, edge_adv)[t].argmax().item())

        if is_clean_correct:
            attempted += 1
            if pred_adv != int(data.y[t].item()):
                success += 1

        with torch.no_grad():
            true_y = int(data.y[t].item())
            p_clean_true = F.softmax(logits_clean[t], dim=0)[true_y].item()

            logits_t_adv = model(data.x, edge_adv)[t]
            p_adv_true = F.softmax(logits_t_adv, dim=0)[true_y].item()

        if is_clean_correct:
            conf_drop_sum += (p_clean_true - p_adv_true)
            conf_n += 1


    asr = (success / attempted) if attempted > 0 else 0.0
    mean_drop = (conf_drop_sum / conf_n) if conf_n > 0 else 0.0
    print(
        f"Nettack-Local | k={N_PERTURBATIONS} | only_illicit={ATTACK_ONLY_ILLICIT} | "
        f"frac={ATTACK_FRACTION} | n_targets={int(targets.numel())} | attempted(clean-correct)={attempted}"
    )
    print(f"ASR={asr:.6f} ({success}/{attempted})")
    print(f"Mean confidence drop (clean-correct targets): {mean_drop:.6f} over n={conf_n}")


if __name__ == "__main__":
    main()
