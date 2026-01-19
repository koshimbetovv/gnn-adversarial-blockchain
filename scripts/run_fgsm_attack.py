import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.elliptic import EllipticDataset
from src.attacks.fgsm import FGSMAttack
from src.utils.model_loader import load_model
from src.training.metrics import (
    get_split_mask, evaluate_logits_on_split, attack_success_rate,
    roc_auc_binary, mean_confidence_drop, asr_pos_neg
)
from src.utils.attack_targets import pick_target_nodes

# ---------- attack parameters ----------
MODEL_NAME = "gcn"
SPLIT = "test"
EPS = 0.05
TARGETED = False
TARGET_LABEL = 0
CLAMP = None  # e.g. (-3.0, 3.0)

# ---------- target selection controls ----------
# If True, attack only illicit nodes (y==1) in the chosen split.
ATTACK_ONLY_ILLICIT = True
# What fraction of eligible nodes in the split to attack. 
ATTACK_FRACTION = 1.0
# If True, restrict targets to nodes the model classifies correctly on clean inputs.
ONLY_CLEAN_CORRECT = True
SEED = 0

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def main():
    device = get_device()
    data = EllipticDataset().get_data().to(device)
    model = load_model(MODEL_NAME, data.num_features, 2, device=device)

    split_mask = get_split_mask(data, SPLIT).to(device)
    with torch.no_grad():
        logits_clean = model(data.x, data.edge_index)

    targets = pick_target_nodes(
        data, logits_clean, split_mask,
        only_illicit=ATTACK_ONLY_ILLICIT,
        fraction=ATTACK_FRACTION,
        only_clean_correct=ONLY_CLEAN_CORRECT,
        seed=SEED,
        device=device,
    )
    if targets.numel() == 0:
        print("No eligible target nodes found for the chosen settings.")
        return

    atk = FGSMAttack(model, data, device, clamp=CLAMP)
    x_adv = atk.attack(targets, eps=EPS, targeted=TARGETED, target_label=TARGET_LABEL)

    with torch.no_grad():
        logits_adv = model(x_adv, data.edge_index)

    # Optional: evaluate only on attacked nodes (recommended, otherwise it gets diluted)
    attack_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    attack_mask[targets] = True
    attack_mask = attack_mask & split_mask  # keep inside split

    roc_clean = roc_auc_binary(logits_clean, data.y, split_mask)
    roc_adv = roc_auc_binary(logits_adv, data.y, split_mask)

    conf_drop, n_used = mean_confidence_drop(
        data.y, logits_clean, logits_adv, attack_mask, only_clean_correct=True
    )

    clean_m = evaluate_logits_on_split(logits_clean, data.y, split_mask, SPLIT)
    adv_m = evaluate_logits_on_split(logits_adv, data.y, split_mask, SPLIT)

    # ASR over attacked subset (recommended when ATTACK_FRACTION < 1)
    asr, ns, na = attack_success_rate(data.y, logits_clean.argmax(1), logits_adv.argmax(1), attack_mask)

    asr_p, sp, ap, asr_n, sn, an = asr_pos_neg(data.y, logits_clean, logits_adv, attack_mask)

    print(f"FGSM | eps={EPS} | TARGETED={TARGETED} | TARGET_LABEL={TARGET_LABEL}")

    print(f"ASR={asr:.6f} ({ns}/{na})")
    print(f"ASR_pos (illicit flips) = {asr_p:.6f} ({sp}/{ap}), ASR_neg (licit flips) = {asr_n:.6f} ({sn}/{an})")
    print(f"F1_pos : {clean_m.f1_pos:.4f} -> {adv_m.f1_pos:.4f}  (drop {clean_m.f1_pos-adv_m.f1_pos:.4f})")
    print(f"F1_macro: {clean_m.f1_macro:.4f} -> {adv_m.f1_macro:.4f} (drop {clean_m.f1_macro-adv_m.f1_macro:.4f})")

    print(f"ROC-AUC (split): {roc_clean:.6f} -> {roc_adv:.6f}")
    print(f"Mean confidence drop (attacked, clean-correct): {conf_drop:.6f} over n={n_used}")


if __name__ == "__main__":
    main()
