import os
import sys
import json
import argparse
import random
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.elliptic import EllipticDataset
from src.attacks.gnia import GNIA
from src.utils.model_loader import load_model, resolve_checkpoint
from src.training.metrics import (
    get_split_mask,
    evaluate_logits_on_split,
    attack_success_rate,
    asr_pos_neg,
    roc_auc_binary,
    mean_confidence_drop,
)

# ---------- Config Globals ----------
MODEL_NAME = "gcn"
DATASET = "elliptic"
LR = 0.01
EPOCHS = 50
BUDGET = 1
TARGET_IDS = "all"
MODEL_DIR = None
SPLIT = "test"
SEED = 42
ATTACK_ONLY_ILLICIT = True
ONLY_CLEAN_CORRECT = True
ATTACK_FRACTION = 1.0

# G-NIA specific params
GNIA_ATTR_TAU = 1.0
GNIA_EDGE_TAU = 1.0
GNIA_FEAT_NUM = None # None means continuous optimization for attributes or kept clean

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def make_run_dir(model_name: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(repo_root, "attacks", f"{model_name}_gnia_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, ts

def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def get_model_weights(model, model_type):
    # Same helper logic as before for GCN/GAT/SAGE extraction
    w1, w2 = None, None
    if model_type.lower() == 'gcn':
        conv1 = model.convs[0]
        conv2 = model.convs[-1]
        w1 = getattr(conv1, 'lin', None)
        if w1 is None and hasattr(conv1, 'weight'): w1 = conv1.weight 
        if w1 is not None and hasattr(w1, 'weight'): w1 = w1.weight.detach()
        elif w1 is not None: w1 = w1.detach()
        
        w2 = getattr(conv2, 'lin', None)
        if w2 is None and hasattr(conv2, 'weight'): w2 = conv2.weight
        if w2 is not None and hasattr(w2, 'weight'): w2 = w2.weight.detach()
        elif w2 is not None: w2 = w2.detach()
        
    elif model_type.lower() == 'gat':
        conv1 = model.convs[0]
        conv2 = model.convs[-1]
        w1 = getattr(conv1, 'lin_src', None)
        if w1 is None: w1 = getattr(conv1, 'lin', None)
        if w1 is not None and hasattr(w1, 'weight'): w1 = w1.weight.detach()
        w2 = getattr(conv2, 'lin_src', None)
        if w2 is None: w2 = getattr(conv2, 'lin', None)
        if w2 is not None and hasattr(w2, 'weight'): w2 = w2.weight.detach()
            
    elif model_type.lower() == 'graphsage':
        conv1 = model.conv1
        conv2 = model.conv2
        w1_linear = getattr(conv1, 'lin_l', None)
        if w1_linear is None: w1_linear = getattr(conv1, 'lin', None)
        if w1_linear is not None: w1 = w1_linear.weight.detach()
        w2_linear = getattr(conv2, 'lin_l', None)
        if w2_linear is None: w2_linear = getattr(conv2, 'lin', None)
        if w2_linear is not None: w2 = w2_linear.weight.detach()

    in_channels = None
    if hasattr(model, 'convs'): in_channels = model.convs[0].in_channels
    elif hasattr(model, 'conv1'): in_channels = model.conv1.in_channels
        
    if w1 is not None and in_channels is not None and w1.shape[0] != in_channels and w1.ndim==2:
         w1 = w1.t()
    
    if w2 is not None and w1 is not None and w1.ndim==2 and w2.ndim==2:
         if w1.shape[1] != w2.shape[0] and w1.shape[1] == w2.shape[1]:
              w2 = w2.t()

    return w1, w2

def train_gnia_multi(args, gnia_model, victim_model, data, target_indices, device):
    optimizer = torch.optim.RMSprop(gnia_model.parameters(), lr=args.lr)
    gnia_model.train()
    
    feat = data.x.to(device)
    edge_index = data.edge_index.to(device)
    n = feat.shape[0]
    
    adj_val = torch.ones(edge_index.shape[1], device=device)
    adj_sparse = torch.sparse_coo_tensor(edge_index, adj_val, (n, n), device=device)
    labels = data.y.to(device)
    
    victim_model.eval()
    with torch.no_grad():
         pred_logits = victim_model(feat, edge_index)
         node_emb = pred_logits.detach()

    if isinstance(target_indices, torch.Tensor):
         target_tensor = target_indices.to(device)
    else:
         target_tensor = torch.tensor(target_indices, device=device)
    mask = torch.isin(edge_index[0], target_tensor)
    target_neighbors = edge_index[1, mask].unique()
    sub_graph_nodes = target_neighbors
    if len(sub_graph_nodes) == 0: sub_graph_nodes = target_tensor
         
    w1, w2 = get_model_weights(victim_model, args.model)
    w1, w2 = w1.to(device), w2.to(device)
    W = torch.mm(w1, w2).t() 
    
    from tqdm import tqdm
    pbar = tqdm(range(args.epochs), desc="GNIA Optimization", leave=False, dynamic_ncols=True)
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Prepare Wlabel/Wsec
        # Optimized: Batch preparation
        unique_targets = target_tensor
        # wlabel: Weight vector for True Label
        # wsec: Weight vector for Best Wrong Label
        # W shape: [OutDim, HiddenDim] (usually) or [OutDim, OutDim] ?? 
        # Actually GNIA expects W1*W2 result.
        # W1: [In, Hid], W2: [Hid, Out]. W1*W2: [In, Out].
        # Using extracted weights: 
        # Code above: W = torch.mm(w1, w2).t() -> [Out, In]?
        # We need weights corresponding to CLASSES. 
        # GNIA paper eq involves W * x.
        # We actually passed `W` to GNIA constructor as weight1, weight2...
        # Here we need `wlabel` and `wsec`.
        # These are usually rows of the final classification layer if it's linear.
        # But for GCN, it's ... complex. 
        # Wait, the GNIA implementation used `W[ori_label]` where W was `mm(w1, w2).t()`.
        # So W is [Out, In].
        
        ori_labels = labels[target_tensor]
        logits_batch = pred_logits[target_tensor]
        
        # Best wrong label
        mask_logits = logits_batch.clone()
        mask_logits.scatter_(1, ori_labels.unsqueeze(1), -1e9)
        best_wrong_labels = mask_logits.argmax(dim=1)
        
        wlabel = W[ori_labels]
        wsec = W[best_wrong_labels]
        
        add_feat, disc_score, masked_score_idx, new_feat = gnia_model(
            target_tensor, sub_graph_nodes, args.budget, feat, adj_sparse, 
            node_emb, wlabel, wsec, train_flag=True, eps=1.0 
        )
        
        # Construct graph
        # ... (same construction logic)
        num_existing = edge_index.shape[1]
        exist_w = torch.ones(num_existing, device=device)
        inj_idx = n
        
        new_edge_src = torch.full((len(sub_graph_nodes),), inj_idx, device=device)
        new_edge_dst = sub_graph_nodes
        
        ne1 = torch.stack([new_edge_src, new_edge_dst], dim=0)
        nw1 = disc_score.reshape(-1)
        ne2 = torch.stack([new_edge_dst, new_edge_src], dim=0)
        nw2 = disc_score.reshape(-1)
        
        full_ei = torch.cat([edge_index, ne1, ne2], dim=1)
        full_ew = torch.cat([exist_w, nw1, nw2])
        
        new_logits = victim_model(new_feat, full_ei, edge_weight=full_ew)
        
        # Loss
        t_logits = new_logits[target_tensor]
        idx = torch.arange(len(target_tensor), device=device)
        target_scores = t_logits[idx, ori_labels]
        
        others = t_logits.clone()
        others[idx, ori_labels] = -1e9
        max_others = others.max(dim=1).values
        
        loss = F.relu(target_scores - max_others).mean()
        
        loss.backward()
        optimizer.step()
            
    # Inference
    with torch.no_grad():
        gnia_model.eval()
        add_feat, disc_score, masked_score_idx, new_feat = gnia_model(
            target_tensor, sub_graph_nodes, args.budget, feat, adj_sparse, 
            node_emb, wlabel, wsec, train_flag=False, eps=1.0
        )
        # ... construct discrete edges ...
        ne1 = torch.stack([torch.full((len(sub_graph_nodes),), inj_idx, device=device), sub_graph_nodes], dim=0)
        nw1 = disc_score.reshape(-1)
        ne2 = torch.stack([sub_graph_nodes, torch.full((len(sub_graph_nodes),), inj_idx, device=device)], dim=0)
        nw2 = disc_score.reshape(-1)
        
        mask1 = nw1 > 0
        mask2 = nw2 > 0
        final_new = torch.cat([ne1[:, mask1], ne2[:, mask2]], dim=1)
        full_ei = torch.cat([edge_index, final_new], dim=1)
        
        # Store injected node info for metrics
        injected_node_features = new_feat[-1].unsqueeze(0)
        
        return new_feat, full_ei, injected_node_features, final_new

def main():
    global MODEL_NAME, DATASET, LR, EPOCHS, BUDGET, TARGET_IDS, MODEL_DIR
    global SPLIT, SEED, ATTACK_ONLY_ILLICIT, ONLY_CLEAN_CORRECT, ATTACK_FRACTION

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default=DATASET)
        parser.add_argument('--model', type=str, default=MODEL_NAME)
        parser.add_argument('--model_dir', type=str, default=MODEL_DIR)
        parser.add_argument('--lr', type=float, default=LR)
        parser.add_argument('--epochs', type=int, default=EPOCHS)
        parser.add_argument('--budget', type=int, default=BUDGET)
        parser.add_argument('--target_ids', type=str, default=TARGET_IDS)
        parser.add_argument('--seed', type=int, default=SEED)
        parser.add_argument('--split', type=str, default=SPLIT)
        parser.add_argument('--attack_fraction', type=float, default=ATTACK_FRACTION)
        args = parser.parse_args()
        
        MODEL_NAME = args.model
        DATASET = args.dataset
        MODEL_DIR = args.model_dir
        LR = args.lr
        EPOCHS = args.epochs
        BUDGET = args.budget
        TARGET_IDS = args.target_ids
        SEED = args.seed
        SPLIT = args.split
        ATTACK_FRACTION = args.attack_fraction

    setup_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ds = EllipticDataset()
    data = ds.data.to(device) 
    
    # Load Model using unified loader
    # If MODEL_DIR is None, loader finds latest in 'models/'
    # If MODEL_DIR is set, we might need to parse run_id or just use directory
    
    # Adapt loader args
    run_id = None
    model_base_dir = "models"
    
    if MODEL_DIR:
        # If user passed full path "models/gcn_...", split it
        # loader expects model_dir="models" and run_id="YYYY...".
        # Or we can just manually load if loader is too strict.
        # Let's try to infer.
        if os.path.isdir(MODEL_DIR):
             model_base_dir = os.path.dirname(MODEL_DIR)
             dirname = os.path.basename(MODEL_DIR)
             # dirname like "gcn_2024..."
             parts = dirname.split('_')
             if len(parts) >= 2:
                 # Attempt to extract ID? 
                 # resolve_checkpoint takes run_id.
                 # Let's just point the loader to the specific file if possible?
                 # No, loader constructs path.
                 # Let's use the explicit load logic if MODEL_DIR is exact path, 
                 # else rely on loader defaults.
                 
                 # Hack: Check if MODEL_DIR contains model.pt
                 if os.path.exists(os.path.join(MODEL_DIR, 'model.pt')):
                      # User provided specific run dir.
                      # Helper:
                      pass
                      
    # To keep it simple and robust with Sweep:
    # Sweep passes `MODEL_NAME`. Loader finds latest.
    # If we want specific, we rely on loader's run_id. 
    # But for now, let's use `load_model` which finds latest by default,
    # OR if MODEL_DIR is set, we manually load like before (keeping robustness).
    
    if MODEL_DIR:
         # Generic manual load (copied from previous robust version)
         # Re-implementing simplified version:
         import json
         config_path = os.path.join(MODEL_DIR, 'config.json')
         pt_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pt")
         if not os.path.exists(pt_path): pt_path = os.path.join(MODEL_DIR, "model.pt")
         
         with open(config_path) as f: cfg = json.load(f)
         from src.utils.model_loader import build_model
         model = build_model(MODEL_NAME, data.num_features, 2, cfg).to(device)
         model.load_state_dict(torch.load(pt_path, map_location=device))
         print(f"Loaded {MODEL_NAME} from {MODEL_DIR}")
    else:
         model = load_model(MODEL_NAME, data.num_features, 2, device=device)

    # Metrics on Clean
    split_mask = get_split_mask(data, SPLIT).to(device)
    
    with torch.no_grad():
        logits_clean = model(data.x, data.edge_index)
        
    clean_m = evaluate_logits_on_split(logits_clean, data.y, split_mask, SPLIT)
    print(f"Clean F1 (Macro): {clean_m.f1_macro:.4f}")

    # Standard Target Selection
    from scripts.run_node_injection_attack import pick_target_nodes
    targets = pick_target_nodes(
        data, logits_clean, split_mask,
        only_illicit=ATTACK_ONLY_ILLICIT,
        fraction=ATTACK_FRACTION,
        only_clean_correct=ONLY_CLEAN_CORRECT,
        seed=SEED,
        device=device
    )
    
    if targets.numel() == 0:
        print("No targets found.")
        return

    # Attack Mask
    attack_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    attack_mask[targets] = True
    
    # Run GNIA
    print(f"--- Attacking {len(targets)} targets with GNIA ---")
    w1, w2 = get_model_weights(model, MODEL_NAME)
    if w1 is None or w2 is None:
         print("Failed to extract weights for GNIA.")
         return

    gnia = GNIA(
        labels=data.y, feat_dim=data.num_features, 
        weight1=w1, weight2=w2, discrete=False, device=device,
        tar_num=len(targets), feat_max=data.x.max(0).values, feat_min=data.x.min(0).values,
        attr_tau=GNIA_ATTR_TAU, edge_tau=GNIA_EDGE_TAU
    ).to(device)
    
    new_feat, new_edge_index, inj_feat, inj_edges = train_gnia_multi(
        type('Args', (object,), {'lr': LR, 'epochs': EPOCHS, 'budget': BUDGET, 'model': MODEL_NAME}),
        gnia, model, data, targets, device
    )
    
    # Adv Evaluation
    with torch.no_grad():
        logits_adv_full = model(new_feat, new_edge_index)
        logits_adv = logits_adv_full[:data.num_nodes]
        
    adv_m = evaluate_logits_on_split(logits_adv, data.y, split_mask, SPLIT)
    
    # ASR metrics
    asr_attacked, ns_attacked, na_attacked = attack_success_rate(
        data.y, logits_clean.argmax(1), logits_adv.argmax(1), attack_mask
    )
    asr_split, ns_split, na_split = attack_success_rate(
        data.y, logits_clean.argmax(1), logits_adv.argmax(1), split_mask
    )
    
    roc_clean = roc_auc_binary(logits_clean, data.y, split_mask)
    roc_adv = roc_auc_binary(logits_adv, data.y, split_mask)

    conf_drop, n_conf_nodes = mean_confidence_drop(
        data.y, logits_clean, logits_adv, attack_mask, only_clean_correct=True
    )

    f1_drop = clean_m.f1_macro - adv_m.f1_macro
    auc_drop = roc_clean - roc_adv

    print(f"F1 drop: {f1_drop:.4f}")
    print(f"ROC-AUC drop: {auc_drop:.4f}")
    print(f"Mean conf drop: {conf_drop:.4f}")
    
    # Save Results
    run_dir, ts = make_run_dir(MODEL_NAME)
    
    # Config
    cfg = {
        "timestamp": ts,
        "attack": "GNIA",
        "model_name": MODEL_NAME,
        "split": SPLIT,
        "params": {
            "lr": LR, "epochs": EPOCHS, "budget": BUDGET,
            "gnia_attr_tau": GNIA_ATTR_TAU, "gnia_edge_tau": GNIA_EDGE_TAU
        },
        "metrics": {
             "clean_f1": clean_m.f1_macro,
             "adv_f1": adv_m.f1_macro,
             "asr_attacked": asr_attacked,
             "asr_split": asr_split
        }
    }
    write_json(os.path.join(run_dir, "config.json"), cfg)
    
    # Full Metrics (match sweep format + drops for Table 4)
    metrics = {
        "attack": "GNIA",
        "model_name": MODEL_NAME,
        "split": SPLIT,
        "n_targets": len(targets),
        "f1": {
            "macro_clean": clean_m.f1_macro,
            "macro_adv": adv_m.f1_macro,
            "drop": f1_drop
        },
        "asr": {
            "attacked": {"value": asr_attacked, "success": ns_attacked, "attempted": na_attacked},
            "split": {"value": asr_split}
        },
        "roc_auc": {
            "clean": roc_clean, 
            "adv": roc_adv,
            "drop": auc_drop
        },
        "mean_confidence_drop": {
            "value": conf_drop,
            "n": n_conf_nodes
        },
        "clean_split_metrics": vars(clean_m),
        "adv_split_metrics": vars(adv_m)
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    print(f"Saved results to {run_dir}")

if __name__ == '__main__':
    main()
