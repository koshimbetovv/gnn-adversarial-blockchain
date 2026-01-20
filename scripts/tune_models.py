import os
import sys
import argparse
import subprocess
import json
import optuna
import shutil
from datetime import datetime

# Define search spaces
def suggest_params(trial, model_name):
    # Common params
    args = []
    
    # Learning rate: log-uniform [1e-4, 5e-2]
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    args.extend(["--lr", str(lr)])
    
    # Weight decay: log-uniform [1e-6, 1e-2]
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    args.extend(["--weight_decay", str(weight_decay)])
    
    # Hidden dim: categorical [32, 64, 128, 256]
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    args.extend(["--hidden_dim", str(hidden_dim)])
    
    # Optimizer: categorical [adam, adamw]
    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    args.extend(["--optimizer", optimizer])
    
    # Scheduler: categorical [none, cosine, step]
    scheduler = trial.suggest_categorical("scheduler", ["none", "cosine", "step"])
    args.extend(["--scheduler", scheduler])
    
    # Grad clip: uniform [0.0, 5.0]
    grad_clip = trial.suggest_float("grad_clip_norm", 0.0, 5.0)
    args.extend(["--grad_clip_norm", str(grad_clip)])

    # Label smoothing: uniform [0.0, 0.2]
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    args.extend(["--label_smoothing", str(label_smoothing)])

    # Model specific
    if model_name in ["gat", "gcn"]:
        # Dropout: uniform [0.0, 0.7]
        dropout = trial.suggest_float("dropout", 0.0, 0.7)
        args.extend(["--dropout", str(dropout)])
        
        # Num layers: categorical [2, 3, 4]
        num_layers = trial.suggest_categorical("num_layers", [2, 3, 4])
        args.extend(["--num_layers", str(num_layers)])
    
    if model_name == "gat":
        # Additional GAT params could go here if search space allowed (e.g. heads)
        pass
        
    if model_name == "graphsage":
        # GraphSAGE doesn't support dropout/num_layers in current arch
        pass
        
    return args

def objective(trial, args):
    script_map = {
        "gat": "scripts/train_gat.py",
        "gcn": "scripts/train_gcn.py",
        "graphsage": "scripts/train_graphsage.py"
    }
    
    script = script_map[args.model]
    
    # Base command
    cmd = [sys.executable, script]
    
    # Suggest params
    trial_args = suggest_params(trial, args.model)
    cmd.extend(trial_args)
    
    # Fixed args
    # Use trial number as seed? Or fixed seed? 
    # User prompt: "seed: use [0,1,2] and report mean/std over seeds for the best config"
    # For tuning, we use a fixed seed to be deterministic PER TRIAL. 
    # Let's use seed=0 for tuning.
    cmd.extend(["--seed", "0"]) 
    
    # Results file
    results_file = f"tuning_results_{args.model}.jsonl"
    cmd.extend(["--results_file", results_file])
    
    # Trial ID
    cmd.extend(["--trial_id", str(trial.number)])
    
    # Device
    cmd.extend(["--device", args.device])
    
    if args.smoke_test:
        cmd.extend(["--epochs", "2"])

    cwd = os.getcwd()
    try:
        # Run process
        # Capture output to avoid cluttering stdout too much, or let it flow?
        # Let's capture and print only if failed.
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        
        if result.returncode != 0:
            print(f"Trial {trial.number} failed!")
            print(result.stderr)
            return float('nan') # Prune
            
        # Read result from file
        # We look for the line with trial_id
        val_f1 = None
        with open(results_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if str(data.get("trial_id")) == str(trial.number):
                        val_f1 = data.get("val_macro_f1")
                        break
                except:
                    continue
        
        if val_f1 is None:
            print(f"Trial {trial.number} completed but no result found in {results_file}")
            print(result.stderr)
            return float('nan')
            
        return val_f1

    except Exception as e:
        print(f"Trial {trial.number} exception: {e}")
        return float('nan')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["gat", "gcn", "graphsage"])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--storage", type=str, default="sqlite:///tuning.db")
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu", help="Device to run training on (cpu/cuda/mps/auto)")
    args = parser.parse_args()
    
    if args.smoke_test:
        args.n_trials = 2
        print("Smoke test mode: 2 trials")

    study_name = args.study_name or f"{args.model}_tuning"
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42) # Deterministic sampler seed
    )
    
    print(f"Starting tuning for {args.model} with {args.n_trials} trials...")
    
    from tqdm import tqdm
    with tqdm(total=args.n_trials, desc="Tuning Progress") as pbar:
        def progress_clbk(study, trial):
            pbar.update(1)
            pbar.set_postfix({"Best F1": f"{study.best_value:.4f}"})
            
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, callbacks=[progress_clbk])
    
    print("\n=== Tuning Complete ===")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best Val Macro F1: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    # Reconstruct reproduction command
    # Note: we need to handle the conditional params logic again or just print best_params
    print("\nReproduction Command:")
    cmd_parts = [f"python scripts/train_{args.model}.py"]
    for k, v in study.best_params.items():
        cmd_parts.append(f"--{k} {v}")
    cmd_parts.append("--seed 0")
    print(" ".join(cmd_parts))
    
    # Save top 10 to CSV
    df = study.trials_dataframe()
    # Filter valid
    df = df[df.state == optuna.trial.TrialState.COMPLETE]
    df = df.sort_values("value", ascending=False)
    
    out_csv = f"tuning_top10_{args.model}.csv"
    df.head(10).to_csv(out_csv, index=False)
    print(f"\nTop 10 trials saved to {out_csv}")

if __name__ == "__main__":
    main()
