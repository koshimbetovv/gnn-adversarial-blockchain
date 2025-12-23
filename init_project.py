import os

structure = {
    "config/datasets": ["ibm_aml.yaml", "ethereum_phishing.yaml"],
    "config/models": ["gcn.yaml", "graphsage.yaml", "rgcn.yaml", "gat.yaml"],
    "config/attacks": ["nettack.yaml", "fgsm.yaml", "pgd.yaml", "random_node_inject.yaml", "random_edge_inject.yaml", "node_inject_plus_fgsm.yaml"],
    "config/experiments": ["exp_baseline.yaml", "exp_attack_sweep.yaml"],

    "data/raw/ibm_aml": [],
    "data/raw/ethereum": [],
    "data/processed/ibm_aml": [],
    "data/processed/ethereum": [],
    "data/splits/ibm_aml": [],
    "data/splits/ethereum": [],

    "src/datasets": ["base_dataset.py", "ibm_aml.py", "ethereum.py"],
    "src/models": ["base_gnn.py", "gcn.py", "graphsage.py", "rgcn.py", "gat.py"],
    "src/attacks": ["base_attack.py", "nettack.py", "fgsm.py", "pgd.py", "random_node_inject.py", "random_edge_inject.py", "node_inject_plus_fgsm.py"],
    "src/training": ["trainer.py", "evaluator.py", "metrics.py"],
    "src/utils": ["seed.py", "logging.py", "graph_utils.py"],
    "src": ["main.py"],

    "experiments/ibm_aml": [],
    "experiments/ethereum": [],
    "results/tables": [],
    "results/figures": [],
    "results/summaries": [],
    "notebooks": [],
    "scripts": ["preprocess_ibm_aml.py", "run_experiment.sh"]
}

for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    for f in files:
        path = os.path.join(folder, f)
        if not os.path.exists(path):
            open(path, "w").close()

print("✅ Project structure created.")
