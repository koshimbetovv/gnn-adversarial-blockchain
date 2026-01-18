gnn-adversarial-blockchain/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── datasets/
│   │   ├── ibm_aml.yaml
│   │   ├── ethereum_phishing.yaml
│   ├── models/
│   │   ├── gcn.yaml
│   │   ├── graphsage.yaml
│   │   ├── rgcn.yaml
│   ├── attacks/
│   │   ├── nettack.yaml
│   │   ├── fgsm.yaml
│   │   ├── pgd.yaml
│   └── experiments/
│       ├── exp_baseline.yaml
│       ├── exp_attack_sweep.yaml
│
├── data/
│   ├── raw/
│   │   ├── ibm_aml/
│   │   ├── ethereum/
│   ├── processed/
│   │   ├── ibm_aml/
│   │   ├── ethereum/
│   └── splits/
│       ├── ibm_aml/
│       └── ethereum/
│
├── src/
│   ├── datasets/
│   │   ├── base_dataset.py
│   │   ├── ibm_aml.py
│   │   └── ethereum.py
│   │
│   ├── models/
│   │   ├── base_gnn.py
│   │   ├── gcn.py
│   │   ├── graphsage.py
│   │   └── rgcn.py
│   │
│   ├── attacks/
│   │   ├── base_attack.py
│   │   ├── nettack.py
│   │   ├── fgsm.py
│   │   └── pgd.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── metrics.py
│   │
│   ├── utils/
│   │   ├── seed.py
│   │   ├── logging.py
│   │   └── graph_utils.py
│   │
│   └── main.py
│
├── experiments/
│   ├── ibm_aml/
│   │   ├── gcn_baseline/
│   │   │   ├── config.yaml
│   │   │   ├── results.json
│   │   │   └── logs.txt
│   │   └── nettack/
│   │       └── ...
│   └── ethereum/
│       └── ...
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── summaries/
│
├── notebooks/
│   ├── data_analysis.ipynb
│   └── visualization.ipynb
│
└── scripts/
    ├── preprocess_ibm_aml.py
    ├── run_experiment.sh
    └── sweep_attacks.py
