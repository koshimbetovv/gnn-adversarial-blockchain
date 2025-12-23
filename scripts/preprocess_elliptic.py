import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def main():
    features = pd.read_csv("../data/raw/elliptic/elliptic_txs_features.csv", header=None)
    classes = pd.read_csv("../data/raw/elliptic/elliptic_txs_classes.csv")
    edges = pd.read_csv("../data/raw/elliptic/elliptic_txs_edgelist.csv")

    features = features.sort_values(by=0)
    x = features.iloc[:, 2:].values
    x = StandardScaler().fit_transform(x)

    y_map = {"1": 1, "2": 0}
    y = classes["class"].map(y_map).fillna(-1).values

    edge_index = torch.tensor(edges[["txId1", "txId2"]].values).t()

    torch.save(
        {
            "x": torch.tensor(x, dtype=torch.float),
            "y": torch.tensor(y, dtype=torch.long),
            "edge_index": edge_index
        },
        "../data/processed/elliptic/data.pt"
    )

if __name__ == "__main__":
    main()
