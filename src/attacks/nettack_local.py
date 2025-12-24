import random
import torch
import torch.nn.functional as F
from attacks.base_attack import BaseAttack

class NettackLocalAttack(BaseAttack):
    def __init__(self, model, data, adj_list, device):
        super().__init__(model, data, device)
        self.adj_list = adj_list
        self.x = data.x.to(device)
        self.y = data.y.to(device)

    def attack(self, target_node, edge_index, n_perturbations=5, sample_size=100):
        edge_set = set(map(tuple, edge_index.t().tolist()))

        for _ in range(n_perturbations):
            neighbors = self.adj_list[target_node]
            all_nodes = set(range(self.data.num_nodes))
            non_neighbors = list(all_nodes - neighbors - {target_node})

            sampled = random.sample(
                non_neighbors, min(sample_size, len(non_neighbors))
            )
            candidates = list(neighbors) + sampled

            best_loss = -1
            best_edge = None

            for u in candidates:
                if (target_node, u) in edge_set:
                    continue

                new_edge_index = torch.tensor(
                    list(edge_set | {(target_node, u)})
                ).t().to(self.device)

                with torch.no_grad():
                    logits = self.model(self.x, new_edge_index)
                    loss = F.cross_entropy(
                        logits[target_node].unsqueeze(0),
                        self.y[target_node].unsqueeze(0)
                    )

                if loss.item() > best_loss:
                    best_loss = loss.item()
                    best_edge = u

            if best_edge is not None:
                edge_set.add((target_node, best_edge))
                self.adj_list[target_node].add(best_edge)

        return torch.tensor(list(edge_set)).t().to(self.device)
