import torch
import torch.nn.functional as F
from attacks.base_attack import BaseAttack
from src.utils.graph_utils import edge_index_to_adj

class NettackStructuralAttack(BaseAttack):
    """
    NETTACK-style greedy targeted structural attack (structure-only)
    """

    def __init__(self, model, data, device):
        super().__init__(model, data, device)
        self.x = data.x.to(device)
        self.y = data.y.to(device)
        self.num_nodes = data.num_nodes

    def attack(self, target_node, edge_index, n_perturbations=5):
        adj = edge_index_to_adj(
            self.num_nodes, edge_index, self.device
        ).detach()

        adj.requires_grad_(True)

        for _ in range(n_perturbations):
            edge_index_adv = adj.nonzero(as_tuple=False).t()

            logits = self.model(self.x, edge_index_adv)
            loss = F.cross_entropy(
                logits[target_node].unsqueeze(0),
                self.y[target_node].unsqueeze(0)
            )

            loss.backward()

            grad = adj.grad.detach()
            scores = grad[target_node].abs()
            scores[target_node] = 0

            u = scores.argmax().item()

            adj[target_node, u] = 1 - adj[target_node, u]
            adj[u, target_node] = adj[target_node, u]

            adj.grad.zero_()

        return adj.nonzero(as_tuple=False).t()
