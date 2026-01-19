"""Nettack-style local greedy structural attack (baseline)."""

import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.attacks.base_attack import BaseAttack


class NettackLocalAttack(BaseAttack):
    def __init__(
        self,
        model,
        data,
        device,
        adj_list=None,
        undirected: bool = False,
        allow_removals: bool = False,
        seed: int | None = 0,
    ):
        super().__init__(model, data, device)
        self.x = data.x.to(device)
        self.y = data.y.to(device)
        self.undirected = bool(undirected)
        self.allow_removals = bool(allow_removals)
        self.adj_list = adj_list
        self.rng = random.Random(seed)

    def _build_adj_list(self, edge_index: torch.Tensor):
        adj = defaultdict(set)
        u_list = edge_index[0].tolist()
        v_list = edge_index[1].tolist()
        for u, v in zip(u_list, v_list):
            adj[u].add(v)
            if self.undirected:
                adj[v].add(u)
        return adj

    def _has_edge(self, edge_set: set[tuple[int, int]], u: int, v: int) -> bool:
        if (u, v) in edge_set:
            return True
        if self.undirected and (v, u) in edge_set:
            return True
        return False

    def _add_edge_set(self, edge_set: set[tuple[int, int]], u: int, v: int):
        edge_set.add((u, v))
        if self.undirected:
            edge_set.add((v, u))

    def _remove_edge_set(self, edge_set: set[tuple[int, int]], u: int, v: int):
        edge_set.discard((u, v))
        if self.undirected:
            edge_set.discard((v, u))

    def _add_edge(self, edge_set: set[tuple[int, int]], adj: dict[int, set[int]], u: int, v: int):
        self._add_edge_set(edge_set, u, v)
        adj[u].add(v)
        if self.undirected:
            adj[v].add(u)

    def _remove_edge(self, edge_set: set[tuple[int, int]], adj: dict[int, set[int]], u: int, v: int):
        self._remove_edge_set(edge_set, u, v)
        adj[u].discard(v)
        if self.undirected:
            adj[v].discard(u)

    @torch.no_grad()
    def _loss_on_target(self, edge_index: torch.Tensor, target_node: int) -> float:
        logits = self.model(self.x, edge_index)
        if int(self.y[target_node].item()) == -1:
            return float("-inf")
        loss = F.cross_entropy(logits[target_node].unsqueeze(0), self.y[target_node].unsqueeze(0))
        return float(loss.item())

    def attack(
        self,
        target_node: int,
        edge_index: torch.Tensor,
        n_perturbations: int = 5,
        sample_size: int = 200,
        include_neighbors: bool = True,
    ) -> torch.Tensor:
        edge_set: set[tuple[int, int]] = set(map(tuple, edge_index.t().tolist()))
        adj = self.adj_list if self.adj_list is not None else self._build_adj_list(edge_index)
        if target_node not in adj:
            adj[target_node] = set()

        all_nodes = set(range(self.data.num_nodes))

        for _ in tqdm(range(int(n_perturbations)), desc="Attacking | Perturbations:"):
            neighbors = set(adj[target_node])
            non_neighbors = list(all_nodes - neighbors - {target_node})
            sampled_non = self.rng.sample(non_neighbors, k=min(sample_size, len(non_neighbors)))

            candidates: list[tuple[str, int]] = []
            for u in sampled_non:
                if not self._has_edge(edge_set, target_node, u):
                    candidates.append(("add", u))

            if include_neighbors and self.allow_removals:
                for u in neighbors:
                    if self._has_edge(edge_set, target_node, u):
                        candidates.append(("remove", u))

            if not candidates:
                break

            best_loss = float("-inf")
            best_op = None

            for op, u in candidates:
                tmp_edges = edge_set.copy()
                if op == "add":
                    self._add_edge_set(tmp_edges, target_node, u)
                else:
                    self._remove_edge_set(tmp_edges, target_node, u)

                new_edge_index = torch.tensor(list(tmp_edges), dtype=torch.long).t().contiguous().to(self.device)
                loss_val = self._loss_on_target(new_edge_index, target_node)

                if loss_val > best_loss:
                    best_loss = loss_val
                    best_op = (op, u)

            if best_op is None:
                break

            op, u = best_op
            if op == "add":
                self._add_edge(edge_set, adj, target_node, u)
            else:
                self._remove_edge(edge_set, adj, target_node, u)

        return torch.tensor(list(edge_set), dtype=torch.long).t().contiguous().to(self.device)
