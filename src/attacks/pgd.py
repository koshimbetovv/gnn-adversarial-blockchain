import torch
import torch.nn.functional as F

from src.attacks.base_attack import BaseAttack


class PGDAttack(BaseAttack):
    """L_inf PGD feature attack for node classification.

    Scalability: optimize `delta` only for attacked nodes.
    """

    def __init__(self, model, data, device, clamp: tuple[float, float] | None = None):
        super().__init__(model, data, device)
        self.x = data.x.to(device).detach()
        self.y = data.y.to(device)
        self.edge_index = data.edge_index.to(device)
        self.clamp = clamp

    def attack(
        self,
        target_nodes: torch.Tensor,
        eps: float = 0.01,
        alpha: float = 0.002,
        steps: int = 10,
        random_start: bool = True,
        targeted: bool = False,
        target_label: int | None = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(target_nodes):
            target_nodes = torch.tensor(target_nodes, dtype=torch.long)
        target_nodes = target_nodes.to(self.device).long().view(-1)

        labeled_mask = self.y[target_nodes] != -1
        target_nodes = target_nodes[labeled_mask]
        if target_nodes.numel() == 0:
            return self.x.clone()

        if targeted:
            if target_label is None:
                raise ValueError("target_label must be provided when targeted=True")
            labels = torch.full((target_nodes.numel(),), int(target_label),
                                device=self.device, dtype=torch.long)
            direction = -1.0
        else:
            labels = self.y[target_nodes].long()
            direction = +1.0

        if random_start:
            delta = (2 * torch.rand((target_nodes.numel(), self.x.size(1)), device=self.device) - 1.0) * float(eps)
        else:
            delta = torch.zeros((target_nodes.numel(), self.x.size(1)), device=self.device)

        delta = delta.clamp(min=-float(eps), max=float(eps)).detach()

        for _ in range(int(steps)):
            delta.requires_grad_(True)

            x_adv = self.x.clone()
            x_adv[target_nodes] = x_adv[target_nodes] + delta

            logits = self.model(x_adv, self.edge_index)
            loss = F.cross_entropy(logits[target_nodes], labels)

            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

            delta = (delta + direction * float(alpha) * grad.sign()).detach()
            delta = delta.clamp(min=-float(eps), max=float(eps))

            if self.clamp is not None:
                x_tmp = self.x.clone()
                x_tmp[target_nodes] = x_tmp[target_nodes] + delta
                x_tmp = torch.clamp(x_tmp, min=self.clamp[0], max=self.clamp[1])
                delta = (x_tmp[target_nodes] - self.x[target_nodes]).detach()
                delta = delta.clamp(min=-float(eps), max=float(eps))

        x_out = self.x.clone()
        x_out[target_nodes] = x_out[target_nodes] + delta
        if self.clamp is not None:
            x_out = torch.clamp(x_out, min=self.clamp[0], max=self.clamp[1])

        return x_out.detach()
