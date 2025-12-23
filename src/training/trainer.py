import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, data, device):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device

    def train(self, epochs=200, lr=0.01):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        for _ in range(epochs):
            self.model.train()
            opt.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(
                out[self.data.y != -1],
                self.data.y[self.data.y != -1]
            )
            loss.backward()
            opt.step()
