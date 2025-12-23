import torch
from sklearn.metrics import accuracy_score

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        mask = data.y != -1
        pred = logits[mask].argmax(dim=1)
        return accuracy_score(data.y[mask].cpu(), pred.cpu())
