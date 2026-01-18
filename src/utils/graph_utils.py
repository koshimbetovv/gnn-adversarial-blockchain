import torch

def edge_index_to_adj(num_nodes, edge_index, device):
    adj = torch.zeros((num_nodes, num_nodes), device=device)
    adj[edge_index[0], edge_index[1]] = 1
    return adj



