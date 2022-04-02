from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import dropout_adj

import torch
from torch_geometric.utils import degree, to_undirected


class EdgeRemoving(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = dropout_adj(edge_index, edge_attr=edge_weights, p=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class EdgeRemoving_weighted(Augmentor):
    def __init__(self, pe: float):
        super(EdgeRemoving_weighted, self).__init__()
        self.pe = pe
        self.threshold = 0.7

    def augment(self, g: Graph) -> Graph:
        x, edge_index, ew = g.unfold()
        edge_weights = degree_drop_weights(edge_index)
        edge_weights = edge_weights / edge_weights.mean() * self.pe
        edge_weights = edge_weights.where(edge_weights < self.threshold, torch.ones_like(edge_weights) * self.threshold)
        sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
        edge_index = edge_index[:, sel_mask]
        return Graph(x=x, edge_index=edge_index, edge_weights=ew)


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
    return weights