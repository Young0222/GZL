from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import drop_feature
from torch_geometric.utils import dropout_adj, degree, to_undirected
import torch


class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class FeatureMasking_weighted(Augmentor):
    def __init__(self, pf: float, node_deg: float):
        super(FeatureMasking_weighted, self).__init__()
        self.pf = pf
        self.node_deg = node_deg
        self.threshold = 0.7

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        w = feature_drop_weights(x, self.node_deg)
        w = w / w.mean() * self.pf
        w = w.where(w < self.threshold, torch.ones_like(w) * self.threshold)
        drop_prob = w
        drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
        x = x.clone()
        x[:, drop_mask] = 0.
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())
    return s
