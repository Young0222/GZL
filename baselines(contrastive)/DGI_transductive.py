import torch
import os.path as osp
import GCL.losses as L
import torch_geometric.transforms as T

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import SingleBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid, Coauthor
import copy
import argparse
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
import sys


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv, act in zip(self.layers, self.activations):
            z = conv(z, edge_index, edge_weight)
            z = act(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
        zn = self.encoder(*self.corruption(x, edge_index))
        return z, g, zn


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, g, zn = encoder_model(data.x, data.edge_index)
    loss = contrast_model(h=z, g=g, hn=zn)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_batch(encoder_model, contrast_model, data, optimizer, batch_size):
    encoder_model.train()
    optimizer.zero_grad()
    z, g, zn = encoder_model(data.x, data.edge_index)
    loss = 0
    num_nodes = z.shape[0]
    num_batches = (num_nodes - 1) // batch_size + 1
    indices = torch.arange(0, num_nodes)
    for i in range(num_batches):
        torch.cuda.empty_cache()
        mask = indices[i * batch_size:(i + 1) * batch_size]
        loss += contrast_model(h=z[mask], g=g, hn=zn[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    device = torch.device('cuda:2')
    path = osp.join(osp.expanduser('~'), 'datasets/'+args.name)
    print("path: ", path)
    if args.name == 'Cora':
        dataset = Planetoid(path, name='Cora')
        coarsen_features = np.load("/home/thu421/lzy/pvldb/coarse_data/cora/"+str(args.coarsening_ratio)+"cora_coarsen_features.npy")
        coarsen_edge = np.load("/home/thu421/lzy/pvldb/coarse_data/cora/"+str(args.coarsening_ratio)+"cora_coarsen_edge.npy")
    elif args.name == 'CS':
        dataset = Coauthor(path, name='CS', transform=T.NormalizeFeatures())
        coarsen_features = np.load("/home/thu421/lzy/pvldb/coarse_data/cs/"+str(args.coarsening_ratio)+"coarsen_features_np.npy")
        coarsen_edge = np.load("/home/thu421/lzy/pvldb/coarse_data/cs/"+str(args.coarsening_ratio)+"coarsen_edge_np.npy")
    elif args.name == 'ogbn':
        dataset = PygNodePropPredDataset(root=path, name="ogbn-arxiv")
        coarsen_features = np.load("/home/thu421/lzy/pvldb/coarse_data/ogbn/"+str(args.coarsening_ratio)+"coarsen_features_np.npy")
        coarsen_edge = np.load("/home/thu421/lzy/pvldb/coarse_data/ogbn/"+str(args.coarsening_ratio)+"coarsen_edge_np.npy")
        print("transform into undirected edges......")
        

    data = dataset[0]
    if args.name == 'ogbn':
        data.y = data.y.squeeze(dim=-1)
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    data_raw = copy.deepcopy(data)
    coarsen_features = torch.from_numpy(coarsen_features)
    coarsen_edge = torch.from_numpy(coarsen_edge)
    data.x = coarsen_features
    data.edge_index = coarsen_edge
    data = data.to(device)
    data_raw = data_raw.to(device)

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=128, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, hidden_dim=128).to(device)
    contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=args.epoch, desc='(T)') as pbar:
        for epoch in range(1, args.epoch+1):
            loss = train_batch(encoder_model, contrast_model, data, optimizer, batch_size=args.batch_size)
            # loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data_raw)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='ogbn') # name in {'Cora', 'CS', 'ogbn'},
    parser.add_argument('--coarsening_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=100)

    args = parser.parse_args()
    main()

