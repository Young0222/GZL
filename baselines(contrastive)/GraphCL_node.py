import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool, GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid, Coauthor
import copy
import argparse
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv_raw(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv_raw, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))
    def forward(self, x, edge_index):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        z = [torch.cat(x, dim=1) for x in [zs]]
        return z


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z = self.encoder(x, edge_index)
        z1 = self.encoder(x1, edge_index1)
        z2 = self.encoder(x2, edge_index2)
        return z, z1, z2


def train(encoder_model, contrast_model, data, optimizer, device):
    encoder_model.train()
    data = data.to(device)
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index)
    loss = contrast_model(h1=z1, h2=z2)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_batch(encoder_model, contrast_model, data, optimizer, device, batch_size):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index)
    loss = 0
    num_nodes = z1.shape[0]
    num_batches = (num_nodes - 1) // batch_size + 1
    indices = torch.arange(0, num_nodes)
    for i in range(num_batches):
        torch.cuda.empty_cache()
        mask = indices[i * batch_size:(i + 1) * batch_size]
        loss += contrast_model(h1=z1[mask], h2=z2[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data, device):
    encoder_model.eval()
    data = data.to(device)
    z, _, _ = encoder_model(data.x, data.edge_index)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    device = torch.device('cuda:0')
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
    input_dim = data.x.shape[1]

    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    gconv = GConv(input_dim=input_dim, hidden_dim=128, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    # contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 101):
            if args.name == 'ogbn':
                loss = train_batch(encoder_model, contrast_model, data, optimizer, device, batch_size=args.batch_size)
            else:
                loss = train(encoder_model, contrast_model, data, optimizer, device)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            elapsed = pbar.format_dict["elapsed"]
            if epoch == 100:
                print("elapsed: ", elapsed)

    test_result = test(encoder_model, data_raw, device)
    print("coarsening_ratio:", args.coarsening_ratio)
    print(f'(E): Final Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='Cora') # name in {'Cora', 'CS', 'ogbn'},
    parser.add_argument('--coarsening_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=100)

    args = parser.parse_args()
    main()
