import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from utils_coarsen import load_data, coarsening
import argparse
import sys


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv in self.layers:
            z = conv(z, edge_index)
            z = self.activation(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        g = torch.cat(gs, dim=1)
        return z, g


class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)


class Encoder(torch.nn.Module):
    def __init__(self, gcn1, gcn2, mlp1, mlp2, aug1, aug2):
        super(Encoder, self).__init__()
        self.gcn1 = gcn1
        self.gcn2 = gcn2
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.aug1 = aug1
        self.aug2 = aug2

    def forward(self, x, edge_index, batch):
        x1, edge_index1, edge_weight1 = self.aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = self.aug2(x, edge_index)
        z1, g1 = self.gcn1(x1, edge_index1, batch)
        z2, g2 = self.gcn2(x2, edge_index2, batch)
        h1, h2 = [self.mlp1(h) for h in [z1, z2]]
        g1, g2 = [self.mlp2(g) for g in [g1, g2]]
        return h1, h2, g1, g2


def data_coarsen(dataloader):
    coarsen_features_list = []
    coarsen_edge_list = []
    coarsen_batch_list = []
    for data in dataloader:
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        args.num_features, args.num_classes, candidate, C_list, Gc_list = coarsening(data, 1-args.coarsening_ratio, args.coarsening_method)
        _, coarsen_features, coarsen_edge, coarsen_batch = load_data(data, candidate, C_list, Gc_list, args.experiment)
        coarsen_features_list.append(coarsen_features)
        coarsen_edge_list.append(coarsen_edge)
        coarsen_batch_list.append(coarsen_batch)
    return coarsen_features_list, coarsen_edge_list, coarsen_batch_list



def train(encoder_model, contrast_model, coarsen_features_list, coarsen_edge_list, coarsen_batch_list, optimizer, device):
    encoder_model.train()
    epoch_loss = 0
    for i in range(len(coarsen_features_list)):
        coarsen_features = coarsen_features_list[i].to(device)
        coarsen_edge = coarsen_edge_list[i].to(device)
        coarsen_batch = coarsen_batch_list[i].to(device)
        optimizer.zero_grad()
        h1, h2, g1, g2 = encoder_model(coarsen_features, coarsen_edge, coarsen_batch)
        loss = contrast_model(h1=h1, h2=h2, g1=g1, g2=g2, batch=coarsen_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss


def test(encoder_model, dataloader, device):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g1 + g2)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result


def main():
    device = torch.device('cuda:0')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = TUDataset(path, name=args.name)
    # dataset = TUDataset(path, name='NCI1')
    dataloader = DataLoader(dataset, batch_size=128)
    input_dim = max(dataset.num_features, 1)

    aug1 = A.Identity()
    aug2 = A.PPRDiffusion(alpha=0.2, use_cache=False)
    gcn1 = GConv(input_dim=input_dim, hidden_dim=args.hidden_size, num_layers=2).to(device)
    gcn2 = GConv(input_dim=input_dim, hidden_dim=args.hidden_size, num_layers=2).to(device)
    mlp1 = FC(input_dim=args.hidden_size, output_dim=args.hidden_size)
    mlp2 = FC(input_dim=args.hidden_size * 2, output_dim=args.hidden_size)
    encoder_model = Encoder(gcn1=gcn1, gcn2=gcn2, mlp1=mlp1, mlp2=mlp2, aug1=aug1, aug2=aug2).to(device)
    contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=0.01)
    coarsen_features_list, coarsen_edge_list, coarsen_batch_list = data_coarsen(dataloader)

    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 101):
            loss = train(encoder_model, contrast_model, coarsen_features_list, coarsen_edge_list, coarsen_batch_list, optimizer, device)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            elapsed = pbar.format_dict["elapsed"]
            if epoch == 100:
                print("elapsed: ", elapsed)

    test_result = test(encoder_model, dataloader, device)
    print("ratio: ", args.coarsening_ratio)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='MUTAG') #'MUTAG', 'PTC_MR', 'IMDB-BINARY'
    parser.add_argument('--coarsening_ratio', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    parser.add_argument('--hidden_size', type=int, default='128')

    args = parser.parse_args()
    main()
