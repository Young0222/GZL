import torch
import os.path as osp
import GCL.losses as L

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import SingleBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from utils_coarsen import load_data, coarsening
import argparse
import sys


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = self.activation(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class FC(nn.Module):
    def __init__(self, hidden_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, local_fc, global_fc):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.local_fc = local_fc
        self.global_fc = global_fc

    def forward(self, x, edge_index, batch):
        z, g = self.encoder(x, edge_index, batch)
        return z, g

    def project(self, z, g):
        return self.local_fc(z), self.global_fc(g)


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
        z, g = encoder_model(coarsen_features, coarsen_edge, coarsen_batch)
        z, g = encoder_model.project(z, g)
        loss = contrast_model(h=z, g=g, batch=coarsen_batch)
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
        z, g = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
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
    dataloader = DataLoader(dataset, batch_size=128)
    input_dim = max(dataset.num_features, 1)

    gconv = GConv(input_dim=input_dim, hidden_dim=args.hidden_size, activation=torch.nn.ReLU, num_layers=2).to(device)
    fc1 = FC(hidden_dim=args.hidden_size * 2)
    fc2 = FC(hidden_dim=args.hidden_size * 2)
    encoder_model = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(device)
    contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)
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
