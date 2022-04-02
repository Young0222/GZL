import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from utils_coarsen import load_data, coarsening
import argparse
import sys
from torch_geometric.utils import to_undirected


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
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

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


def data_coarsen(dataloader):
    coarsen_features_list = []
    coarsen_edge_list = []
    coarsen_batch_list = []
    for data in dataloader:
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        # print("transform into undirected edges......")
        # data.y = data.y.squeeze(dim=-1)
        # data.edge_index = to_undirected(data.edge_index, data.num_nodes)

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
        _, _, _, _, g1, g2 = encoder_model(coarsen_features, coarsen_edge, coarsen_batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=coarsen_batch)
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
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        print("g shape: ", g.shape)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    print("x: ", x.shape, "y: ", y.shape)
    sys.exit()
    result = SVMEvaluator(linear=True)(x, y, split)
    return result


def main():
    device = torch.device('cuda:0')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = TUDataset(path, name=args.name)
    dataloader = DataLoader(dataset, batch_size=128)
    input_dim = max(dataset.num_features, 1)

    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    gconv = GConv(input_dim=input_dim, hidden_dim=args.hidden_size, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
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
