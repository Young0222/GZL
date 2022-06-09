import argparse
import os.path as osp
import random
import pickle as pkl
import networkx as nx
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import sys
import numpy as np
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid, CitationFull, Coauthor, Amazon, WikiCS
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model_graph import Encoder, Model, drop_feature
from eval import label_classification
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from GCL.eval import SVMEvaluator
from utils_coarsen import load_data, coarsening
import sys


def readout(z):
    N = z.shape[0]
    res = torch.sum(z, dim=0) / N
    return torch.sigmoid(res)


def train(model: Model, coarsen_features_list, coarsen_edge_list, coarsen_batch_list, lambd, xi, nmb_communities):
    model.train()
    epoch_loss = 0
    for i in range(len(coarsen_features_list)):
        optimizer.zero_grad()
        edge_index_1 = dropout_adj(coarsen_edge_list[i], p=drop_edge_rate_1)[0]
        edge_index_2 = dropout_adj(coarsen_edge_list[i], p=drop_edge_rate_2)[0]
        x_1 = drop_feature(coarsen_features_list[i], drop_feature_rate_1)
        x_2 = drop_feature(coarsen_features_list[i], drop_feature_rate_2)
        z1, _ = model(x_1, edge_index_1, coarsen_batch_list[i])
        z2, _ = model(x_2, edge_index_2, coarsen_batch_list[i])
        z_raw, _ = model(coarsen_features_list[i], coarsen_edge_list[i], coarsen_batch_list[i])
        z1_graph = readout(z1)
        z2_graph = readout(z2)
        z_raw_graph = readout(z_raw)
        output_dim = z1.size(1)
        C = torch.nn.Linear(in_features=output_dim, out_features=nmb_communities, bias=False, device=z1.device)
        loss, C = model.loss(z1, z2, z_raw, z1_graph, z2_graph, z_raw_graph, C, lambd, xi, nmb_communities)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    with torch.no_grad():
        w = C.weight.data.clone()
        w = nn.functional.normalize(w, dim=0, p=2)
        C.weight.copy_(w)

    return epoch_loss


def test(model: Model, dataloader, device, final=False):
    model.eval()
    x = []
    y = []
    for data in dataloader:
        data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        z, g = model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }


def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'CS', 'Physics', 'Computers', 'Photo', 'Wiki', 'ogbn']
        name = 'dblp' if name == 'DBLP' else name
        if name in ['Cora', 'CiteSeer', 'PubMed']: 
            return Planetoid(
            path,
            name)
        elif name in ['CS', 'Physics']:
            return Coauthor(
            path,
            name,
            transform=T.NormalizeFeatures())
        elif name in ['Computers', 'Photo']:
            return Amazon(
            path,
            name,
            transform=T.NormalizeFeatures())
        elif name in ['Wiki']:
            return WikiCS(
            path,
            transform=T.NormalizeFeatures())
        elif name in ["ogbn"]:
            dataset = PygNodePropPredDataset(
                root=path,
                name="ogbn-arxiv",
            )
            return dataset
        else:
            return CitationFull(
            path,
            name)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # r_inv = np.power(rowsum, -1).flatten()
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx


def data_coarsen(dataloader, device):
    coarsen_features_list = []
    coarsen_edge_list = []
    coarsen_batch_list = []
    for data in dataloader:
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        args.num_features, args.num_classes, candidate, C_list, Gc_list = coarsening(data, 1-args.coarsening_ratio, args.coarsening_method)
        _, coarsen_features, coarsen_edge, coarsen_batch = load_data(data, candidate, C_list, Gc_list, args.experiment)
        coarsen_features_list.append(coarsen_features.to(device))
        coarsen_edge_list.append(coarsen_edge.to(device))
        coarsen_batch_list.append(coarsen_batch.to(device))
    return coarsen_features_list, coarsen_edge_list, coarsen_batch_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PTC_MR') #'MUTAG', 'PTC_MR', 'MSRC_21'
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--coarsening_ratio', type=float, default=0.1)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'

    parser.add_argument('--nmb_communities', type=int, default=100)
    parser.add_argument('--num_hidden', type=int, default=128)
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.2)
    parser.add_argument('--drop_feature_rate_1', type=float, default=0.0)
    parser.add_argument('--drop_feature_rate_2', type=float, default=0.0)
    parser.add_argument('--xi', type=float, default=0.08)
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu': nn.RReLU(), })[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']
    lambd = config['lambd']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    # name = config['dataset']
    # nmb_communities = config['nmb_communities']
    # num_hidden = config['num_hidden']
    # drop_edge_rate_1 = config['drop_edge_rate_1']
    # drop_edge_rate_2 = config['drop_edge_rate_2']
    # drop_feature_rate_1 = config['drop_feature_rate_1']
    # drop_feature_rate_2 = config['drop_feature_rate_2']
    # xi = config['xi']

    name = args.dataset
    nmb_communities = args.nmb_communities
    num_hidden = args.num_hidden
    drop_edge_rate_1 = args.drop_edge_rate_1
    drop_edge_rate_2 = args.drop_edge_rate_2
    drop_feature_rate_1 = args.drop_feature_rate_1
    drop_feature_rate_2 = args.drop_feature_rate_2
    xi = args.xi

    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = TUDataset(path, name=name)
    dataloader = DataLoader(dataset, batch_size=128)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    coarsen_features_list, coarsen_edge_list, coarsen_batch_list = data_coarsen(dataloader, device)

    encoder = Encoder(dataset.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    norm_list = []

    # training phase
    for epoch in range(1, num_epochs + 1):
        loss = train(model, coarsen_features_list, coarsen_edge_list, coarsen_batch_list, lambd, xi, nmb_communities)
        now = t()
        if epoch == num_epochs:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, 'f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    # testing phase
    print("=== Final ===")
    print("ratio: ", args.coarsening_ratio, "k: ", args.nmb_communities,"h: ", args.num_hidden,"p1: ", args.drop_edge_rate_1,"p2: ", args.drop_edge_rate_2,"p3: ", args.drop_feature_rate_1,"p4: ", args.drop_feature_rate_2,"x: ", args.xi)
    for i in range(2):
        test_result = test(model, dataloader, device, final=True)
        print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


