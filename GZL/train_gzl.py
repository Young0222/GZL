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

from model import Encoder, Model, drop_feature
from eval import label_classification
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected


def readout(z):
    N = z.shape[0]
    res = torch.sum(z, dim=0) / N
    return torch.sigmoid(res)

def train(model: Model, x, edge_index, D_inv, lambd, xi, nmb_communities):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    z_raw = model(data.x, data.edge_index)
    z1_graph = readout(z1)
    z2_graph = readout(z2)
    z_raw_graph = readout(z_raw)
    
    C = torch.nn.Linear(in_features=z2.shape[1], out_features=nmb_communities, bias=False, device=z1.device)
    loss, C = model.loss(z1, z2, z_raw, z1_graph, z2_graph, z_raw_graph, D_inv, C, lambd, xi, nmb_communities)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        w = C.weight.data.clone()
        w = nn.functional.normalize(w, dim=0, p=2)
        C.weight.copy_(w)

    return loss.item()


def train_batch(model: Model, x, edge_index, D_inv, lambd, xi, nmb_communities, batch_size):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    z_raw = model(data.x, data.edge_index)
    z1_graph = readout(z1)
    z2_graph = readout(z2)
    z_raw_graph = readout(z_raw)
    output_dim = z1.size(1)
    C = torch.nn.Linear(in_features=output_dim, out_features=nmb_communities, bias=False, device=z1.device)

    loss = 0
    num_nodes = z1.shape[0]
    num_batches = (num_nodes - 1) // batch_size + 1
    indices = torch.arange(0, num_nodes)
    for i in range(num_batches):
        torch.cuda.empty_cache()
        mask = indices[i * batch_size:(i + 1) * batch_size]
        loss_batch, C = model.loss(z1[mask], z2[mask], z_raw[mask], z1_graph, z2_graph, z_raw_graph, D_inv, C, lambd, xi, nmb_communities)
        loss += loss_batch
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        w = C.weight.data.clone()
        w = nn.functional.normalize(w, dim=0, p=2)
        C.weight.copy_(w)

    return loss.item()


def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    label_classification(z, y, ratio=0.1)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--coarsening_ratio', type=float, default=0.1)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    args = parser.parse_args()
    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu': nn.RReLU(), })[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    nmb_communities = config['nmb_communities']
    lambd = config['lambd']
    xi = config['xi']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']


    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    if args.dataset in ["ogbn"]:
        print("transform into undirected edges......")
        data.y = data.y.squeeze(dim=-1)
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    if args.dataset == 'Cora':
        coarsen_features = np.load("/home/thu421/lzy/pvldb/coarse_data/cora/"+str(args.coarsening_ratio)+"cora_coarsen_features.npy")
        coarsen_edge = np.load("/home/thu421/lzy/pvldb/coarse_data/cora/"+str(args.coarsening_ratio)+"cora_coarsen_edge.npy")
    elif args.dataset == 'CS':
        coarsen_features = np.load("/home/thu421/lzy/pvldb/coarse_data/cs/"+str(args.coarsening_ratio)+"coarsen_features_np.npy")
        coarsen_edge = np.load("/home/thu421/lzy/pvldb/coarse_data/cs/"+str(args.coarsening_ratio)+"coarsen_edge_np.npy")
    elif args.dataset == 'ogbn':
        coarsen_features = np.load("/home/thu421/lzy/pvldb/coarse_data/ogbn/"+str(args.coarsening_ratio)+"coarsen_features_np.npy")
        coarsen_edge = np.load("/home/thu421/lzy/pvldb/coarse_data/ogbn/"+str(args.coarsening_ratio)+"coarsen_edge_np.npy")


    coarsen_features = torch.from_numpy(coarsen_features)
    coarsen_edge = torch.from_numpy(coarsen_edge)
    edges = coarsen_edge
    N = coarsen_features.shape[0] # number of samples to assign

    # building adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0, :], edges[1, :])), shape=(N, N), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    I_NN_row = np.arange(N)
    I_NN_data = np.ones(N)
    I_NN = sp.coo_matrix((I_NN_data, (I_NN_row, I_NN_row)), shape=(N, N), dtype=np.float32)
    D =  I_NN - (lambd/xi) * (adj + adj.T)
    D_inv = sp.linalg.inv(D)
    D_inv = np.transpose(D_inv)
    D_inv = sparse_mx_to_torch_sparse_tensor(D_inv)

    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    # adj = adj.to_dense()
    # I_NN = torch.eye(N, device=adj.device)
    # D =  I_NN - (lambd/xi) * (adj + adj.t())
    # D_inv = D.inverse()
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    D_inv = D_inv.to(device)
    edges = edges.to(device)
    coarsen_features = coarsen_features.to(device)

    encoder = Encoder(dataset.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    norm_list = []

    # training phase
    for epoch in range(1, num_epochs + 1):
        if args.dataset == 'ogbn':
            loss = train_batch(model, coarsen_features, edges, D_inv, lambd, xi, nmb_communities, batch=args.batch_size)
        else:
            loss = train(model, coarsen_features, edges, D_inv, lambd, xi, nmb_communities)
        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, 'f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    # testing phase
    print("=== Final ===")
    print("ratio: ", args.coarsening_ratio)
    for i in range(10):
        test(model, data.x, data.edge_index, data.y, final=True)

