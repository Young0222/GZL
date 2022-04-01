import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
# import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull, Coauthor, Amazon, WikiCS
# from torch_geometric.utils import dropout_adj
from torch_geometric.nn import Node2Vec
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected

# from model import Encoder, Model, drop_feature
from eval import label_classification
# from utils import load_data, coarsening
import numpy as np


def train(model):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(model, data, final=False):
    model.eval()
    z = model()
    # acc = model.test(z[data.train_mask], data.y[data.train_mask], z[data.test_mask], data.y[data.test_mask], max_iter=150) # 使用train_mask训练一个分类器，用test_mask分类

    acc, _ = label_classification(z, data.y, ratio=0.1)
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--coarsening_ratio', type=float, default=0.1)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    parser.add_argument('--walk_length', type=int, default=20)
    parser.add_argument('--context_size', type=int, default=10)
    parser.add_argument('--num_epoch', type=int, default=0, help="如果设置了num_epoch，将忽略config中的设置")
    # parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config['seed'])
    random.seed(12345)

    # learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    # num_proj_hidden = config['num_proj_hidden']
    # activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu': nn.RReLU(), })[config['activation']]
    # base_model = ({'GCNConv': GCNConv})[config['base_model']]
    # num_layers = config['num_layers']

    # drop_edge_rate_1 = config['drop_edge_rate_1']
    # drop_edge_rate_2 = config['drop_edge_rate_2']
    # drop_feature_rate_1 = config['drop_feature_rate_1']
    # drop_feature_rate_2 = config['drop_feature_rate_2']
    # tau = config['tau']
    if args.num_epoch > 0:
        num_epochs = args.num_epoch
    else:
        num_epochs = config['num_epochs']
    # weight_decay = config['weight_decay']

    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'CS', 'Physics', 'Computers', 'Photo', 'Wiki', 'ogbn-arxiv']
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
        elif name in ["ogbn-arxiv"]:
            dataset = PygNodePropPredDataset(
                root=path,
                name="ogbn-arxiv",
            )
            return dataset
        else:
            return CitationFull(
            path,
            name)

    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    if args.dataset in ["ogbn-arxiv"]:
        print("transform into undirected edges......")
        data.y = data.y.squeeze(dim=-1)
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    print("===="*20)
    print("SHAPE of features", data.x.shape)
    # coarsen_features = torch.from_numpy(np.load(osp.join('/home/thu421/lzy/pvldb/coarse_data', args.dataset.lower(), str(args.coarsening_ratio) + args.dataset.lower() + '_coarsen_features.npy')))
    # coarsen_edge = torch.from_numpy(np.load(osp.join('/home/thu421/lzy/pvldb/coarse_data', args.dataset.lower(), str(args.coarsening_ratio) + args.dataset.lower() + '_coarsen_edge.npy')))
    
    # print("SHAPE of coarsen features", coarsen_features.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    # coarsen_features = coarsen_features.to(device)
    # coarsen_edge = coarsen_edge.to(device)

    # encoder = Encoder(dataset.num_features, num_hidden, activation,
    #                   base_model=base_model, k=num_layers).to(device)
    # model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # model = Node2Vec(data.edge_index, embedding_dim=num_hidden, walk_length=args.walk_length, context_size=args.context_size, sparse=True).to(device)
    model = Node2Vec(data.edge_index, p=0.25, q=4, embedding_dim=num_hidden, walk_length=args.walk_length, context_size=args.context_size, sparse=True).to(device)
    
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
    loader = model.loader(batch_size=128, shuffle=True)

    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        # loss = train(model, data.x, data.edge_index)
        loss = train(model)

        now = t()
        if epoch % 10 == 0:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, 'f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Final ===")
    for i in range(3):
        acc =  test(model, data, final=True)
        print("acc: ", acc)
