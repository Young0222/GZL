import copy
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import BootstrapContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import WikiCS, Planetoid, Coauthor
import copy
import argparse
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected


class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch'):
        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    _, _, h1_pred, h2_pred, h1_target, h2_target = encoder_model(data.x, data.edge_index, data.edge_attr)
    loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred, h1_target=h1_target.detach(), h2_target=h2_target.detach())
    loss.backward()
    optimizer.step()
    encoder_model.update_target_encoder(0.99)
    return loss.item()


def train_batch(encoder_model, contrast_model, data, optimizer, batch_size):
    encoder_model.train()
    optimizer.zero_grad()
    _, _, h1_pred, h2_pred, h1_target, h2_target = encoder_model(data.x, data.edge_index, data.edge_attr)
    loss = 0
    num_nodes = h1_pred.shape[0]
    num_batches = (num_nodes - 1) // batch_size + 1
    indices = torch.arange(0, num_nodes)
    for i in range(num_batches):
        torch.cuda.empty_cache()
        mask = indices[i * batch_size:(i + 1) * batch_size]
        loss += contrast_model(h1_pred=h1_pred[mask], h2_pred=h2_pred[mask], h1_target=h1_target.detach()[mask], h2_target=h2_target.detach()[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    h1, h2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = torch.cat([h1, h2], dim=1)
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

    aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=128, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=128).to(device)
    contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='L2L').to(device)

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
