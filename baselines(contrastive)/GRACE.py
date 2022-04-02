import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Coauthor
import copy
import argparse
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_batch(encoder_model, contrast_model, data, optimizer, batch_size):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = 0
    num_nodes = h1.shape[0]
    num_batches = (num_nodes - 1) // batch_size + 1
    indices = torch.arange(0, num_nodes)
    for i in range(num_batches):
        torch.cuda.empty_cache()
        mask = indices[i * batch_size:(i + 1) * batch_size]
        loss += contrast_model(h1[mask], h2[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
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

    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=128, activation=torch.nn.ReLU, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=128, proj_dim=128).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)

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
