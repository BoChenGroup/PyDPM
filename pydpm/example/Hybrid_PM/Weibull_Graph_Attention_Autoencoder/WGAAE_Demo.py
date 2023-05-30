# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import argparse
import random
import numpy as np
import scipy.sparse as sp

from pydpm.model import WGAAE
from pydpm.utils import *
from pydpm.dataloader.graph_data import Graph_Processer
from pydpm.metric.roc_score import ROC_AP_SCORE

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

parser = argparse.ArgumentParser()

# device
parser.add_argument("--gpu_id", type=int, default=0, help="the id of gpu to deploy")
parser.add_argument('--seed', type=int, default=123, help='Setting random seed')

# dataset
parser.add_argument('--dataset', type=str, default='cora', help='Dataset string')
parser.add_argument('--dataset_path', type=str, default='../../../dataset/Planetoid', help="the file path of dataset")

# network settings
parser.add_argument('--z_dims', type=list, default=[64, 64, 64], help='Output dimension list')
parser.add_argument('--hid_dims', type=list, default=[128, 128, 128], help='Hidden dimension list')
parser.add_argument('--out_dim', type=int, default=32, help='Dimension of output')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads in GAT')

# optimizer
parser.add_argument("--lr", type=float, default=0.001, help="Adam: learning rate")

# training
parser.add_argument('--task', type=str, default='prediction', help='Prediction, clustering or classification')
parser.add_argument("--num_epochs", type=int, default=30000, help="Number of epochs of training")
parser.add_argument('--is_subgraph', type=bool, default=False, help='Whether subgraph')
parser.add_argument('--is_sample', type=bool, default=True, help='Whether sample nodes')
parser.add_argument('--num_sample', type=int, default=1500, help='Number of sampling nodes')
# parser.add_argument("--batch_size", type=int, default=1000, help="Size of the batches")
parser.add_argument("--MBratio", type=int, default=100, help="number of epochs of training")
parser.add_argument('--graph_lh', type=str, default='Laplacian', help='Graph likelihood')
parser.add_argument('--lambda', type=float, default=1.0, help='lamda')
parser.add_argument('--theta_norm', type=bool, default=False, help='Whether theta norm')

args = parser.parse_args()
args.device = 'cpu' if not torch.cuda.is_available() else f'cuda:{args.gpu_id}'


seed_everything(args.seed)

# Prepare for dataset
dataset = Planetoid(args.dataset_path, args.dataset)
data = dataset[0].to(args.device)
data.edge_index = data.edge_index[[1, 0]]
graph_processer = Graph_Processer()

adj_csc = graph_processer.graph_from_edges(data.edge_index, data.num_nodes).tocsc()
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = graph_processer.edges_split_from_graph(adj_csc)

# For encoder input and graph likelihood
adj_train = adj_train + sp.eye(adj_train.shape[0])
data.edge_index = graph_processer.edges_from_graph(adj_train.tocoo(), args.device)

model = WGAAE(in_dim=dataset.num_features, out_dim=args.out_dim, z_dims=args.z_dims, hid_dims=args.hid_dims, num_heads=args.num_heads, device=args.device)
optim = torch.optim.Adam(model.parameters())

# Training
best_AUC = best_AP = 0
for epoch_index in range(args.num_epochs):
    if epoch_index <= 200:
        for i in range(20):
            _, _ = model.train_one_epoch(data=data, optim=optim, epoch_index=epoch_index, is_sample=args.is_sample, is_subgraph=args.is_subgraph, args=args, is_train=False)
    else:
        for i in range(5):
            _, _ = model.train_one_epoch(data=data, optim=optim, epoch_index=epoch_index, is_sample=args.is_sample, is_subgraph=args.is_subgraph, args=args, is_train=False)
    train_local_params, Loss = model.train_one_epoch(data=data, optim=optim, epoch_index=epoch_index, is_sample=args.is_sample, is_subgraph=args.is_subgraph, args=args)

    if args.task == 'classification':
        [train_loss, train_loss_cls, train_recon_llh, train_graph_llh] = Loss
    else:
        [train_loss, train_recon_llh, train_graph_llh] = Loss

    if epoch_index % 1 == 0:
        test_local_params, Loss = model.test_one_epoch(data, is_sample=args.is_sample, is_subgraph=args.is_subgraph, args=args)

        # if args.task == 'classification':
        #     [test_loss, test_loss_cls, test_recon_llh, test_graph_llh] = Loss
        # else:
        #     [test_loss, test_recon_llh, test_graph_llh] = Loss
        # On classification task
        # accs = []
        # for mask in [dataset.train_mask, dataset.val_mask, dataset.test_mask]:
        #     accs.append(int((pred[mask] == dataset.y[mask]).sum()) / int(mask.sum()))
        # [train_acc, val_acc, tmp_test_acc] = accs
        # best_test_acc = np.maximum(best_test_acc, tmp_test_acc)

        # On prediction task
        theta = test_local_params[1]
        # Construct theta_concat for prediction
        theta_concat = None
        for layer in range(model._model_setting.num_layers):
            if layer == 0:
                theta_concat = model.u[layer] * theta[layer]
            else:
                theta_concat = torch.cat([theta_concat, model.u[layer] * theta[layer]], 0)
        theta_concat = theta_concat.cpu().detach().numpy()

        metric = ROC_AP_SCORE(test_edges, test_edges_false, adj_csc, emb=theta_concat.T)
        best_AUC = np.maximum(best_AUC, metric._AUC)
        best_AP = np.maximum(best_AP, metric._AP)

        print(f'Epoch[{epoch_index}|{args.num_epochs}]: loss:{train_loss}, graph_lh:{train_graph_llh}, recon_lh:{train_recon_llh}.'
              f' best_AUC:{best_AUC}, best_AP: {best_AP}')




