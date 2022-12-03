# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import argparse
import random
import numpy as np

from pydpm.model import WGAAE
from pydpm.utils import *
# from pydpm.utils._graph_utils.graph_logging import init_wandb, log
# from pydpm.utils._graph_utils.preprocessing import mask_test_edges
from pydpm.dataloader.graph_data import *
from pydpm.metric.roc_score import ROC_AP_SCORE

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

parser = argparse.ArgumentParser()

# device
parser.add_argument("--gpu_id", type=int, default=0, help="the id of gpu to deploy")

# path
parser.add_argument('--dataset', type=str, default='cora', help='Dataset string')




parser.add_argument('--task', type=str, default='prediction', help='Prediction, clustering or classification')
parser.add_argument('--seed', type=int, default=123, help='Setting random seed')
parser.add_argument("--n_epochs", type=int, default=30000, help="Number of epochs of training")
parser.add_argument('--is_sample', type=bool, default=False, help='Whether to sample subgraph or not')
parser.add_argument("--batch_size", type=int, default=1000, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="Adam: learning rate")
parser.add_argument('--K', type=list, default=[32, 32, 32], help='Output dimension list')
parser.add_argument('--H', type=list, default=[64, 64, 64], help='Hidden dimension list')
parser.add_argument('--out_dim', type=int, default=32, help='Dimension of output')
parser.add_argument('--head_num', type=int, default=4, help='Number of heads in GAT')
parser.add_argument('--graph_lh', type=str, default='Laplacian', help='Graph likelihood')
parser.add_argument('--lamda', type=float, default=1.0, help='lamda')
parser.add_argument('--theta_norm', type=bool, default=False, help='Whether theta norm')
args = parser.parse_args()

init_wandb(name=f'GAT-{args.dataset}', heads=args.head_num, epochs=args.n_epochs,
           hidden_channels=args.H[0], lr=args.lr, device=args.device)
# Prepare for dataset
path = '../../dataset/Planetoid'
if not os.path.exists(path):
    os.mkdir(path)
dataset = Planetoid(path, args.dataset)
data = dataset[0].to(args.device)
# edge_index_t = dataset.edge_index[0]
data.edge_index = data.edge_index[[1, 0]]
adj_csc = graph_from_edges(data.edge_index, data.num_nodes, to_sparsetesor=False)[0].tocsc()

# Task: prediction
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_csc)

# For encoder input and graph likelihood
# adj_norm = normalize(adj_train)
adj_train = adj_train + sp.eye(adj_train.shape[0])
data.edge_index = sp_adj_to_edges(adj_train.tocoo(), args.device)

# Set random seed
seed_everything(args.seed)

# args.K[-1] = dataset.num_classes
# args.batch_size = dataset[0].num_nodes
model = WGAAE(args.K, args.H, dataset.num_features, args.head_num, args.out_dim, args.device)
model.initial(data, cls=dataset.num_classes, task=args.task, batch_size=args.batch_size, n_epochs=args.n_epochs, MBratio=1)

model = model.cuda()
model_opt = torch.optim.Adam(model.parameters())

# Training
best_AUC = best_AP = 0
for epoch in range(args.n_epochs):
    if epoch <= 100:
        for i in range(20):
            _, _ = model.train_full_graph(model_opt=model_opt, data=data, is_sample=args.is_sample, update_phi=False)
    else:
        for i in range(5):
            _, _ = model.train_full_graph(model_opt=model_opt, data=data, is_sample=args.is_sample, update_phi=False)
    train_local_params, Loss = model.train_full_graph(model_opt=model_opt, data=data, is_sample=args.is_sample, update_phi=True)

    if args.task == 'classification':
        [loss, loss_cls, recon_llh, graph_llh] = Loss
    else:
        [loss, recon_llh, graph_llh] = Loss

    # On classification task
    if epoch % 1 == 0:
        test_local_params = model.test_full_graph(data)
        # pred = test_local_params[0]
        # pred = pred.argmax(dim=-1)

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
        for layer in range(model._model_setting.T):
            if layer == 0:
                theta_concat = model.u[layer] * theta[layer]
            else:
                theta_concat = torch.cat([theta_concat, model.u[layer] * theta[layer]], 0)
        theta_concat = theta_concat.cpu().detach().numpy()

        metric = ROC_AP_SCORE(test_edges, test_edges_false, adj_csc, emb=theta_concat.T)
        best_AUC = np.maximum(best_AUC, metric._AUC)
        best_AP = np.maximum(best_AP, metric._AP)
        log(Epoch=epoch, Loss=loss, graph_llh=graph_llh, recon_llh=recon_llh,
            AUC=metric._AUC, AP=metric._AP, best_AUC=best_AUC, best_AP=best_AP)






