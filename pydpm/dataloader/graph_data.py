#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import numpy as np
import os
import pickle
import scipy.sparse as sp
import torch
import torch_geometric.datasets as datasets
from torch_sparse import SparseTensor
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Dataset, DataLoader
from ..utils.utils import cosine_simlarity

def graph_from_data(data, threshold, binary=True):
    graph = cosine_simlarity(data, data)
    graph[(np.arange(graph.shape[0]), np.arange(graph.shape[0]))] = 0
    graph[np.where(graph < threshold)] = 0
    if binary:
        graph[np.where(graph > threshold - 0.01)] = 1

    return graph

def graph_from_edges(edge_index, n_nodes, to_sparsetesor=True):

    if to_sparsetesor:
        sp_adj = SparseTensor(row=edge_index[0],
                              col=edge_index[1],
                              value=torch.ones(edge_index[0].shape[0], device=edge_index.device),
                              sparse_sizes=(n_nodes, n_nodes))
        adj = sp_adj.to_dense()
    else:
        row = edge_index[0].detach().cpu().numpy()
        col = edge_index[1].detach().cpu().numpy()
        data = np.ones_like(row)
        sp_adj = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        adj = sp_adj.todense()

    return sp_adj, adj

def sp_adj_to_edges(adj, device):
    edge_index = torch.empty((2, adj.row.shape[0]), dtype=torch.long, device=device)
    edge_index[0] = torch.tensor(adj.row, dtype=torch.long, device=device)
    edge_index[1] = torch.tensor(adj.col, dtype=torch.long, device=device)

    return edge_index

def normalize(sp_adj):
    """
    :param sp_adj: scipy.sparse.coo_matrix
    :return: normalized_adj
    """
    adj_ = sp_adj + sp.eye(sp_adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

# ======================================== CustomDataset ======================================================== #

class CustomDataset(InMemoryDataset):
    def __init__(self, root, data_name, transform=None, pre_transform=None, threshold=None):
        super().__init__(root, transform, pre_transform)

        with open(os.path.join(root, data_name), 'rb') as f:
            data = pickle.load(f)
            x = torch.from_numpy(data['fea']).to(torch.float32)
            y = None
            if 'label' in data.keys():
                y = torch.from_numpy(data['label']).to(torch.long)

            # Construct direct graph
            if 'graph' in data.keys():
                graph_ = data['graph']
            else:
                graph_ = cosine_simlarity(data['fea'], data['fea'])
                graph_[(np.arange(graph_.shape[0]), np.arange(graph_.shape[0]))] = 0
                if threshold is not None:
                    graph_[np.where(graph_ < threshold)] = 0
                    graph_[np.where(graph_ > threshold - 0.01)] = 1

            sp_graph = sp.coo_matrix(graph_, shape=graph_.shape)
            row = torch.from_numpy(sp_graph.row).to(torch.long)
            col = torch.from_numpy(sp_graph.col).to(torch.long)
            val = torch.from_numpy(sp_graph.data).to(torch.float32)
            self.edge_value = val
            edge_index = torch.stack([row, col], dim=0)

            # Build train, valid and test datasets
            train_mask = torch.zeros(x.size(0), dtype=torch.bool)
            train_mask[torch.tensor(data['tr'])] = True

            val_mask = torch.zeros(x.size(0), dtype=torch.bool)
            val_mask[torch.tensor(data['va'])] = True

            test_mask = torch.zeros(x.size(0), dtype=torch.bool)
            test_mask[torch.tensor(data['te'])] = True

        self.data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)


def graph_dataloader(root='../dataset/', data_name='Planetoid', transform=None, pre_transform=None, threshold=None,
               batch_size=500, shuffle=True, drop_last=True, num_workers=4):

    dataset = CustomDataset(root=root, data_name=data_name, transform=transform, pre_transform=pre_transform,
                            threshold=threshold)

    print('Dataset of {} has been processed!'.format(data_name))
    return dataset
