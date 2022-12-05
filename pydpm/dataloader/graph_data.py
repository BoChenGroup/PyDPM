#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import pickle
import numpy as np
import scipy.sparse as sp

from ..utils.utils import cosine_simlarity

import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Dataset, DataLoader


class Graph_Processer(object):
    def __init__(self):
        pass

    def graph_from_node_feature(self, data, threshold, binary=True):
        graph = cosine_simlarity(data, data)
        graph[(np.arange(graph.shape[0]), np.arange(graph.shape[0]))] = 0
        graph[np.where(graph < threshold)] = 0
        if binary:
            graph[np.where(graph > threshold - 0.01)] = 1

        return graph

    def graph_from_edges(self, edge_index, n_nodes, to_sparse=True):

        if to_sparse:
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

    def edges_from_graph(self, adj, device, is_sparse=True):

        if is_sparse:
            edge_index = torch.empty((2, adj.row.shape[0]), dtype=torch.long, device=device)
            edge_index[0] = torch.tensor(adj.row, dtype=torch.long, device=device)
            edge_index[1] = torch.tensor(adj.col, dtype=torch.long, device=device)
        else:
            # TODO
            pass

        return edge_index

    def graph_normalize(self, sp_adj, is_sparse=True):
        """
        :param sp_adj: scipy.sparse.coo_matrix
        :return: normalized_adj
        """
        if is_sparse:
            adj_ = sp_adj + sp.eye(sp_adj.shape[0])
            rowsum = np.array(adj_.sum(1))
            degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
            adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        else:
            # TODO
            pass
        return adj_normalized

    def graph_to_tuple(self, sparse_mx, is_sparse: True):
        if not sp.isspmatrix_coo(sparse_mx):
            sparse_mx = sparse_mx.tocoo()
        coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
        values = sparse_mx.data
        shape = sparse_mx.shape
        return coords, values, shape

    def distribution_from_graph(self, adj, alpha, measure):
        # adj is a dense matrix
        if measure == 'degree':
            prob = torch.pow(adj.sum(0), alpha)
        elif measure == 'core':
            pass
        elif measure == 'uniform':
            prob = np.ones(adj.shape[0])
        else:
            raise ValueError('Undefined sampling method!')

        prob = prob / prob.sum()
        return prob.detach().cpu().numpy()

    def subgraph_from_graph(self, sp_adj, sample_nodes, is_sparse=True):
        sp_csr_adj = sp_adj.tocsr()
        sample_adj = sp_csr_adj[sample_nodes, :][:, sample_nodes].tocoo()
        sample_adj = SparseTensor.from_scipy(sample_adj).to_dense()

        return sample_adj

    def edges_split_from_graph(self, adj, is_sparse=True):
        # Function to build test set with 10% positive links
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
        # TODO: Clean up.

        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0

        adj_triu = sp.triu(adj)
        adj_tuple = self.sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        edges_all = self.sparse_to_tuple(adj)[0]
        num_test = int(np.floor(edges.shape[0] / 10.))
        num_val = int(np.floor(edges.shape[0] / 20.))

        all_edge_idx = list(range(edges.shape[0]))
        np.random.seed(1)  # add by wcj
        np.random.shuffle(all_edge_idx)
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges[test_edge_idx]
        val_edges = edges[val_edge_idx]
        train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        test_edges_false = []
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])

        assert ~ismember(test_edges_false, edges_all)
        assert ~ismember(val_edges_false, edges_all)
        assert ~ismember(val_edges, train_edges)
        assert ~ismember(test_edges, train_edges)
        assert ~ismember(val_edges, test_edges)

        data = np.ones(train_edges.shape[0])

        # Re-build adj matrix
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        adj_train = adj_train + adj_train.T

        # NOTE: these edge lists only contain single direction of edge!
        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

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
