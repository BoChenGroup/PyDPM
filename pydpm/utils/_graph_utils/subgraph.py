# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import numpy as np
import networkx as nx
import scipy.sparse as sp
from torch_sparse import SparseTensor
from torch import tensor
import torch
import copy


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
    edge_index[0] = tensor(adj.row, dtype=torch.long, device=device)
    edge_index[1] = tensor(adj.col, dtype=torch.long, device=device)

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


def get_distribution(adj, alpha, measure):
    # adj is a dense matrix
    if measure == 'degree':
        prob = torch.pow(adj.sum(0), alpha)
    elif measure == 'core':
        pass
    elif measure == 'uniform':
        prob = np.ones(adj.shape[0])
    else:
        raise ValueError('Undefined sampling method!')

    prob = prob/prob.sum()
    return prob.detach().cpu().numpy()

def sample_subgraph(sp_adj, sample_nodes):
    sp_csr_adj = sp_adj.tocsr()
    sample_adj = sp_csr_adj[sample_nodes, :][:, sample_nodes].tocoo()
    sample_adj = SparseTensor.from_scipy(sample_adj).to_dense()

    return sample_adj








