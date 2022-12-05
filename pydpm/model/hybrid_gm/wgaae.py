# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import copy
import warnings
import numpy as np
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor

from ...dataloader.graph_data import Graph_Processer
from ..basic_model import Basic_Model
from ...sampler import Basic_Sampler
from ...utils import *

import warnings

warnings.filterwarnings("ignore")

class WGAAE(Basic_Model, nn.Module):
    def __init__(self, in_dim: int, out_dim: int, z_dims: list, hid_dims: list, num_heads: int, device='gpu'):
        '''
        '''
        super(WGAAE, self).__init__()
        setattr(self, '_model_name', 'WGAAE')

        self._model_setting.in_dim = in_dim
        self._model_setting.out_dim = out_dim
        self._model_setting.z_dims = z_dims
        self._model_setting.hid_dims = hid_dims
        self._model_setting.num_layers = len(self._model_setting.z_dims)
        self._model_setting.num_heads = num_heads
        self._model_setting.device = device

        self.real_min = torch.tensor(2.2e-10, device=self._model_setting.device)

        self.initial()

    def initial(self):

        self.graph_processer = Graph_Processer()

        _model_setting = self._model_setting
        self.wgaae_encoder = WGAAE_Encoder(_model_setting.in_dim, _model_setting.out_dim, _model_setting.z_dims, _model_setting.hid_dims, _model_setting.num_heads, _model_setting.device)
        self.wgaae_decoder = WGAAE_Decoder(_model_setting.in_dim, _model_setting.z_dims, _model_setting.device)
        self.global_params = self.whai_decoder.global_params
        self._hyper_params = self.whai_decoder._hyper_params

        self.u = []
        for layer_index in range(self._model_setting.num_layers):
            self.u.append(torch.nn.Parameter(torch.empty([1, 1])).to(self.device))
            nn.init.trunc_normal_(self.u[layer_index].data)


    def log_max(self, x: torch.Tensor):
        '''
        return log(x+eps)
        '''
        return torch.log(torch.max(x, self.real_min))

    def bern_possion_link(self, x):
        return 1.0 - torch.exp(-x)

    def inner_product(self, x, dropout=0):
        # default dropout = 0
        x = F.dropout(x, dropout, training=self.training)
        x_t = x.permute(1, 0)
        x = x @ x_t
        # out = x.reshape(-1)
        return x

    def KL_GamWei(self, Gam_shape: torch.Tensor, Gam_scale: torch.Tensor, Wei_shape: torch.Tensor, Wei_scale: torch.Tensor):
        '''
        Calculate the KL divergence between Gamma distribution and Weibull distribution
        '''
        eulergamma = torch.tensor(0.5772, dtype=torch.float32, device=self._model_setting.device)
        part1 = eulergamma * (1 - 1 / Wei_shape) + self.log_max(Wei_scale / Wei_shape) + 1 + Gam_shape * torch.log(Gam_scale)
        part2 = -torch.lgamma(Gam_shape) + (Gam_shape - 1) * (self.log_max(Wei_scale) - eulergamma / Wei_shape)
        part3 = -Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + 1 / Wei_shape))
        KL = part1 + part2 + part3

        return KL

    def loss(self, data, pred, phi, theta, k, l, is_sample=False, task='prediction'):
        # TODO mean or sum on reson_llh and graph_llh
        x = data.x
        theta_concat = None
        for layer_index in range(self._model_setting.num_layers):
            if layer_index == 0:
                theta_concat = self.u[layer_index] * theta[layer_index]
            else:
                theta_concat = torch.cat([theta_concat, self.u[layer_index] * theta[layer_index]], dim=0)

        # Reconstruction likelihood
        x_T = x.permute(1, 0)
        re_x = torch.matmul(torch.tensor(phi[0], dtype=torch.float32, device=self._model_setting.device), theta[0])
        recon_llh = -1.0 * torch.sum(x_T * self.log_max(re_x) - re_x - torch.lgamma(x_T + 1.0))

        # Graph likelihood
        norm = torch.tensor(self.adj_nodes * self.adj_nodes / ((self.adj_nodes * self.adj_nodes - self.adj_sum) * 2)).to(self.device)
        if is_sample:
            pass
            # TODO
            # prob = self.prob
            # sample_nodes = np.random.choice(self.adj_nodes, size=self.n_sample_nodes, replace=False, p=prob)
            # sample_adj = self.graph_processer.sample_subgraph(self.adj_coo, sample_nodes).to(self.device)
            # num_sampled = self.n_sample_nodes
            # sum_sampled = sample_adj.sum() + self.real_min
            # pos_weight = torch.tensor((num_sampled * num_sampled - sum_sampled) / sum_sampled).to(self.device)
            # sub_theta_concat = theta_concat.T[sample_nodes, :]
            # inner_product = self.InnerProductDecoder(sub_theta_concat, dropout=0.).to(self.device)
            # recon_graph = self.bern_possion_link(inner_product).to(self.device)
            # graph_llh = 0.01 * norm * F.binary_cross_entropy_with_logits(recon_graph, sample_adj, pos_weight=pos_weight, reduction='sum')
        else:
            pos_weight = torch.tensor((self.adj_nodes * self.adj_nodes - self.adj_sum) / self.adj_sum).to(self.device)
            innner_product = self.inner_product(theta_concat.T, dropout=0.).to(self.device)
            recon_graph = self.bern_possion_link(innner_product).to(self.device)
            graph_llh = 0.002 * norm * F.binary_cross_entropy_with_logits(recon_graph, self.adj, pos_weight=pos_weight, reduction='sum')

        # KL divergence
        # TODO: grad problem
        # kl_loss = -1.0 * self.z_dimsL_GamWei(self._gamma_prior, self._gamma_prior, k[-1], l[-1]).reshape(-1).mean()
        # for layer in range(self._model_setting.T - 1):
        #     kl_loss += -1.0 * self.z_dimsL_GamWei(torch.tensor(self.global_params.Phi[layer + 1], dtype=torch.float32).to(self.device) @ theta[layer + 1], self._gamma_prior, k[layer], l[layer]).reshape(-1).mean()

        Loss = recon_llh + graph_llh #+ 0 * kl_loss

        if task == 'classification':
            cls_loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])
            Loss += cls_loss
            return [Loss, cls_loss, recon_llh, graph_llh]

        return [Loss, recon_llh, graph_llh]


    def train_one_epoch(self, data, optim: Optimizer, epoch_index, args=None, is_train=True, is_subgraph=False):

        if epoch_index == 0:
            self.num_classes = data.class_num # load from dataset
            self.pred_layer = nn.Linear(self.in_hid_dims[-1] + self._model_setting.z_dims[-1], self.num_classes).to(self.device)  # prediction layer: batch_szie * cls_num

            # Prepare for graph
            self.adj_nodes = data.x.shape[0]
            self.sp_adj, self.adj = self.graph_processer.graph_from_edges(edge_index=data.edge_index, n_nodes=self.adj_nodes)
            self.adj_coo = self.sp_adj.to_scipy(layout='coo')
            self.adj_sum = self.adj.sum()
            self.alpha = 2.0
            self.measure = 'degree'
            self.prob = self.graph_processer.distribution_from_graph(self.adj, self.alpha, self.measure)

        if is_subgraph:
            pass
        else:
            self.train_full_graph(data, optim, epoch_index, args, is_train=is_train)

    def train_full_graph(self, data, optim: Optimizer, epoch_index, args=None, is_train=True):
        if is_train:
            self.train()
        else:
            self.eval()

        theta, k, l = self.wgaae_encoder(data.x, data.edge_index, is_train=True)
        pred = theta[-1].permute(1, 0),
        # pred: [N, cls], theta: [K, N]
        loss, recon_llh, graph_llh = self.loss(data, pred, theta, k, l, args.is_sample, task=args.task)

        if is_train:
            self.global_params.Phi = self.wgaae_decoder.update_phi(data.x, theta)

        optim.zero_grad()
        loss.backward()
        optim.step()

        for layer_index in range(self._model_setting.num_layers):
            theta[layer_index] = torch.tensor(theta[layer_index], dtype=torch.float, device=self.device)

        return [pred, theta], [loss, recon_llh, graph_llh]

    # def train_sub_graph(self, model_opt, dataloader):
    #     # TODO
    #     pass
    #
    # def test_full_graph(self, data):
    #     self.eval()
    #     with torch.no_grad():
    #         [pred, theta, _, _] = self.forward(data.x, data.edge_index, is_train=False)
    #
    #     return [pred, theta]
    #
    # def test_sub_grpah(self, dataloader):
    #     # TODO
    #     pass
    #
    # def load(self, checkpoint_path: str, directory_path: str):
    #     '''
    #     Load the model parameters from the checkpoint and the specified directory.
    #     Inputs:
    #         model_path : [str] the path to load the model.
    #
    #     '''
    #     assert os.path.exists(checkpoint_path), 'Path Error: can not find the path to load the checkpoint'
    #     assert os.path.exists(directory_path), 'Path Error: can not find the path to load the directory'
    #
    #     # load parameters of neural network
    #     checkpoint = torch.load(checkpoint_path)
    #     self.load_state_dict(checkpoint['state_dict'])
    #
    #     # load parameters of basic model
    #     model = np.load(directory_path, allow_pickle=True).item()
    #     for params in ['global_params', 'local_params', '_model_setting', '_hyper_params']:
    #         if params in model:
    #             setattr(self, params, model[params])
    #
    # def save(self, model_path: str = '../save_models'):
    #     '''
    #     Save the model to the checkpoint the specified directory.
    #     Inputs:
    #         model_path : [str] the path to save the model, default '../save_models/WGAAE.npy' and '../save_models/WGAAE.pth'
    #     '''
    #     # create the trained model path
    #     if not os.path.isdir(model_path):
    #         os.mkdir(model_path)
    #
    #     # save parameters of neural network
    #     torch.save({'state_dict': self.state_dict()}, model_path + '/' + self._model_name + '.pth')
    #     print('parameters of neural network have been saved by ' + model_path + '/' + self._model_name + '.pth')
    #
    #     # save parameters of basic model
    #     model = {}
    #     for params in ['global_params', 'local_params', '_model_setting', '_hyper_params']:
    #         if params in dir(self):
    #             model[params] = getattr(self, params)
    #
    #     np.save(model_path + '/' + self._model_name + '.npy', model)
    #     print('parameters of basic model have been saved by ' + model_path + '/' + self._model_name + '.npy')

class WGAAE_Encoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, z_dims: list, hid_dims: list, num_heads: int, device='gpu'):
        super(WGAAE_Encoder, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.z_dims = z_dims
        self.hid_dims = hid_dims
        self.num_layers = len(hid_dims)
        self.num_heads = num_heads
        self.device = device

        self.real_min = torch.tensor(2.2e-10, dtype=torch.float, device=self.device)
        self.theta_max = torch.tensor(1000.0, dtype=torch.float, device=self.device)
        self.wei_shape_min = torch.tensor(1e-1, dtype=torch.float, device=self.device)
        self.wei_shape_max = torch.tensor(1000.0, dtype=torch.float, device=self.device)

        # Network for WGAAE_encoder
        self.fc_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        self.h_encoders = nn.ModuleList()
        self.shape_encoders = nn.ModuleList()
        self.scale_encoders = nn.ModuleList()

        self.in_hid_dims = [self.in_dim] + self.hid_dims
        for layer in range(self.num_layers):
            # why?
            # if layer == self.num_layers - 1:
            #     if layer == 0:
            #         self.h_encoders.append(GATConv(self.in_hid_dims[layer], (self.in_hid_dims[layer] + self.in_hid_dims[layer + 1]) // 2, self.num_heads, dropout=0.6))
            #         self.h_encoders.append(GATConv((self.in_hid_dims[layer] + self.in_hid_dims[layer + 1]) // 2 * self.num_heads, self.in_hid_dims[layer + 1], heads=1, concat=False, dropout=0.6))
            #     else:
            #         self.h_encoders.append(GATConv(self.in_hid_dims[layer], self.in_hid_dims[layer + 1] // self.num_heads, heads=self.num_heads, dropout=0.6))

            self.h_encoders.append(GATConv(self.in_hid_dims[layer], self.in_hid_dims[layer + 1] // self.num_heads, heads=self.num_heads, dropout=0.6).to(self.device))
            self.shape_encoders.append((nn.Linear(self.z_dims[layer], self.z_dims[layer])).to(self.device))
            self.scale_encoders.append((nn.Linear(self.z_dims[layer], self.z_dims[layer])).to(self.device))

            self.skip_layers.append(nn.Linear(self.in_hid_dims[layer + 1], self.in_hid_dims[layer + 1]).to(self.device))
            self.norm_layers.append(nn.BatchNorm1d(self.in_hid_dims[layer + 1]).to(self.device))
            self.fc_layers.append(nn.Linear(self.in_hid_dims[layer + 1], self.z_dims[layer]).to(self.device))

    def log_max(self, x: torch.Tensor):
        '''
        return log(x+eps)
        '''
        return torch.log(torch.max(x, self.real_min))

    def encoder_left(self, x: torch.Tensor, edge_index: torch.Tensor, layer_index: int):
        '''
        Encoder for hidden layers
        Inputs:
            x         : Input of current layer
            edge_index: Input adj of current layer
            num_layer : Index of layers

        Outputs:
            The x produced by the encoder
        '''
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.h_encoders[layer_index](x, edge_index)
        x = x + self.skip_layers[layer_index](x)
        x = self.norm_layers[layer_index](x)

        return x

    def encoder_right(self, x: torch.tensor, layer_index: int, phi: torch.Tensor, theta: torch.Tensor):
        '''
        Encoder for parameters of Weibull distribution
        Inputs:
            x         : Input of current layer
            num_layer : Index of layers

        Outputs:
            k, l : The parameters of Weibull distribution produced by the encoder
        '''
        k = F.softplus(self.shape_encoders[layer_index](x))
        l = F.softplus(self.scale_encoders[layer_index](x))

        return k.permute(1, 0), l.permute(1, 0)


    def reparameterize(self, Wei_shape: torch.Tensor, Wei_scale: torch.Tensor, layer_index, num_samples=10):
        '''
        Reparameterization trick for Weibull distribution
        Inputs:
            Wei_shape : Shape-parameter in Weibull distribution
            Wei_scale : Scale-parameter in Weibull distribution
            num_layer : Index of layer to reparameterize on

        Outputs:
            theta : The latent matrix (The variables obey Weibull distribution with reparameterization trick)
        '''
        eps = torch.FloatTensor(num_samples, self.z_dims[layer_index], Wei_shape.shape[1]).uniform_(0.0, 1.0).to(self.device)
        Wei_shape = Wei_shape.unsqueeze(0).repeat(num_samples, 1, 1)
        Wei_scale = Wei_scale.unsqueeze(0).repeat(num_samples, 1, 1)
        theta = Wei_scale * torch.pow(- self.log_max(1 - eps), 1 / Wei_shape)
        theta = torch.clamp(theta.mean(0), self.real_min.item(), self.theta_max.item())

        return theta  ## v*n

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, is_train=True):
        '''

        '''
        h = []
        for layer_index in range(self.num_layers):
            if layer_index == 0:
                h_ = self.encoder_left(x, edge_index, layer_index)
                h.append(F.softplus(self.fc_layers[layer_index](h_)))
            else:
                h.append(self.encoder_left(h_, edge_index, layer_index))


        k = [[] for _ in range(self.num_layers)]
        l = [[] for _ in range(self.num_layers)]
        theta = [[] for _ in range(self.num_layers)]
        for layer_index in range(self.num_layers - 1, -1, -1):
            k[layer_index], l[layer_index] = self.encoder_right(h[layer_index], layer_index, self.phi, theta)
            # no need to use .item()
            k[layer_index] = torch.clamp(k[layer_index], self.wei_shape_min, self.wei_shape_max) # max = 1 / 2.2e-10
            l[layer_index] = torch.clamp(l[layer_index], self.real_min)

            if is_train:
                l[layer_index] = l[layer_index] / torch.exp(torch.lgamma(1.0 + 1.0 / k[layer_index]))
                theta[layer_index] = self.reparameterize(k[layer_index], l[layer_index], layer_index)
            # why?
            # else:
            #     l[i] = l[i] / torch.exp(torch.lgamma(1.0 + 1.0 / k[i]))
            #     theta[i] = self.reparameterize(k[i], l[i], x.shape[0], i)
            #     theta[i] = torch.min(l[i], self._theta_max)

        return theta, k, l

class WGAAE_Decoder(Basic_Model):
    def __init__(self, in_dim: int, z_dims: list, device: str='gpu'):
        super(WGAAE_Decoder, self).__init__()

        self._model_setting.V = in_dim
        self._model_setting.K = z_dims
        self._model_setting.T = len(self._model_setting.K)
        self._model_setting.device = 'cpu' if device == 'cpu' else 'gpu'

        self.real_min = 2.2e-10

        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''
        self._sampler = Basic_Sampler(self._model_setting.device)

        self.initial()


    def inital(self):

        self.MBObserved = 0
        self.NDot = [0] * self._model_setting.T
        self.Xt_to_t1 = [0] * self._model_setting.T
        self.WSZS = [0] * self._model_setting.T
        self.EWSZS = [0] * self._model_setting.T

        # global parameters
        self.global_params.Phi = []
        for t in range(self._model_setting.T):
            if t == 0:
                self.global_params.Phi.append(0.2 + 0.8 * np.random.rand(self._model_setting.V, self._model_setting.K[t]))
            else:
                self.global_params.Phi.append(0.2 + 0.8 * np.random.rand(self._model_setting.K[t-1], self._model_setting.K[t]))
            self.global_params.Phi[t] = self.global_params.Phi[t] / np.maximum(self.real_min, self.global_params.Phi[t].sum(0))

    def ProjSimplexSpecial(self, Phi_tmp: np.ndarray, Phi_old: np.ndarray, epsilon):
        Phinew = Phi_tmp - (Phi_tmp.sum(0) - 1) * Phi_old
        # if np.where(Phinew[:, :] <= 0)[0].size > 0:
        #     Phinew = np.maximum(epsilon, Phinew)
        #     Phinew = Phinew / np.maximum(realmin, Phinew.sum(0))
        Phinew = np.maximum(epsilon, Phinew)
        Phinew = Phinew / np.maximum(realmin, Phinew.sum(0))
        return Phinew

    def update_phi(self, x_t: np.ndarray, theta: list, args):
        '''
        TLASGR-MCMC for updating Phi
        '''
        '''
        TLASGR-MCMC for updating Phi
        Inputs:
            x_t     : [ndarray] (batch_szie)*(V) input of data
            theta   : [Tensor] (K_t)*(batch_szie) factor score matrices at different layers
            args    : Hyper-parameters
        '''

        if self.MBObserved == 0:
            num_updates = args.MBratio * args.num_epochs
            self.ForgetRate = np.power((0 + np.linspace(1, num_updates, num_updates)), -0.9)
            epsit = np.power((20 + np.linspace(1, num_updates, num_updates)), -0.7)
            self.epsit = 1 * epsit / epsit[0]

        x_t = np.array(x_t, order='C').astype('float64')
        for t in range(self._model_setting.T):
            phi_t = np.array(self.global_params.Phi[t], order='C').astype('float64')
            theta_t = np.array(theta[t].detach().cpu().numpy(), order='C').astype('float64')
            if t == 0:
                self.Xt_to_t1[t], self.WSZS[t] = self._sampler.multi_aug(x_t, phi_t, theta_t)
            else:
                self.Xt_to_t1[t], self.WSZS[t] = self._sampler.crt_multi_aug(self.Xt_to_t1[t - 1], phi_t, theta_t)

            self.EWSZS[t] = args.MBratio * self.WSZS[t]
            if (self.MBObserved == 0):
                self.NDot[t] = self.EWSZS[t].sum(0)
            else:
                self.NDot[t] = (1 - self.ForgetRate[self.MBObserved]) * self.NDot[t] + self.ForgetRate[self.MBObserved] * self.EWSZS[t].sum(0)
            tmp = self.EWSZS[t] + 0.1
            tmp = (1 / np.maximum(self.NDot[t], self.real_min)) * (tmp - tmp.sum(0) * phi_t)
            tmp1 = (2 / np.maximum(self.NDot[t], self.real_min)) * phi_t

            tmp = phi_t + self.epsit[self.MBObserved] * tmp + np.sqrt(self.epsit[self.MBObserved] * tmp1) * np.random.randn(phi_t.shape[0], phi_t.shape[1])
            self.global_params.Phi[t] = self.ProjSimplexSpecial(tmp, phi_t, self.real_min)

        self.MBObserved += 1

        return self.global_params.Phi

    # def Sample_Phi(self, WSZS_t, Eta_t):  # (array, scalar)
    #     Kt = WSZS_t.shape[0]
    #     Kt1 = WSZS_t.shape[1]
    #     Phi_t_shape = WSZS_t + Eta_t
    #     Phi_t = np.zeros([Kt, Kt1])
    #     Phi_t = np.random.gamma(Phi_t_shape, 1)
    #     #    for kt in range(Kt):
    #     #        for kt1 in range(Kt1):
    #     #            Phi_t[kt,kt1] = a.gamma(Phi_t_shape[kt,kt1],1)
    #     Phi_t = Phi_t / Phi_t.sum(0)
    #     return Phi_t

    # def updatePhi_full(self, train_data, Theta):
    #     for t in range(self._model_setting.T):
    #         Theta[t] = np.array(Theta[t].cpu().detach().numpy(), order='C').astype('float64')
    #         if t == 0:
    #             Xt = np.array(train_data.cpu().detach().numpy(), order='C')
    #             self.Xt_to_t1[t], self.WSZS[t] = self.sampler.multi_aug(Xt.astype('double'),
    #                                                                              self.global_params.Phi[t],
    #                                                                              Theta[t])
    #         else:
    #             self.Xt_to_t1[t], self.WSZS[t] = self.sampler.crt_multi_aug(
    #                 self.Xt_to_t1[t - 1].astype('double'),
    #                 self.Phi[t],
    #                 self.Theta[t])
    #         self.global_params.Phi[t][:, :] = self.Sample_Phi(self.WSZS[t], self.Eta[t])
