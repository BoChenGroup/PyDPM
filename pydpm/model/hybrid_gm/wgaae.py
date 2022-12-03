# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor

import pickle
import os
import copy
from tqdm import tqdm
from ..basic_model import Basic_Model
from ...sampler import Basic_Sampler
from ...utils import *
from ...utils._graph_utils.subgraph import *
import warnings
warnings.filterwarnings("ignore")

class WGAAE(Basic_Model, nn.Module):
    def __init__(self, K:list, H:list, V, head_num:int, out_dim:int, device='gpu'):
        super(WGAAE, self).__init__()
        setattr(self, '_model_name', 'WGAAE')

        self._model_setting.K = K
        self._model_setting.H = H
        self._model_setting.T = len(K)
        self._model_setting.V = V
        self.H_dim = [self._model_setting.V] + self._model_setting.H
        self.head_num = head_num
        self.out_dim = out_dim
        self.device = device
        self._model_setting.device = 'cpu' if device == 'cpu' else 'gpu'
        self._real_min = torch.tensor(2.2e-10, dtype=torch.float, device=device)
        self._gamma_prior = torch.tensor(1.0, dtype=torch.float, device=device)

        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''
        self._sampler = Basic_Sampler(self._model_setting.device)

        self.wgaae_encoder = WGAAE_Encoder(K, H, V, head_num, out_dim, device)
        self.wgaae_decoder = WGAAE_Decoder(len(K), self._sampler)

    def initial(self, data, cls, task, batch_size=128, n_epochs=100, MBratio=100):
        self._model_setting.n_epochs = n_epochs
        self._model_setting.batch_size = batch_size
        self.cls = cls
        self.MBratio = MBratio
        self._real_min_phi = 1e-30
        self.task = task
        self.pred_layer = nn.Linear(self.H_dim[-1] + self._model_setting.K[-1], self.cls).to(self.device) # prediction layer: batch_szie * cls_num

        # Prepare for graph
        self.adj_nodes = data.x.shape[0]
        self.n_sample_nodes = batch_size
        self.sp_adj, self.adj = graph_from_edges(edge_index=data.edge_index, n_nodes=self.adj_nodes)
        self.adj_coo = self.sp_adj.to_scipy(layout='coo')
        self.adj_sum = self.adj.sum()
        self.alpha = 2.0
        self.measure = 'degree'
        self.prob = get_distribution(self.adj, self.alpha, self.measure)

        self.u = []
        for layer in range(self._model_setting.T):
            self.u.append(torch.nn.Parameter(self.weight_init([1, 1])).to(self.device))
        # Prepare for updating Phi
        self.global_params.Phi = self.init_phi()

        self.wgaae_encoder.inital(self.global_params.Phi, batch_size)
        self.wgaae_decoder.inital(self.global_params.Phi, n_epochs, MBratio)
        self.dropout = torch.nn.Dropout(p=0.4)


    def log_max(self, x):
        '''
        return log(x+eps)
        '''
        return torch.log(torch.max(x, self._real_min.to(self.device)))


    def KL_GamWei(self, gam_shape, gam_scale, wei_shape, wei_scale):
        '''
        Calculate the KL divergence between Gamma distribution and Weibull distribution
        '''
        euler_mascheroni_c = torch.tensor(0.5772, dtype=torch.float32, device=self.device)
        t1 = torch.log(wei_shape) + torch.lgamma(gam_shape)
        t2 = - gam_shape * torch.log(wei_scale * gam_scale)
        t3 = euler_mascheroni_c * (gam_shape / wei_shape - 1) - 1
        t4 = gam_scale * wei_scale * torch.exp(torch.lgamma(1 + 1 / wei_shape))
        return (t1 + t2 + t3 + t4).sum(1).mean()

    def init_phi(self):
        '''
        Initialize the Phi randomly
        '''
        Phi = []
        for t in range(self._model_setting.T):  # 0:T-1
            # self.Eta.append((np.ones(self._model_setting.T) * 0.1)[t])
            if t == 0:
                Phi.append(0.2 + 0.8 * np.float32(np.random.rand(self._model_setting.V, self._model_setting.K[t])))
            else:
                Phi.append(0.2 + 0.8 * np.float32(np.random.rand(self._model_setting.K[t - 1], self._model_setting.K[t])))
            Phi[t] = Phi[t] / np.maximum(self._real_min.item(), Phi[t].sum(0))  # maximum every elements
        return Phi


    def InnerProductDecoder(self, x, dropout):
        # default dropout = 0
        x = F.dropout(x, dropout)
        x_t = x.permute(1, 0)
        x = x @ x_t
        # out = x.reshape(-1)
        return x

    def bern_possion_link(self, x):
        return 1.0 - torch.exp(-x)

    def weight_init(self, shape):
        w = torch.empty(shape)
        return nn.init.trunc_normal_(w, mean=0, std=0.01, a=-0.02, b=0.02)

    def bias_init(self, shape):
        return nn.init.trunc_normal_(shape, std=0.01)

    def compute_loss(self, data, pred, theta, k, l, is_sample=False, task='prediction'):
        # TODO mean or sum on reson_llh and graph_llh
        x = data.x
        theta_concat = None
        for layer in range(self._model_setting.T):
            if layer == 0:
                theta_concat = self.u[layer] * theta[layer]
            else:
                theta_concat = torch.cat([theta_concat, self.u[layer] * theta[layer]], 0)

        # Reconstruction likelihood
        x_T = x.permute(1, 0)
        recon_llh = -1.0 * torch.sum(x_T * self.log_max(torch.matmul(torch.tensor(self.global_params.Phi[0], dtype=torch.float32, device=self.device), theta[0]))\
                        - torch.matmul(torch.tensor(self.global_params.Phi[0], dtype=torch.float32, device=self.device), theta[0])\
                        - torch.lgamma(x_T + 1.0))

        # Graph likelihood
        norm = torch.tensor(self.adj_nodes * self.adj_nodes / ((self.adj_nodes * self.adj_nodes - self.adj_sum) * 2)).to(self.device)
        if is_sample:
            prob = self.prob
            sample_nodes = np.random.choice(self.adj_nodes, size=self.n_sample_nodes, replace=False, p=prob)
            sample_adj = sample_subgraph(self.adj_coo, sample_nodes).to(self.device)
            num_sampled = self.n_sample_nodes
            sum_sampled = sample_adj.sum() + self._real_min
            pos_weight = torch.tensor((num_sampled * num_sampled - sum_sampled) / sum_sampled).to(self.device)
            sub_theta_concat = theta_concat.T[sample_nodes, :]
            inner_product = self.InnerProductDecoder(sub_theta_concat, dropout=0.).to(self.device)
            recon_graph = self.bern_possion_link(inner_product).to(self.device)
            graph_llh = 0.01 * norm * F.binary_cross_entropy_with_logits(recon_graph, sample_adj, pos_weight=pos_weight, reduction='sum')
        else:
            pos_weight = torch.tensor((self.adj_nodes * self.adj_nodes - self.adj_sum) / self.adj_sum).to(self.device)
            innner_product = self.InnerProductDecoder(theta_concat.T, dropout=0.).to(self.device)
            recon_graph = self.bern_possion_link(innner_product).to(self.device)
            graph_llh = 0.002 * norm * F.binary_cross_entropy_with_logits(recon_graph, self.adj, pos_weight=pos_weight, reduction='sum')

        # KL divergence
        # TODO: grad problem
        # kl_loss = -1.0 * self.KL_GamWei(self._gamma_prior, self._gamma_prior, k[-1], l[-1]).reshape(-1).mean()
        # for layer in range(self._model_setting.T - 1):
        #     kl_loss += -1.0 * self.KL_GamWei(torch.tensor(self.global_params.Phi[layer + 1], dtype=torch.float32).to(self.device) @ theta[layer + 1], self._gamma_prior, k[layer], l[layer]).reshape(-1).mean()


        Loss = recon_llh + graph_llh #+ 0 * kl_loss

        if task == 'classification':
            cls_loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])
            Loss += cls_loss
            return [Loss, cls_loss, recon_llh, graph_llh]

        return [Loss, recon_llh, graph_llh]


    def forward(self, x, edge_index, is_train=False):
        theta, k, l = self.wgaae_encoder(x, edge_index, is_train)
        # pred = self.pred_layer(torch.cat([h[-1], theta[-1].permute(1, 0)], 1))
        pred = theta[-1].permute(1, 0)

        return [pred, theta, k, l]

    def train_full_graph(self, model_opt, data, is_sample, update_phi):
        self.train()

        model_opt.zero_grad()
        [pred, theta, k, l] = self.forward(data.x, data.edge_index, is_train=True)
        # pred: [N, cls], theta: [K, N]
        Loss = self.compute_loss(data, pred, theta, k, l, is_sample, task=self.task)
        [loss, recon_llh, graph_llh] = Loss

        if update_phi:
            self.global_params.Phi = self.wgaae_decoder(data.x, theta)
            self.wgaae_encoder.phi = self.global_params.Phi

        loss.backward()
        model_opt.step()

        for layer in range(self._model_setting.T):
            theta[layer] = tensor(theta[layer], dtype=torch.float, device=self.device)

        return [pred, theta], [loss, recon_llh, graph_llh]

    def train_sub_graph(self, model_opt, dataloader):
        # TODO
        pass


    def test_full_graph(self, data):
        self.eval()
        with torch.no_grad():
            [pred, theta, _, _] = self.forward(data.x, data.edge_index, is_train=False)

        return [pred, theta]

    def test_sub_grpah(self, dataloader):
        # TODO
        pass

    def load(self, checkpoint_path: str, directory_path: str):
        '''
        Load the model parameters from the checkpoint and the specified directory.
        Inputs:
            model_path : [str] the path to load the model.

        '''
        assert os.path.exists(checkpoint_path), 'Path Error: can not find the path to load the checkpoint'
        assert os.path.exists(directory_path), 'Path Error: can not find the path to load the directory'

        # load parameters of neural network
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['state_dict'])

        # load parameters of basic model
        model = np.load(directory_path, allow_pickle=True).item()
        for params in ['global_params', 'local_params', '_model_setting', '_hyper_params']:
            if params in model:
                setattr(self, params, model[params])

    def save(self, model_path: str = '../save_models'):
        '''
        Save the model to the checkpoint the specified directory.
        Inputs:
            model_path : [str] the path to save the model, default '../save_models/WGAAE.npy' and '../save_models/WGAAE.pth'
        '''
        # create the trained model path
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        # save parameters of neural network
        torch.save({'state_dict': self.state_dict()}, model_path + '/' + self._model_name + '.pth')
        print('parameters of neural network have been saved by ' + model_path + '/' + self._model_name + '.pth')

        # save parameters of basic model
        model = {}
        for params in ['global_params', 'local_params', '_model_setting', '_hyper_params']:
            if params in dir(self):
                model[params] = getattr(self, params)

        np.save(model_path + '/' + self._model_name + '.npy', model)
        print('parameters of basic model have been saved by ' + model_path + '/' + self._model_name + '.npy')

class WGAAE_Encoder(nn.Module):
    def __init__(self, K, H, V, head_num, out_dim, device):
        super(WGAAE_Encoder, self).__init__()

        self.K = K
        self.H = H
        self.T = len(K)
        self.V = V
        self.H_dim = [self.V] + self.H
        self.head_num = head_num
        self.out_dim = out_dim
        self.device = device

        self._real_min = torch.tensor(2.2e-10, dtype=torch.float, device=device)
        self._theta_max = torch.tensor(1000.0, dtype=torch.float, device=device)
        self._wei_shape_min = torch.tensor(1e-1, dtype=torch.float, device=device)
        self._wei_shape_max = torch.tensor(1000.0, dtype=torch.float, device=device)

        # Network for WGAAE_encoder
        self.h_encoder = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.shape_encoder = nn.ModuleList()
        self.scale_encoder = nn.ModuleList()
        self.Mlp = nn.ModuleList()

        for layer in range(self.T):
            if layer == self.T - 1:
                if layer == 0:
                    self.h_encoder.append(GATConv(self.H_dim[layer], (self.H_dim[layer] + self.H_dim[layer + 1]) // 2, self.head_num, dropout=0.6))
                    self.h_encoder.append(GATConv((self.H_dim[layer] + self.H_dim[layer + 1]) // 2 * self.head_num, self.H_dim[layer + 1], heads=1, concat=False, dropout=0.6))
                else:
                    self.h_encoder.append(GATConv(self.H_dim[layer], self.H_dim[layer + 1] // self.head_num, heads=self.head_num, dropout=0.6))
                self.skips.append(nn.Linear(self.H_dim[layer + 1], self.H_dim[layer + 1]))
                self.norms.append(nn.BatchNorm1d(self.H_dim[layer + 1]))
                self.Mlp.append(nn.Linear(self.H_dim[layer + 1], self.K[layer]).to(self.device))
                self.shape_encoder.append(nn.Linear(self.K[layer], self.K[layer]).to(self.device))
                self.scale_encoder.append(nn.Linear(self.K[layer], self.K[layer]).to(self.device))
            elif layer == 0:
                self.h_encoder.append(GATConv(self.H_dim[layer], self.H_dim[layer + 1] // self.head_num, heads=self.head_num, dropout=0.6))
                self.skips.append(nn.Linear(self.H_dim[layer + 1], self.H_dim[layer + 1]))
                self.norms.append(nn.BatchNorm1d(self.H_dim[layer + 1]))
                self.Mlp.append(nn.Linear(self.H_dim[layer + 1], self.K[layer]).to(self.device))
                self.shape_encoder.append((nn.Linear(self.K[layer], self.K[layer])).to(self.device))
                self.scale_encoder.append((nn.Linear(self.K[layer], self.K[layer])).to(self.device))
            else:
                self.h_encoder.append(GATConv(self.H_dim[layer], self.H_dim[layer + 1] // self.head_num, heads=self.head_num, dropout=0.6))
                self.skips.append(nn.Linear(self.H_dim[layer + 1], self.H_dim[layer + 1]))
                self.norms.append(nn.BatchNorm1d(self.H_dim[layer + 1]))
                self.Mlp.append(nn.Linear(self.H_dim[layer + 1], self.K[layer]).to(self.device))
                self.shape_encoder.append((nn.Linear(self.K[layer], self.K[layer])).to(self.device))
                self.scale_encoder.append((nn.Linear(self.K[layer], self.K[layer])).to(self.device))

    def inital(self, phi, batch_size):
        self.batch_szie = batch_size
        self.phi = phi

    def log_max(self, x):
        '''
        return log(x+eps)
        '''
        return torch.log(torch.max(x, self._real_min.to(self.device)))

    def encoder_gat(self, x, edge_index, num_layer):
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
        # if num_layer == self._model_setting.T - 1:
        x = self.h_encoder[num_layer](x, edge_index)
        if self.T == 1:
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.h_encoder[num_layer + 1](x, edge_index)

        x = x + self.skips[num_layer](x)
        x = self.norms[num_layer](x)# self.norms[num_layer](x)

        # else:
        #     x = F.elu(self.h_encoder[num_layer](x, edge_index))
        return x

    def encoder_right(self, x, num_layer, phi, theta):
        '''
        Encoder for parameters of Weibull distribution
        Inputs:
            x         : Input of current layer
            num_layer : Index of layers

        Outputs:
            k, l : The parameters of Weibull distribution produced by the encoder
        '''
        k = F.softplus(self.shape_encoder[num_layer](x))
        l = F.softplus(self.scale_encoder[num_layer](x))

        return k.permute(1, 0), l.permute(1, 0)

    def reparameterize(self, Wei_shape, Wei_scale, batch_size, num_layer):
        '''
        Reparameterization trick for Weibull distribution
        Inputs:
            Wei_shape : Shape-parameter in Weibull distribution
            Wei_scale : Scale-parameter in Weibull distribution
            num_layer : Index of layer to reparameterize on

        Outputs:
            theta : The latent matrix (The variables obey Weibull distribution with reparameterization trick)
        '''
        sample_num = 10
        eps = torch.FloatTensor(sample_num, self.K[num_layer], batch_size).uniform_(0.0, 1.0).to(self.device)
        Wei_shape = Wei_shape.unsqueeze(0).repeat(sample_num, 1, 1)
        Wei_scale = Wei_scale.unsqueeze(0).repeat(sample_num, 1, 1)
        theta = Wei_scale * torch.pow(- self.log_max(1 - eps), 1 / Wei_shape)
        theta = torch.clamp(theta.mean(0), self._real_min.item(), self._theta_max.item())

        return theta  ## v*n

    def forward(self, x, edge_index, is_train=True):
        theta = [0] * self.T
        h = []
        for i in range(self.T):
            if i == 0:
                h.append(self.encoder_gat(x, edge_index, i))
            else:
                h.append(self.encoder_gat(h[-1], edge_index, i))

        for i in range(self.T):
            h[i] = F.softplus(self.Mlp[i](h[i]))

        k = [[] for _ in range(self.T)]
        l = [[] for _ in range(self.T)]
        for i in range(self.T - 1, -1, -1):
            k[i], l[i] = self.encoder_right(h[i], i, self.phi, theta)
            k[i] = torch.clamp(k[i], self._wei_shape_min.item(), self._wei_shape_max.item()) # max = 1 / 2.2e-10
            l[i] = torch.clamp(l[i], self._real_min.item())

            if is_train:
                l[i] = l[i] / torch.exp(torch.lgamma(1.0 + 1.0 / k[i]))
                theta[i] = self.reparameterize(k[i], l[i], x.shape[0], i)
            else:
                l[i] = l[i] / torch.exp(torch.lgamma(1.0 + 1.0 / k[i]))
                theta[i] = self.reparameterize(k[i], l[i], x.shape[0], i)
                # theta[i] = torch.min(l[i], self._theta_max)

        return theta, k, l

class WGAAE_Decoder(nn.Module):
    def __init__(self, T, sampler):
        super(WGAAE_Decoder, self).__init__()

        self._real_min = torch.tensor(1e-30)
        self._real_min_phi = 1e-30
        self.T = T
        self._sampler = sampler

        self.NDot = [0] * T
        self.Xt_to_t1 = [0] * T
        self.WSZS = [0] * T
        self.EWSZS = [0] * T
        self.Eta = []
        self.train_num = 0

        self.phi = []


    def inital(self, phi, n_epochs, MBratio):
        self.phi = phi
        self.MBratio = MBratio
        n_updates = MBratio * n_epochs
        epsit = np.power((20 + np.linspace(1, n_updates,n_updates)), -0.7)
        self._epsit = 1 * epsit / epsit[0]
        self._ForgetRate = np.power((0 + np.linspace(1, n_updates, n_updates)), -0.9)

    def reset_para(self, n_updates, MBratio):
        """
        Reset private parameters about updating Phi
        inputs:
            n_updates   : [int] Total counts for updating Phi
            batch_size  : [int] The batch_size for updating Phi
            MBratio     : [int] Length of dataloader for updating Phi in training stage
        """
        self.train_num = 0
        epsit = np.power((20 + np.linspace(1, n_updates, n_updates)), -0.7)
        self._epsit = 1 * epsit / epsit[0]
        self._ForgetRate = np.power((0 + np.linspace(1, n_updates, n_updates)), -0.9)
        self.MBratio = MBratio

    def ProjSimplexSpecial(self, Phi_tmp, Phi_old, epsilon):
        Phinew = Phi_tmp - (Phi_tmp.sum(0) - 1) * Phi_old
        if np.where(Phinew[:, :] <= 0)[0].size > 0:
            Phinew = np.maximum(epsilon, Phinew)
            Phinew = Phinew / np.maximum(realmin, Phinew.sum(0))
        Phinew = Phinew / np.maximum(realmin, Phinew.sum(0))
        return Phinew

    def updatePhi(self, Xt, Theta, MBratio, MBObserved):
        '''
        TLASGR-MCMC for updating Phi
        '''
        Xt = np.array(np.transpose(Xt.cpu().detach().numpy()), order='C').astype('double')
        for t in range(self.T):
            self.phi[t] = np.array(self.phi[t], order='C').astype('float64')
            Theta[t] = np.array(Theta[t].cpu().detach().numpy(), order='C').astype('float64')
            if t == 0:
                self.Xt_to_t1[t], self.WSZS[t] = self._sampler.multi_aug(Xt, self.phi[t], Theta[t])
            else:
                self.Xt_to_t1[t], self.WSZS[t] = self._sampler.crt_multi_aug(self.Xt_to_t1[t - 1], self.phi[t],
                                                                                  Theta[t])
            self.EWSZS[t] = MBratio * self.WSZS[t]
            if (MBObserved == 0):
                self.NDot[t] = self.EWSZS[t].sum(0)
            else:
                self.NDot[t] = (1 - self._ForgetRate[MBObserved]) * self.NDot[t] + self._ForgetRate[MBObserved] * \
                               self.EWSZS[t].sum(0)
            tmp = self.EWSZS[t] + 0.1
            tmp = (1 / np.maximum(self.NDot[t], self._real_min_phi)) * (tmp - tmp.sum(0) * self.phi[t])
            tmp1 = (2 / np.maximum(self.NDot[t], self._real_min_phi)) * self.phi[t]

            tmp = self.phi[t] + self._epsit[MBObserved] * tmp + np.sqrt(self._epsit[MBObserved] * tmp1) * np.random.randn(
                self.phi[t].shape[0], self.phi[t].shape[1])
            self.phi[t] = self.ProjSimplexSpecial(tmp, self.phi[t], 0)

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

    def forward(self, x, theta):
        self.updatePhi(x, theta, self.MBratio, self.train_num)
        self.train_num += 1
        return self.phi
