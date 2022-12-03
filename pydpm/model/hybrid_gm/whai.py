"""
===========================================
WHAI: WEIBULL HYBRID AUTOENCODING INFERENCE FOR DEEP TOPIC MODELING
Hao Zhang, Bo Chen， Dandan Guo and Mingyuan Zhou
Published as a conference paper at ICLR 2018

===========================================

"""

# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import copy
import pickle
import warnings
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from ..basic_model import Basic_Model
from ...sampler import Basic_Sampler
from ...utils import *

warnings.filterwarnings("ignore")

class WHAI(Basic_Model, nn.Module):
    def __init__(self, in_dim: int, z_dims: list, hid_dims: list, device: str='cpu'):
        """
        The basic model for WHAI
        Inputs:
            K      : [list] Number of topics at different layers in WHAI;
            H      : [list] Size of dimension at different hidden layers in WHAI;
            V      : [int] Length of the vocabulary for convolutional layers in WHAI;
            device : [str] 'cpu' or 'gpu';

        Attributes:
            @public:
                global_params : [Params] the global parameters of the probabilistic model
                local_params  : [Params] the local parameters of the probabilistic model
                h_encoder     : [Modulelist] the convolutional layers for latent representation for WHAI
                shape_encoder : [Modulelist] the convolutional layers for shape-parameters in Weibull distribution
                scale_encoder : [Modulelist] the convolutional layers for scale-parameters in Weibull distribution

            @private:
                _model_setting   : [Params] the model settings of the probabilistic model
                _hyper_params    : [Params] the hyper parameters of the probabilistic model
                _model_setting.T : [int] the network depth
                _real_min        : [Tensor] the parameter to prevent overflow

        """
        super(WHAI, self).__init__()
        setattr(self, '_model_name', 'WHAI')
        self._model_setting.in_dim = in_dim
        self._model_setting.z_dims = z_dims
        self._model_setting.hid_dims = hid_dims
        self._model_setting.num_layers = len(self._model_setting.z_dims)
        self._model_setting.device = device

        self.real_min = torch.tensor(2.2e-10, device=self._model_setting.device)

        self.initial()

    def initial(self):
        _model_setting = self._model_setting
        self.whai_encoder = WHAI_Encoder(_model_setting.in_dim, _model_setting.z_dims, _model_setting.hid_dims, _model_setting.device)
        self.whai_decoder = WHAI_Decoder(_model_setting.in_dim, _model_setting.z_dims, _model_setting.device)
        self.global_params = self.whai_decoder.global_params
        self._hyper_params = self.whai_decoder._hyper_params

    def log_max(self, x: torch.Tensor):
        '''
        return log(x+eps)
        '''
        return torch.log(torch.max(x, self.real_min))

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

    def loss(self, x: torch.Tensor, phi: list, theta: list, k: list, l: list):
        '''
        Compute loss with KL divergence and likelihood
        '''
        kl_loss = [0 for _ in range(self._model_setting.num_layers)]
        kl_weight = [0.05 for _ in range(self._model_setting.num_layers)]
        kl_term = 0.0
        for t in range(self._model_setting.num_layers):
            if t == self._model_setting.num_layers - 1:
                kl_loss[t] = torch.sum(self.KL_GamWei(torch.tensor(1.0, dtype=torch.float32, device=self._model_setting.device),
                                                      torch.tensor(1.0, dtype=torch.float32, device=self._model_setting.device), k[t], l[t]))
            else:
                kl_loss[t] = torch.sum(
                    self.KL_GamWei(torch.matmul(torch.tensor(phi[t + 1], dtype=torch.float32, device=self._model_setting.device), theta[t + 1]),
                                   torch.tensor(1.0, dtype=torch.float32, device=self._model_setting.device), k[t], l[t]))
            kl_term += kl_loss[t] * kl_weight[t]

        x = x.permute([1, 0])
        re_x = torch.matmul(torch.tensor(phi[0], dtype=torch.float32, device=self._model_setting.device), theta[0])
        likelihood = torch.sum(x * self.log_max(re_x) - re_x - torch.lgamma(x + 1))

        return -(likelihood + kl_term) / x.shape[1], likelihood, likelihood + kl_term

    def train_one_epoch(self, dataloader: DataLoader, optim: Optimizer, epoch_index, args=None, is_train=True):
        '''
        Train for one epoch
        Inputs:
            model_opt  : Optimizer for model
            dataloader : Train dataset with form of dataloader
            epoch      : Current epoch on training stage

        Attributes:
            local_params.theta : Concatenation of theta with total dataset
            local_params.label : Concatenation of label with total dataset
        '''
        loss_t, likelihood_t, elbo_t = 0.0, 0.0, 0.0

        if is_train:
            self.train()
        else:
            self.eval()

        train_bar = tqdm(iterable=dataloader)
        for i, (train_data, train_label) in enumerate(train_bar):
            train_bar.set_description(f'Epoch [{epoch_index}/{args.num_epochs}]')
            train_bar.set_postfix(loss=loss_t/(i+1), likelihood=likelihood_t/(i+1), lb=elbo_t/(i+1))

            theta, k, l = self.whai_encoder(torch.tensor(train_data, dtype=torch.float32, device=self._model_setting.device), self.global_params.Phi, self.training)

            if is_train:
                args.MBratio = len(dataloader)
                args.MBObserved = args.MBratio * epoch_index + i
                self.global_params.Phi = self.whai_decoder.update_phi(train_data, theta, args)

            loss, likelihood, elbo = self.loss(torch.tensor(train_data, dtype=torch.float32, device=self._model_setting.device), self.global_params.Phi, theta, k, l)

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_t += loss.item()
            likelihood_t += likelihood.item()
            elbo_t += elbo.item()

            if not hasattr(self.local_params, 'Theta'):
                self.local_params.Theta = [0 for _ in range(self._model_setting.num_layers)]
                for t in range(self._model_setting.num_layers):
                    self.local_params.Theta[t] = theta[t].cpu().detach().numpy().T
            else:
                for t in range(self._model_setting.num_layers):
                    self.local_params.Theta[t] = np.concatenate((self.local_params.Theta[t], theta[t].cpu().detach().numpy().T))

        return copy.deepcopy(self.local_params)

    # ================================================================================================================== #
    # # TODO
    # def test_one_epoch(self, dataloader):
    #     '''
    #     Test for one epoch
    #     Inputs:
    #         dataloader : Test dataset with form of dataloader
    #
    #     Outputs:
    #         local_theta : Concatenation of theta with total dataset
    #         local_label : Concatenation of label with total dataset
    #         full dataset   : Total dataset
    #     '''
    #     self.eval()
    #     full_data = None
    #     local_theta = None
    #     local_label = None
    #     loss_t, likelihood_t, lb_t = 0.0, 0.0, 0.0
    #
    #     test_bar = tqdm(iterable=dataloader)
    #     with torch.no_grad():
    #         for i, (test_data, tfidf, test_label) in enumerate(test_bar):
    #             test_bar.set_description(f'Testing stage: ')
    #             test_bar.set_postfix(loss=loss_t / (i + 1), likelihood=likelihood_t / (i + 1), lb=lb_t / (i + 1))
    #
    #             if local_label is None:
    #                 local_label = test_label.detach().numpy()
    #                 full_data = test_data
    #             else:
    #                 local_label = np.concatenate((local_label, test_label.detach().numpy()))
    #                 full_data = np.concatenate((full_data, test_data))
    #
    #             test_data = torch.tensor(test_data, dtype=torch.float).to(self.device)
    #             test_label = torch.tensor(test_label, dtype=torch.long).to(self.device)
    #
    #             theta, k, l = self.forward(test_data, is_train=False)
    #             loss, likelihood, lb = self.compute_loss(test_data, theta, k, l)
    #             theta_1 = torch.tensor(theta[0], dtype=torch.float32).to(self.device).permute(1, 0)
    #
    #             loss_t += loss.cpu().detach().numpy()
    #             likelihood_t = likelihood.cpu().detach().numpy()
    #             lb_t = lb.cpu().detach().numpy()
    #
    #             if local_theta is None:
    #                 local_theta = theta_1.cpu().detach().numpy()
    #             else:
    #                 local_theta = np.concatenate((local_theta, theta_1.cpu().detach().numpy()))
    #
    #     return local_theta, local_label, full_data
    # def save_phi(self, phi_path, epoch):
    #     self.vision_phi(outpath=f'{phi_path}/{epoch}/')
    #     torch.save(self.state_dict(), phi_path + '/topic_pretrain_{}.pth'.format(str(self.whai_decoder.train_num)))
    #     with open(phi_path + '/Phi_{}.pkl'.format(str(self.whai_decoder.train_num)), 'wb') as f:
    #         pickle.dump(self.global_params.Phi, f)
    #
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
    #         model_path : [str] the path to save the model, default '../save_models/WHAI.npy' and '../save_models/WHAI.pth'
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
    #
    #
    # def vision_phi(self, outpath='phi_output', top_n=25, topic_diversity=True):
    #     '''
    #     Visualization of Phi and getting diversity on each layers
    #     inputs:
    #         outpath         : [str] The path of visualization of Phi
    #         top_n           : [int] Number of words to display
    #         topic_diversity : [bool] Whether to get topic diversity
    #
    #     If topic_diversity is True, this function will print the diversity of each layers.
    #     '''
    #     def get_diversity(topics):
    #         word = []
    #         for line in topics:
    #             word += line
    #         word_unique = np.unique(word)
    #         return len(word_unique) / len(word)
    #
    #     if not os.path.exists(outpath):
    #         os.makedirs(outpath)
    #     phi = 1
    #     for num, phi_layer in enumerate(self.global_params.Phi):
    #         phi = np.dot(phi, phi_layer)
    #         phi_k = phi.shape[1]
    #         path = os.path.join(outpath, 'phi' + str(num) + '.txt')
    #         f = open(path, 'w')
    #         topic_word = []
    #         for each in range(phi_k):
    #             top_n_words = self.get_top_n(phi[:, each], top_n)
    #             topic_word.append(top_n_words.split()[:25])
    #             f.write(top_n_words)
    #             f.write('\n')
    #         f.close()
    #         if topic_diversity:
    #             td_value = get_diversity(topic_word)
    #         print('topic diversity at layer {}: {}'.format(num, td_value))
    #
    # def get_top_n(self, phi, top_n):
    #     '''
    #     Get top n words of each topic
    #     Inputs:
    #         phi   : The loading matrix
    #         top_n : Number of words to get
    #
    #     Outputs:
    #         Top n words
    #     '''
    #     top_n_words = ''
    #     idx = np.argsort(-phi)
    #     for i in range(top_n):
    #         index = idx[i]
    #         top_n_words += self._model_setting.voc[index]
    #         top_n_words += ' '
    #     return top_n_words


class Conv1D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int=1, device: str='cpu'):
        '''
        convolutional layer
        Inputs:
        Attributes:
        '''
        super(Conv1D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.device = device

        if kernel_size == 1:  # faster 1x1 conv
            self.W = Parameter(torch.empty([in_dim, out_dim], device=self.device))
            self.b = Parameter(torch.zeros(out_dim, device=self.device))
            self._reset_parameters()

        else:  # was used to train LM
            raise NotImplementedError

    def _reset_parameters(self):
        if self.kernel_size == 1:
            nn.init.xavier_normal_(self.W.data)
            nn.init.uniform_(self.b.data)

    def forward(self, x):
        '''
        Input:
        Outputs:
        '''
        if self.kernel_size == 1:
            # x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.W)
            # size_out = x.size()[:-1] + (self.nf,)
            x = torch.mm(x.view(-1, x.size(-1)), self.W) + self.b
            x = x.view(x.size()[:-1] + (self.out_dim,))
        else:
            raise NotImplementedError
        return x

class WHAI_Encoder(nn.Module):
    def __init__(self, in_dim: int, z_dims: list, hid_dims: list, device: str = 'cpu'):
        '''
        Inputs:
        Outputs:
        '''
        super(WHAI_Encoder, self).__init__()

        self.in_dim = in_dim
        self.z_dims = z_dims
        self.hid_dims = hid_dims
        self.num_layers = len(z_dims)
        self.device = device
        self.real_min = torch.tensor(2.2e-10, device=self.device)

        self.h_encoders = nn.ModuleList([Conv1D(in_dim, out_dim, 1, self.device) for in_dim, out_dim in zip([self.in_dim]+self.hid_dims[:-1], self.hid_dims)])
        self.shape_encoders = nn.ModuleList([Conv1D(h_dim, 1, 1, self.device) for h_dim in self.hid_dims])
        self.scale_encoders = nn.ModuleList([Conv1D(h_dim, z_dim, 1, self.device) for z_dim, h_dim in zip(self.z_dims, self.hid_dims)])

    def log_max(self, x: torch.Tensor):
        '''
        return log(x+eps)
        Inputs:
        Outputs:
        '''
        return torch.log(torch.max(x, self.real_min))

    def encoder_left(self, x: torch.Tensor, layer_index: int):
        '''
        Encoder for hidden layers
        Inputs:
        Outputs:
        '''
        if layer_index == 0:
            x = F.softplus(self.h_encoders[layer_index](self.log_max(1 + x)))
        else:
            x = F.softplus(self.h_encoders[layer_index](x))
        return x

    def encoder_right(self, x: torch.Tensor, layer_index: int, phi: np.ndarray=None, theta: torch.Tensor=None):
        '''
        Encoder for parameters of Weibull distribution
        Inputs:
        Outputs:
        '''
        k_tmp = torch.max(torch.exp(self.shape_encoders[layer_index](x)), self.real_min).view(-1, 1)
        k_tmp = k_tmp.repeat(1, self.z_dims[layer_index])
        l = torch.max(torch.exp(self.scale_encoders[layer_index](x)), self.real_min)
        if layer_index != self.num_layers - 1:
            # k = torch.max(k_tmp + torch.matmul(torch.tensor(phi, dtype=torch.float32, device=self.device), theta).permute(1, 0), self._real_min).to(self.device)
            k = torch.max(k_tmp, self.real_min)
        else:
            k = torch.max(k_tmp, self.real_min)
        return k.permute(1, 0), l.permute(1, 0)

    def reparameterize(self, Wei_shape: torch.Tensor, Wei_scale: torch.Tensor, layer_index: int):
        '''
        Reparameterization trick for Weibull distribution
        Inputs:
        Outputs:
        '''
        eps = torch.FloatTensor(self.z_dims[layer_index], Wei_shape.shape[1]).uniform_(0.2, 0.8).to(self.device)
        theta = Wei_scale * torch.pow(-self.log_max(1 - eps), 1 / Wei_shape)
        return theta

    def forward(self, x: torch.Tensor, phi: list=[], is_train=True):
        '''
        Inputs:
        Outputs:
        '''
        assert len(phi) == self.num_layers, Warning('The length of phi list should be equal to the number of hidden layers')

        if is_train:
            self.train()
        else:
            self.eval()

        # upward
        h = []
        for layer_index in range(self.num_layers):
            if layer_index == 0:
                h.append(self.encoder_left(x, layer_index))
            else:
                h.append(self.encoder_left(h[-1], layer_index))

        # downward
        k = [[] for _ in range(self.num_layers)]
        l = [[] for _ in range(self.num_layers)]
        theta = [[] for _ in range(self.num_layers)]
        for layer_index in range(self.num_layers - 1, -1, -1):
            if layer_index == self.num_layers - 1:
                k[layer_index], l[layer_index] = self.encoder_right(h[layer_index], layer_index, 0, 0)
            else:
                k[layer_index], l[layer_index] = self.encoder_right(h[layer_index], layer_index, phi[layer_index+1], theta[layer_index+1])

            k[layer_index] = torch.clamp(k[layer_index], 0.1, 10.0)
            l[layer_index] = torch.clamp(l[layer_index], 1e-10)

            if is_train:
                l[layer_index] = l[layer_index] / torch.exp(torch.lgamma(1.0 + 1.0 / k[layer_index]))
                theta[layer_index] = self.reparameterize(k[layer_index], l[layer_index], layer_index)
            else:
                theta[layer_index] = torch.min(l[layer_index], torch.tensor(1000.0))

        return theta, k, l


class WHAI_Decoder(Basic_Model):
    def __init__(self, in_dim: int, z_dims: list, device: str='gpu'):
        '''
            @public:
                T                           : Number of layers
                sample                      : Sampler used for updating phi
                phi           : [list] T (K_t-1)*(K_t) factor loading matrices at different layers
                train_num                   : [int] Current counts of updating Phi
                drop_out                    :  Probability of an element to be zeroed in neural network
                Ndot, xt_to_t1, WSZS, EWSZS : Intermediate variables parameters in updating Phi

            @private:
                _real_min_phi       : [float] scalar, the parameter to prevent overflow in updating Phi
                _ForgetRate, _epsit : Parameters in updating Phi
        '''
        super(WHAI_Decoder, self).__init__()
        setattr(self, '_model_name', 'WHAI_Decoder')

        self._model_setting.V = in_dim
        self._model_setting.K = z_dims
        self._model_setting.T = len(self._model_setting.K)
        self._model_setting.device = 'cpu' if device == 'cpu' else 'gpu'

        self.real_min = 2.2e-10

        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''
        self._sampler = Basic_Sampler(self._model_setting.device)

        self.initial()

    def initial(self):

        # hyper parameters



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
        if np.where(Phinew[:, :] <= 0)[0].size > 0:
            Phinew = np.maximum(epsilon, Phinew)
            Phinew = Phinew / np.maximum(realmin, Phinew.sum(0))
        Phinew = Phinew / np.maximum(realmin, Phinew.sum(0))
        return Phinew

    def update_phi(self, x_t: np.ndarray, theta: list, args):
        '''
        TLASGR-MCMC for updating Phi
        '''

        NDot = [0] * self._model_setting.T
        Xt_to_t1 = [0] * self._model_setting.T
        WSZS = [0] * self._model_setting.T
        EWSZS = [0] * self._model_setting.T

        num_updates = args.MBratio * args.num_epochs
        ForgetRate = np.power((0 + np.linspace(1, num_updates, num_updates)), -0.9)
        epsit = np.power((20 + np.linspace(1, num_updates, num_updates)), -0.7)
        epsit = 1 * epsit / epsit[0]

        x_t = np.array(x_t, order='C').astype('float64')
        for t in range(self._model_setting.T):
            phi_t = np.array(self.global_params.Phi[t], order='C').astype('float64')
            theta_t = np.array(theta[t].detach().cpu().numpy(), order='C').astype('float64')
            if t == 0:
                Xt_to_t1[t], WSZS[t] = self._sampler.multi_aug(x_t, phi_t, theta_t)
            else:
                Xt_to_t1[t], WSZS[t] = self._sampler.crt_multi_aug(Xt_to_t1[t - 1], phi_t, theta_t)

            EWSZS[t] = args.MBratio * WSZS[t]
            if (args.MBObserved == 0):
                NDot[t] = EWSZS[t].sum(0)
            else:
                NDot[t] = (1 - ForgetRate[args.MBObserved]) * NDot[t] + ForgetRate[args.MBObserved] * EWSZS[t].sum(0)
            tmp = EWSZS[t] + 0.1
            tmp = (1 / np.maximum(NDot[t], self.real_min)) * (tmp - tmp.sum(0) * phi_t)
            tmp1 = (2 / np.maximum(NDot[t], self.real_min)) * phi_t

            tmp = phi_t + epsit[args.MBObserved] * tmp + np.sqrt(epsit[args.MBObserved] * tmp1) * np.random.randn(phi_t.shape[0], phi_t.shape[1])
            self.global_params.Phi[t] = self.ProjSimplexSpecial(tmp, phi_t, 0)

        return self.global_params.Phi