"""
===========================================
WHAI: WEIBULL HYBRID AUTOENCODING INFERENCE FOR DEEP TOPIC MODELING
Hao Zhang, Bo Chen， Dandan Guo and Mingyuan Zhou
Published as a conference paper at ICLR 2018

===========================================

"""

# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import warnings
import os
import copy
from tqdm import tqdm
from ._basic_model import Basic_Model
from .._sampler import Basic_Sampler
from .._utils import *
warnings.filterwarnings("ignore")

class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx, device):
        '''
        convolutional layer
        Inputs:
            nf: Size of dimension produced by the convolution
            rf: Size of the convolving kernel
            nx: Size of dimension in the input

        Attributes:
            w (Tensor): the learnable weights of the module of shape
            b (Tensor): the learnable bias of the module of shape
        '''
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf).to(device)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf).to(device))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        '''
        Input:
            x: Input of convolutional layer

        Outputs:
            x: The fresh x produced by the convolution

        '''
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x

    # def __repr__(self):


class WHAI(Basic_Model, nn.Module):
    def __init__(self, K: list, H:list, V:int, device='gpu'):
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

        self._model_setting.K = K
        self._model_setting.H = H
        self._model_setting.V = V
        self._model_setting.T = len(K)
        self.H_dim = [self._model_setting.V] + self._model_setting.H
        self._model_setting.device = 'cpu' if device == 'cpu' else 'gpu'
        self.data_device = device
        self._real_min = torch.tensor(1e-30)
        self.h_encoder = nn.ModuleList([Conv1D(self.H_dim[i + 1], 1, self.H_dim[i], device) for i in range(self._model_setting.T)])
        self.shape_encoder = nn.ModuleList([Conv1D(1, 1, in_dim, device) for in_dim in self._model_setting.H])
        self.scale_encoder = nn.ModuleList([Conv1D(k_dim, 1, h_dim, device) for k_dim, h_dim in zip(self._model_setting.K, self._model_setting.H)])

        assert self._model_setting.device in ['cpu',
                                              'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''

        self._sampler = Basic_Sampler(self._model_setting.device)


    def initial(self, voc=None, cls=None, batch_size=100, n_epochs=100, MBratio=100):
        '''
        Initial the parameters of WHAI with the settings about documents
        Inputs:
            voc         : [list] V list, vocabulary with length of V
            cls         : [int] Classes of documents
            batch_size  : [int] The batch_size for updating Phi and preparing dataset
            n_epochs    : [int] Number of epochs in training stage
            MBratio     : [int] Length of dataloader for updating Phi in training stage

        Attributes:
            @public:
                global_params.Phi           : [list] T (K_t-1)*(K_t) factor loading matrices at different layers
                train_num                   : [int] Current counts of updating Phi
                drop_out                    :  Probability of an element to be zeroed in neural network
                Ndot, xt_to_t1, WSZS, EWSZS : Intermediate variables parameters in updating Phi

            @private:
                _real_min_phi       : [float] scalar, the parameter to prevent overflow in updating Phi
                _ForgetRate, _epsit : Parameters in updating Phi


        '''
        self._model_setting.n_epochs = n_epochs
        self._model_setting.batch_size = batch_size
        self._model_setting.voc = voc
        self._model_setting.cls = cls
        self._model_setting.MBratio = MBratio
        self._real_min_phi = 1e-30


        self.NDot = [0] * self._model_setting.T
        self.Xt_to_t1 = [0] * self._model_setting.T
        self.WSZS = [0] * self._model_setting.T
        self.EWSZS = [0] * self._model_setting.T

        self.global_params.Phi = self.init_phi()

        n_updates = MBratio * self._model_setting.n_epochs
        self._ForgetRate = np.power((0 + np.linspace(1, n_updates, n_updates)), -0.9)

        epsit = np.power((20 + np.linspace(1, n_updates,
                                           n_updates)), -0.7)
        self._epsit = 1 * epsit / epsit[0]

        self.train_num = 0
        self.dropout = torch.nn.Dropout(p=0.4)



    def reset_para(self, n_updates, batch_size, MBratio):
        """
        Reset private parameters about updating Phi
        inputs:
            n_updates   : [int] Total counts for updating Phi
            batch_size  : [int] The batch_size for updating Phi
            MBratio     : [int] Length of dataloader for updating Phi in training stage
        """
        self.train_num = 0
        self._model_setting.batch_size = batch_size
        self._ForgetRate = np.power((0 + np.linspace(1, n_updates, n_updates)), -0.9)

        epsit = np.power((20 + np.linspace(1, n_updates,
                                           n_updates)), -0.7)
        self._epsit = 1 * epsit / epsit[0]
        self._model_setting.MBratio = MBratio

    def vision_phi(self, outpath='phi_output', top_n=25, topic_diversity=True):
        '''
        Visualization of Phi and getting diversity on each layers
        inputs:
            outpath         : [str] The path of visualization of Phi
            top_n           : [int] Number of words to display
            topic_diversity : [bool] Whether to get topic diversity

        If topic_diversity is True, this function will print the diversity of each layers.
        '''
        def get_diversity(topics):
            word = []
            for line in topics:
                word += line
            word_unique = np.unique(word)
            return len(word_unique) / len(word)

        if not os.path.exists(outpath):
            os.makedirs(outpath)
        phi = 1
        for num, phi_layer in enumerate(self.global_params.Phi):
            phi = np.dot(phi, phi_layer)
            phi_k = phi.shape[1]
            path = os.path.join(outpath, 'phi' + str(num) + '.txt')
            f = open(path, 'w')
            topic_word = []
            for each in range(phi_k):
                top_n_words = self.get_top_n(phi[:, each], top_n)
                topic_word.append(top_n_words.split()[:25])
                f.write(top_n_words)
                f.write('\n')
            f.close()
            if topic_diversity:
                td_value = get_diversity(topic_word)
            print('topic diversity at layer {}: {}'.format(num, td_value))

    def get_top_n(self, phi, top_n):
        '''
        Get top n words of each topic
        Inputs:
            phi   : The loading matrix
            top_n : Number of words to get

        Outputs:
            Top n words
        '''
        top_n_words = ''
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += self._model_setting.voc[index]
            top_n_words += ' '
        return top_n_words

    def log_max(self, x):
        '''
        return log(x+eps)
        '''
        return torch.log(torch.max(x, self._real_min.to(self.data_device)))

    def reparameterize(self, Wei_shape, Wei_scale, num_layer):
        '''
        Reparameterization trick for Weibull distribution
        Inputs:
            Wei_shape : Shape-parameter in Weibull distribution
            Wei_scale : Scale-parameter in Weibull distribution
            num_layer : Index of layer to reparameterize on

        Outputs:
            theta : The latent matrix (The variables obey Weibull distribution with reparameterization trick)
        '''
        eps = torch.cuda.FloatTensor(self._model_setting.K[num_layer], self._model_setting.batch_size).uniform_(0.2, 0.8)
        theta = Wei_scale * torch.pow(-self.log_max(1 - eps), 1 / Wei_shape)
        return theta  ## v*n

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape, Wei_scale):
        '''
        Calculate the KL divergence between Gamma distribution and Weibull distribution
        '''
        eulergamma = torch.tensor(0.5772, dtype=torch.float32)
        part1 = eulergamma.to(self.data_device) * (1 - 1 / Wei_shape) + self.log_max(
            Wei_scale / Wei_shape) + 1 + Gam_shape * torch.log(Gam_scale)
        part2 = -torch.lgamma(Gam_shape) + (Gam_shape - 1) * (self.log_max(Wei_scale) - eulergamma.to(self.data_device) / Wei_shape)
        part3 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + 1 / Wei_shape))
        KL = part1 + part2 + part3
        return KL

    def init_phi(self):
        '''
        Initialize the Phi randomly
        '''
        Phi = []
        for t in range(self._model_setting.T):  # 0:T-1
            if t == 0:
                Phi.append(0.2 + 0.8 * np.float32(np.random.rand(self._model_setting.V, self._model_setting.K[t])))
            else:
                Phi.append(0.2 + 0.8 * np.float32(np.random.rand(self._model_setting.K[t - 1], self._model_setting.K[t])))
            Phi[t] = Phi[t] / np.maximum(self._real_min, Phi[t].sum(0))  # maximum every elements
        return Phi

    def input_phi(self, theta):
        ## for phi NN update
        ## todo something
        return None

    def encoder_left(self, x, num_layer):
        '''
        Encoder for hidden layers
        Inputs:
            x         : Input of current layer
            num_layer : Index of layers

        Outputs:
            The x produced by the encoder
        '''
        if num_layer == 0:
            x = torch.nn.functional.softplus(self.h_encoder[num_layer](self.log_max(1 + x)))
        else:
            x = torch.nn.functional.softplus(self.h_encoder[num_layer](x))
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
        k_tmp = torch.max(torch.exp(self.shape_encoder[num_layer](x)), self._real_min.to(self.data_device)).view(-1, 1)
        k_tmp = k_tmp.repeat(1, self._model_setting.K[num_layer])
        l = torch.max(torch.exp(self.scale_encoder[num_layer](x)), self._real_min.to(self.data_device))
        if num_layer != self._model_setting.T - 1:
            k = torch.max(k_tmp, self._real_min.to(self.data_device))
        else:
            k = torch.max(k_tmp, self._real_min.to(self.data_device))
        return k.permute(1, 0), l.permute(1, 0)

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
        for t in range(self._model_setting.T):
            self.global_params.Phi[t] = np.array(self.global_params.Phi[t], order='C').astype('float64')
            Theta[t] = np.array(Theta[t].cpu().detach().numpy(), order='C').astype('float64')
            if t == 0:
                self.Xt_to_t1[t], self.WSZS[t] = self._sampler.multi_aug(Xt, self.global_params.Phi[t], Theta[t])
            else:
                self.Xt_to_t1[t], self.WSZS[t] = self._sampler.crt_multi_aug(self.Xt_to_t1[t - 1], self.global_params.Phi[t],
                                                                                  Theta[t])
            self.EWSZS[t] = MBratio * self.WSZS[t]
            if (MBObserved == 0):
                self.NDot[t] = self.EWSZS[t].sum(0)
            else:
                self.NDot[t] = (1 - self._ForgetRate[MBObserved]) * self.NDot[t] + self._ForgetRate[MBObserved] * \
                               self.EWSZS[t].sum(0)
            tmp = self.EWSZS[t] + 0.1
            tmp = (1 / np.maximum(self.NDot[t], self._real_min_phi)) * (tmp - tmp.sum(0) * self.global_params.Phi[t])
            tmp1 = (2 / np.maximum(self.NDot[t], self._real_min_phi)) * self.global_params.Phi[t]

            tmp = self.global_params.Phi[t] + self._epsit[MBObserved] * tmp + np.sqrt(self._epsit[MBObserved] * tmp1) * np.random.randn(
                self.global_params.Phi[t].shape[0], self.global_params.Phi[t].shape[1])
            self.global_params.Phi[t] = self.ProjSimplexSpecial(tmp, self.global_params.Phi[t], 0)


    def compute_loss(self, x, theta, k, l):
        '''
        Compute loss with KL divergence and likelihood
        '''
        kl_loss = [0] * self._model_setting.T
        kl_weight = [0.05 for i in range(self._model_setting.T)]
        for i in range(self._model_setting.T):
            if i == self._model_setting.T - 1:
                kl_loss[i] = torch.sum(self.KL_GamWei(torch.tensor(1.0, dtype=torch.float32).to(self.data_device),
                                                      torch.tensor(1.0, dtype=torch.float32).to(self.data_device), k[i], l[i]))
            else:
                kl_loss[i] = torch.sum(
                    self.KL_GamWei(torch.matmul(torch.tensor(self.global_params.Phi[i + 1], dtype=torch.float32).to(self.data_device),
                                                theta[i + 1]), torch.tensor(1.0, dtype=torch.float32).to(self.data_device), k[i],
                                   l[i]))
        kl_part = [weight * kl for weight, kl in zip(kl_weight, kl_loss)]
        likelihood = torch.sum(
            x.permute(1, 0) * self.log_max(torch.matmul(torch.tensor(self.global_params.Phi[0], dtype=torch.float32).to(self.data_device),
                                                        theta[0])) - torch.matmul(
                torch.tensor(self.global_params.Phi[0], dtype=torch.float32).to(self.data_device), theta[0])
            - torch.lgamma(x.permute(1, 0) + 1))
        return -(torch.sum(torch.stack(kl_part)) + likelihood) / self._model_setting.batch_size, likelihood, torch.sum(
            torch.stack(kl_loss)) + likelihood

    def forward(self, x, is_train=False):
        '''
        Inputs:
            x        : The batch_size * V count data
            is_train : Whether to update Phi

        Outputs:
            theta : The batch_size * K latent matrix
            WHAI_loss, LikeliHood, LB : The loss for optimizing and collecting
        '''

        if is_train:
            MBObserved = self.train_num
        theta = [0] * self._model_setting.T
        h = []
        for i in range(self._model_setting.T):
            if i == 0:
                h.append(self.encoder_left(x, i))
            else:
                h.append(self.encoder_left(h[-1], i))
        k = [[] for _ in range(self._model_setting.T)]
        l = [[] for _ in range(self._model_setting.T)]

        for i in range(self._model_setting.T - 1, -1, -1):
            if i == self._model_setting.T - 1:
                k[i], l[i] = self.encoder_right(h[i], i, 0, 0)
            else:
                k[i], l[i] = self.encoder_right(h[i], i, self.global_params.Phi[i + 1], theta[i + 1])

            k[i] = torch.clamp(k[i], 0.1, 10.0)
            l[i] = torch.clamp(l[i], 1e-10)
            if is_train:
                l[i] = l[i] / torch.exp(torch.lgamma(1.0 + 1.0 / k[i]))
                theta[i] = self.reparameterize(k[i], l[i], i)
            else:
                theta[i] = torch.min(l[i], torch.tensor(1000.0))

        WHAI_LOSS, LikeliHood, LB = self.compute_loss(x, theta, k, l)
        if is_train:
            self.train_num += 1
            self.updatePhi(x, theta, self._model_setting.MBratio, MBObserved)

        return torch.tensor(theta[0], dtype=torch.float).to(self.data_device).permute(1, 0), WHAI_LOSS, LikeliHood, LB

    def train_one_epoch(self, model_opt, dataloader, epoch):
        '''
        Train for one epoch
        Inputs:
            model_opt  : Optimizer for model
            dataloader : Train data with form of dataloader
            epoch      : Current epoch on training stage

        Attributes:
            local_params.theta : Concatenation of theta with total data
            local_params.label : Concatenation of label with total data
        '''
        self.train()
        self.local_params.theta = None
        self.local_params.label = None
        loss_t, likelihood_t, lb_t = 0.0, 0.0, 0.0

        train_bar = tqdm(iterable=dataloader)
        for i, (train_data, tfidf, train_label) in enumerate(train_bar):
            train_bar.set_description(f'Epoch [{epoch}/{self._model_setting.n_epochs}]')
            train_bar.set_postfix(loss=loss_t/(i+1), likelihood=likelihood_t/(i+1), lb=lb_t/(i+1))

            if self.local_params.label is None:
                self.local_params.label = train_label.detach().numpy()
            else:
                self.local_params.label = np.concatenate((self.local_params.label, train_label.detach().numpy()))

            train_data = torch.tensor(train_data, dtype=torch.float).to(self.data_device)
            train_label = torch.tensor(train_label, dtype=torch.long).to(self.data_device)

            theta, loss, likelihood, lb = self.forward(train_data, True)
            loss.backward()
            model_opt.step()
            model_opt.zero_grad()

            loss_t += loss.cpu().detach().numpy()
            likelihood_t += likelihood.cpu().detach().numpy()
            lb_t += lb.cpu().detach().numpy()

            if self.local_params.theta is None:
                self.local_params.theta = theta.cpu().detach().numpy()
            else:
                self.local_params.theta = np.concatenate((self.local_params.theta, theta.cpu().detach().numpy()))

        return copy.deepcopy(self.local_params)

    def test_one_epoch(self, dataloader):
        '''
        Test for one epoch
        Inputs:
            dataloader : Test data with form of dataloader

        Outputs:
            local_theta : Concatenation of theta with total data
            local_label : Concatenation of label with total data
            full data   : Total data
        '''
        self.eval()
        full_data = None
        local_theta = None
        local_label = None
        loss_t, likelihood_t, lb_t = 0.0, 0.0, 0.0

        test_bar = tqdm(iterable=dataloader)
        with torch.no_grad():
            for i, (test_data, tfidf, test_label) in enumerate(test_bar):
                test_bar.set_description(f'Testing stage: ')
                test_bar.set_postfix(loss=loss_t / (i + 1), likelihood=likelihood_t / (i + 1), lb=lb_t / (i + 1))

                if local_label is None:
                    local_label = test_label.detach().numpy()
                    full_data = test_data
                else:
                    local_label = np.concatenate((local_label, test_label.detach().numpy()))
                    full_data = np.concatenate((full_data, test_data))

                test_data = torch.tensor(test_data, dtype=torch.float).to(self.data_device)
                test_label = torch.tensor(test_label, dtype=torch.long).to(self.data_device)

                theta, loss, likelihood, lb = self.forward(test_data, is_train=False)

                loss_t += loss.cpu().detach().numpy()
                likelihood_t = likelihood.cpu().detach().numpy()
                lb_t = lb.cpu().detach().numpy()

                if local_theta is None:
                    local_theta = theta.cpu().detach().numpy()
                else:
                    local_theta = np.concatenate((local_theta, theta.cpu().detach().numpy()))

        return local_theta, local_label, full_data


    def save_phi(self, phi_path, epoch):
        self.vision_phi(outpath=f'{phi_path}/{epoch}/')
        torch.save(self.state_dict(), phi_path + '/topic_pretrain_{}.pth'.format(str(self.train_num)))
        with open(phi_path + '/Phi_{}.pkl'.format(str(self.train_num)), 'wb') as f:
            pickle.dump(self.global_params.Phi, f)


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

    def save(self, model_path: str = './save_models'):
        '''
        Save the model to the checkpoint the specified directory.
        Inputs:
            model_path : [str] the path to save the model, default './save_models/WHAI.npy' and './save_models/WHAI.pth'
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

