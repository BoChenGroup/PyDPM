"""
===========================================
Sawtooth Factorial Topic Embeddings Guided Gamma Belief Network
Zhibin Duan, Dongsheng Wang, Bo Chen, Chaojie Wang, Wenchao Chen, Yewen Li, Jie Ren and Mingyuan Zhou
Published as a conference paper at ICML 2021

===========================================

"""

# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import os

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "tanh":
        return nn.Tanh()
    else:
        raise RuntimeError("activation should be relu/tanh/softplus, not {}".format(activation))


class ResBlock(nn.Module):
    """Simple MLP block with residual connection.

    Args:
        in_features: the feature dimension of each output sample.
        out_features: the feature dimension of each output sample.
        activation: the activation function of intermediate layer, relu or gelu.
    """

    def __init__(self, in_features, out_features, activation="relu"):
        super(ResBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)

        self.bn = nn.BatchNorm1d(out_features)
        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        if self.in_features == self.out_features:
            out = self.fc2(self.activation(self.fc1(x)))
            return self.activation(self.bn(x + out))
        else:
            x = self.fc1(x)
            out = self.fc2(self.activation(x))
            return self.activation(self.bn(x + out))


class SawETM(nn.Module):
    """Simple implementation of the <<Sawtooth Factorial Topic Embeddings Guided Gamma Belief Network>>

    Args
        args: the set of arguments used to characterize the hierarchical neural topic model.
        device: the physical hardware that the model is trained on.
        pretrained_embeddings: if not None, initialize each word embedding in the vocabulary with pretrained Glove embeddings.
    """

    def __init__(self, embed_size: int, vocab_size: int, num_topics_list: list, num_hiddens_list: list, word_embeddings=None, device: str='cpu'):
        """
        The basic model for SawETM
        Inputs:
            in_dim : [int] dimension of input
            encoder_hid_dims : [list] list of dimension in encoder
            decoder_hid_dims : [list] list of dimension in decoder
            z_dim : [int] dimension of the latent variable
            device : [str] 'cpu' or 'gpu';
        """

        super(SawETM, self).__init__()
        setattr(self, '_model_name', 'SawETM')

        # constants
        self.device = device
        self.gam_prior = torch.tensor(1.0, dtype=torch.float, device=device)
        self.real_min = torch.tensor(1e-30, dtype=torch.float, device=device)
        self.theta_max = torch.tensor(1000.0, dtype=torch.float, device=device)
        self.wei_shape_min = torch.tensor(1e-1, dtype=torch.float, device=device)
        self.wei_shape_max = torch.tensor(100.0, dtype=torch.float, device=device)

        # hyper-parameters
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_topics_list = num_topics_list
        self.num_hiddens_list = num_hiddens_list
        assert len(num_topics_list) == len(num_hiddens_list)
        self.num_layers = len(num_topics_list)

        # learnable word embeddings
        if word_embeddings is not None:
            self.rho = nn.Parameter(torch.from_numpy(word_embeddings).float())
        else:
            self.rho = nn.Parameter(
                torch.empty(self.vocab_size, self.embed_size).normal_(std=0.02))

        # topic embeddings for different latent layers
        self.alpha = nn.ParameterList([])
        for n in range(self.num_layers):
            self.alpha.append(nn.Parameter(
                torch.empty(self.num_topics_list[n], self.embed_size).normal_(std=0.02)))

        # deterministic mapping to obtain hierarchical features
        self.h_encoder = nn.ModuleList([])
        for n in range(self.num_layers):
            if n == 0:
                self.h_encoder.append(
                    ResBlock(self.vocab_size, self.num_hiddens_list[n]))
            else:
                self.h_encoder.append(
                    ResBlock(self.num_hiddens_list[n - 1], self.num_hiddens_list[n]))

        # variational encoder to obtain posterior parameters
        self.q_theta = nn.ModuleList([])
        for n in range(self.num_layers):
            if n == self.num_layers - 1:
                self.q_theta.append(
                    nn.Linear(self.num_hiddens_list[n], 2 * self.num_topics_list[n]))
            else:
                self.q_theta.append(nn.Linear(
                    self.num_hiddens_list[n] + self.num_topics_list[n], 2 * self.num_topics_list[n]))

    def log_max(self, x):
        return torch.log(torch.max(x, self.real_min))

    def reparameterize(self, shape, scale, sample_num=50):
        """
            Returns a sample from a Weibull distribution via reparameterization.
        """
        shape = shape.unsqueeze(0).repeat(sample_num, 1, 1)
        scale = scale.unsqueeze(0).repeat(sample_num, 1, 1)
        eps = torch.rand_like(shape, dtype=torch.float, device=self.device)
        samples = scale * torch.pow(- self.log_max(1 - eps), 1 / shape)
        return torch.clamp(samples.mean(0), self.real_min.item(), self.theta_max.item())

    def kl_weibull_gamma(self, wei_shape, wei_scale, gam_shape, gam_scale):
        """
            Returns the Kullback-Leibler divergence between a Weibull distribution and a Gamma distribution.
        """
        euler_mascheroni_c = torch.tensor(0.5772, dtype=torch.float, device=self.device)
        t1 = torch.log(wei_shape) + torch.lgamma(gam_shape)
        t2 = - gam_shape * torch.log(wei_scale * gam_scale)
        t3 = euler_mascheroni_c * (gam_shape / wei_shape - 1) - 1
        t4 = gam_scale * wei_scale * torch.exp(torch.lgamma(1 + 1 / wei_shape))
        return (t1 + t2 + t3 + t4).sum(1).mean()

    def get_nll(self, x, x_reconstruct):
        """
            Returns the negative Poisson likelihood of observational count data.
        """
        log_likelihood = self.log_max(x_reconstruct) * x - torch.lgamma(1.0 + x) - x_reconstruct
        neg_log_likelihood = - torch.sum(log_likelihood, dim=1, keepdim=False).mean()
        return neg_log_likelihood

    def get_phi(self):
        """
            Returns the factor loading matrix by utilizing sawtooth connection.
        """
        phis = []
        for n in range(self.num_layers):
            if n == 0:
                phi = torch.softmax(torch.mm(
                    self.rho, self.alpha[n].transpose(0, 1)), dim=0)
            else:
                phi = torch.softmax(torch.mm(
                    self.alpha[n - 1].detach(), self.alpha[n].transpose(0, 1)), dim=0)
            phis.append(phi)
        return phis

    def compute_loss(self, x, x_recon, ks, lambs):
        nll = self.get_nll(x, x_recon[0])

        kl_loss = []
        for n in range(self.num_layers):
            if n == self.num_layers - 1:
                kl_loss.append(self.kl_weibull_gamma(
                    ks[n], lambs[n], self.gam_prior, self.gam_prior))
            else:
                kl_loss.append(self.kl_weibull_gamma(
                    ks[n], lambs[n], x_recon[n + 1], self.gam_prior))

        return nll, sum(kl_loss)

    def forward(self, x, is_training=True):
        """
            Forward pass: compute the kl loss and data likelihood.
        """
        hidden_feats = []
        for n in range(self.num_layers):
            if n == 0:
                hidden_feats.append(self.h_encoder[n](x))
            else:
                hidden_feats.append(self.h_encoder[n](hidden_feats[-1]))

        # =================================================================================
        self.phis = self.get_phi()

        ks = []
        lambs = []
        thetas = []
        phi_by_theta_list = []
        for n in range(self.num_layers - 1, -1, -1):
            if n == self.num_layers - 1:
                joint_feat = hidden_feats[n]
            else:
                joint_feat = torch.cat((hidden_feats[n], phi_by_theta_list[0]), dim=1)

            k, lamb = torch.chunk(F.softplus(self.q_theta[n](joint_feat)), 2, dim=1)
            k = torch.clamp(k, self.wei_shape_min.item(), self.wei_shape_max.item())
            lamb = torch.clamp(lamb, self.real_min.item())

            if is_training:
                lamb = lamb / torch.exp(torch.lgamma(1 + 1 / k))
                theta = self.reparameterize(k, lamb, sample_num=3) if n == 0 else self.reparameterize(k, lamb)
            else:
                theta = torch.min(lamb, self.theta_max)

            phi_by_theta = torch.mm(theta, self.phis[n].t())
            phi_by_theta_list.insert(0, phi_by_theta)
            thetas.insert(0, theta)
            lambs.insert(0, lamb)
            ks.insert(0, k)

        return thetas, phi_by_theta_list, ks, lambs

    # train and test
    def train_one_epoch(self, dataloader, model_opt, epoch_idx, args):
        '''
        Train for one epoch
        Inputs:
            model_opt  : Optimizer for model
            dataloader : Train dataset with form of dataloader
            epoch_idx  : Current epoch on training stage
            args       : Argument dict

        Return:
            local_theta   : Concatenation of theta with total dataset

        '''
        local_theta = None
        local_labels = None
        loss_t, kl_loss_t, likelihood_t = 0., 0., 0.

        self.train()
        train_bar = tqdm(iterable=dataloader)
        for batch_idx, (batch_x, labels) in enumerate(train_bar):
            # tqdm
            train_bar.set_description(f'Epoch [{epoch_idx}/{args.num_epochs}]')
            train_bar.set_postfix(loss=loss_t / (batch_idx + 1), KL_loss=kl_loss_t / (batch_idx + 1),
                                  likelihood=likelihood_t / (batch_idx + 1))
            # forward
            batch_x = batch_x.float().to(self.device)
            thetas, recon_x, ks, lambs = self.forward(batch_x, is_training=True)

            nll, kl_loss = self.compute_loss(batch_x, recon_x, ks, lambs)
            # negative ELBO
            loss = nll + kl_loss

            # backward
            loss.backward()
            model_opt.step()
            model_opt.zero_grad()

            # accumulate loss
            loss_t += loss.item()
            kl_loss_t += kl_loss.item()
            likelihood_t += nll.item()

            # collect output
            theta = thetas[0]
            if local_theta is None:
                local_theta = theta.cpu().detach().numpy()
                local_labels = labels.cpu().detach().numpy()
            else:
                local_theta = np.concatenate((local_theta, theta.cpu().detach().numpy()))
                local_labels = np.concatenate((local_labels, labels.cpu().detach().numpy()))

        return copy.deepcopy(local_theta), copy.deepcopy(local_labels)

    def test_one_epoch(self, dataloader):
        '''
        Test for one epoch
        Inputs:
            dataloader : Train dataset with form of dataloader

        Return:
            local_theta   : Concatenation of theta with total dataset
        '''
        local_theta = None
        local_labels = None
        loss_t, kl_loss_t, likelihood_t = 0., 0., 0.

        self.eval()
        with torch.no_grad():
            test_bar = tqdm(iterable=dataloader)
            for i, (batch_x, labels) in enumerate(test_bar):
                test_bar.set_description(f'Testing stage: ')
                test_bar.set_postfix(loss=loss_t / (i + 1), KL_loss=kl_loss_t / (i + 1),
                                     likelihood=likelihood_t / (i + 1))

                # forward
                batch_x = batch_x.float().to(self.device)
                thetas, recon_x, ks, lambs = self.forward(batch_x, is_training=True)

                nll, kl_loss = self.compute_loss(batch_x, recon_x, ks, lambs)
                # negative ELBO
                loss = nll + kl_loss

                # accumulate loss
                loss_t += loss.item()
                kl_loss_t += kl_loss.item()
                likelihood_t += nll.item()

                # collect output
                theta = thetas[0]
                if local_theta is None:
                    local_theta = theta.cpu().detach().numpy()
                    local_labels = labels.cpu().detach().numpy()
                else:
                    local_theta = np.concatenate((local_theta, theta.cpu().detach().numpy()))
                    local_labels = np.concatenate((local_labels, labels.cpu().detach().numpy()))

        return copy.deepcopy(local_theta), copy.deepcopy(local_labels)

    # save and load
    def save(self, model_path: str = '../save_models'):
        """
        save model
        Inputs:
            model_path : [str] the path to save the model, default '../save_models/SawETM.pth';
        """
        # create the directory path
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        # Save the model
        torch.save({'state_dict': self.state_dict()}, model_path + '/' + self._model_name + '.pth')
        print('model has been saved by ' + model_path + '/' + self._model_name + '.pth')

    def load(self, model_path):
        """
        load model
        Inputs:
            model_path : [str] the path to load the model;
        """
        assert os.path.exists(model_path), 'Path Error: can not find the path to load the model'
        # Load the model
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['state_dict'])
