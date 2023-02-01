"""
===========================================
RealNVP
DENSITY ESTIMATION USING REAL NVP
Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
Publihsed in ICLR 2017

===========================================
"""

# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from tqdm import tqdm
import copy
import os


class NFlow(nn.Module):
    def __init__(self, in_dim: int, flows: list, device: str = "cpu"):
        """
        The basic model for Normalizing Flows
        Inputs:
            in_dim : [int] the size of input;
            flows  : [list] list of flows;
            device : [str] 'cpu' or 'cuda:0';
        """
        super(NFlow, self).__init__()
        setattr(self, '_model_name', 'NFlow')

        self.in_dim = in_dim
        self.prior = MultivariateNormal(torch.zeros(in_dim).to(device), torch.eye(in_dim).to(device))
        self.flows = nn.ModuleList(flows).to(device=device)
        self.device = device

    def forward(self, x):
        '''
        Forward process of flow-based model
        Inputs:
            x : [tensor] input tensor;
        Outputs:
            z : [tensor] latent representation
            prior_logprob : [tensor] log prior;
            log_det : [tensor] log determinant of transformation;
        '''
        bsz, _ = x.shape
        log_det = torch.zeros(bsz)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z):
        '''
        Inverse process of flow-based model
        Inputs:
            z : [tensor] latent representation;
        Outputs:
            x : [tensor] input tensor;
            log_det : [tensor] log determinant of transformation;
        '''
        bsz, _ = z.shape
        log_det = torch.zeros(bsz)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        """
        Sample from noise via transformation
        Inputs:
            n_samples : [int] number of samples;
        Outputs:
            x : [tensor] simples from z
        """
        z = self.prior.sample((n_samples,))
        x, _ = self.inverse(z)
        return x

    def train_one_epoch(self, model_opt, dataloader, epoch, n_epochs):
        '''
        Train for one epoch
        Inputs:
            model_opt  : Optimizer for model
            dataloader : Train dataset with form of dataloader
            epoch      : Current epoch on training stage
            n_epoch    : Total number of epochs on training stage

        Attributes:
            local_z      : Concatenation of z with total dataset

        '''
        self.train()
        self.local_z = None
        loss_t, logprior_t, logdet_t = 0, 0, 0
        train_bar = tqdm(iterable=dataloader)
        for i, data in enumerate(train_bar):
            train_bar.set_description(f'Epoch [{epoch}/{n_epochs}]')
            train_bar.set_postfix(loss=loss_t / (i + 1), prior=logprior_t / (i + 1), logdet=logdet_t / (i + 1))
            data = data.view(data.size(0), self.in_dim).to(self.device)

            z, prior_logprob, log_det = self.forward(data)
            loss = - torch.mean(prior_logprob + log_det)
            model_opt.zero_grad()
            loss.backward()
            model_opt.step()

            loss_t += (prior_logprob + log_det).mean().item()
            logprior_t += (prior_logprob).mean().item()
            logdet_t += (log_det).mean().item()

            if self.local_z is None:
                self.local_z = z.cpu().detach().numpy()
            else:
                self.local_z = np.concatenate((self.local_z, z.cpu().detach().numpy()))

        return copy.deepcopy(self.local_z)


    def test_one_epoch(self, dataloader):
        '''
        Test for one epoch
        Inputs:
            dataloader : Train dataset with form of dataloader

        Attributes:
            local_z      : Concatenation of z with total dataset
        '''
        self.eval()
        local_z = None
        loss_t, logprior_t, logdet_t = 0, 0, 0
        test_bar = tqdm(iterable=dataloader)
        for i, data in enumerate(test_bar):
            test_bar.set_description(f'Testing stage: ')
            test_bar.set_postfix(loss=loss_t / (i + 1), prior=logprior_t / (i + 1), logdet=logdet_t / (i + 1))
            data = data.view(data.size(0), self.in_dim).to(self.device)

            z, prior_logprob, log_det = self.forward(data)

            loss_t += (prior_logprob + log_det).mean().item()
            logprior_t += (prior_logprob).mean().item()
            logdet_t += (log_det).mean().item()

            if local_z is None:
                local_z = z.cpu().detach().numpy()
            else:
                local_z = np.concatenate((local_z, z.cpu().detach().numpy()))

        return copy.deepcopy(local_z)

    def save(self, model_path: str = '../save_models'):
        """
        save model
        Inputs:
            model_path : [str] the path to save the model, default '../save_models/NFlow.pth';
        """
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


class Net(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class RealNVP_2D(nn.Module):
    '''
    RealNVP
    DENSITY ESTIMATION USING REAL NVP
    Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
    Publihsed in  2017
    '''
    def __init__(self, dim: int, hidden_dim: int, net=Net, device: str = "cpu"):
        '''
        The basic Implementation of RealNVP for 2D data
        Inputs:
            in_dim      : [int] the size of input;
            hidden_dim  : [int] latent dimension;
            net         : [nn.Module] backbone;
            device      : [str] 'cpu' or 'cuda:0';
        Attributes:
            t1  : network of shift while fixing xd
            s1  : network of scale while fixing xd
            t2  : network of shift while fixing xD
            s2  : network of scale while fixing xD
        '''
        super().__init__()
        self.dim = dim
        self.device = device
        # net for step 1
        self.t1 = net(dim//2, dim//2, hidden_dim).to(self.device)
        self.s1 = net(dim//2, dim//2, hidden_dim).to(self.device)
        # net for step 2
        self.t2 = net(dim//2, dim//2, hidden_dim).to(self.device)
        self.s2 = net(dim//2, dim//2, hidden_dim).to(self.device)

    def forward(self, x):
        '''
        Forward process for RealNVP_2D with 2 steps
        step 1 : fix xd
        step 2 : fix xD
        '''
        xd, xD = x[:, :self.dim // 2], x[:, self.dim // 2:]
        # step 1
        s_1 = self.s1(xd)
        t_1 = self.t1(xd)
        xD = t_1 + xD * torch.exp(s_1)
        # step 2
        s_2 = self.s2(xD)
        t_2 = self.t2(xD)
        xd = t_2 + xd * torch.exp(s_2)
        # get final z
        z = torch.cat([xd, xD], dim=1)
        log_det = torch.sum(s_1, dim=1) + torch.sum(s_2, dim=1)
        return z, log_det

    def inverse(self, z):
        '''
        Inverse process for RealNVP_2D with 2 steps
        step 1 : fix xD
        step 2 : fix xd
        '''
        zd, zD = z[:, :self.dim // 2], z[:, self.dim // 2:]
        # step 1
        t_2 = self.t2(zD)
        s_2 = self.s2(zD)
        zd = (zd - t_2) * torch.exp(-s_2)
        # step 2
        t_1 = self.t1(zd)
        s_1 = self.s1(zd)
        zD = (zD - t_1) * torch.exp(-s_1)
        # get final x
        x = torch.cat([zd, zD], dim=1)
        log_det = torch.sum(-s_1, dim=1) + torch.sum(-s_2, dim=1)
        return x, log_det

