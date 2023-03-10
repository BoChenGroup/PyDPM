"""
===========================================
VAE
Auto-Encoding Variational Bayes
Diederik P. Kingma， Max Welling
Publihsed in 2014

===========================================
"""

# Author: Muyao Wang <flare935694542@163.com>, Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import os

class VAE(nn.Module):
    def __init__(self, in_dim: int, z_dim: int, encoder_hid_dims: list, decoder_hid_dims: list, device='cpu'):
        """
        The basic model for VAE
        Inputs:
            x_dim : [int] the size of input;
            h_dim1 : [int] latent dimension1;
            h_dim2 : [int] latent dimension2;
            z_dim : [int] the normal distribution dimension;
            device : [str] 'cpu' or 'gpu';
        """
        super(VAE, self).__init__()
        setattr(self, '_model_name', 'VAE')
        self.z_dim = z_dim
        self.in_dim = in_dim
        self.encoder_hid_dims = encoder_hid_dims
        self.decoder_hid_dims = decoder_hid_dims
        self.device = device
        self.vae_encoder = VAE_Encoder(in_dim=self.in_dim, hid_dims=self.encoder_hid_dims, z_dim=self.z_dim, device=self.device)
        self.vae_decoder = VAE_Decoder(in_dim=self.in_dim, hid_dims=self.decoder_hid_dims, z_dim=self.z_dim, device=self.device)

    def compute_loss(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE, KLD

    def forward(self, x):
        """
        Forward process of VAE
        Inputs:
            x : [tensor] input tensor;
        Outputs:
            recon_x : [tensor] reconstruction of x
            mu : [tensor] mean of posterior distribution;
            log_var : [tensor] log variance of posterior distribution;
        """
        z, mu, log_var = self.vae_encoder(x.view(x.shape[0], -1))
        recon_x = self.vae_decoder(z)
        return recon_x, mu, log_var

    def sample(self, batch_size):
        """
        Sample from generator
        Inputs:
            batch_size : [int] number of img which you want;
        Outputs:
            recon_x : [tensor] reconstruction of x
        """
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        recon_x = self.vae_decoder(z)
        return recon_x

    def show(self):
        """
        Show the learned latent variables' effect
        Outputs:
            recon_x : [tensor] reconstruction of x
        """
        x, y = torch.meshgrid([torch.arange(-3, 3, 0.5), torch.arange(-3, 3, 0.5)])
        z = torch.stack((x, y), 2).view(x.shape[0]**2, 2).to(self.device)
        # z = torch.randn(batch_size, self.z_dim).to(self.device)
        recon_x = self.vae_decoder(z)
        # print(recon_x.shape)
        return recon_x

    def train_one_epoch(self, model_opt, dataloader, epoch, n_epochs):
        '''
        Train for one epoch
        Inputs:
            model_opt  : Optimizer for model
            dataloader : Train dataset with form of dataloader
            epoch      : Current epoch on training stage
            n_epoch    : Total number of epochs on training stage

        Attributes:
            local_mu      : Concatenation of mu with total dataset
            local_log_var : Concatenation of log_var with total dataset
        '''
        self.train()
        self.local_mu = None
        self.local_log_var = None
        loss_t, kl_loss_t, likelihood_t = 0, 0, 0
        train_bar = tqdm(iterable=dataloader)
        for i, (data, _) in enumerate(train_bar):
            train_bar.set_description(f'Epoch [{epoch}/{n_epochs}]')
            train_bar.set_postfix(loss=loss_t / (i + 1), KL_loss=kl_loss_t / (i + 1), likelihood=likelihood_t / (i + 1))
            data = data.view(data.size(0), self.in_dim).to(self.device)
            recon_x, mu, log_var = self.forward(data)
            llh, kl_loss = self.compute_loss(recon_x, data, mu, log_var)
            loss = llh + kl_loss

            loss.backward()
            model_opt.step()
            model_opt.zero_grad()

            loss_t += loss.item()
            kl_loss_t += kl_loss.item()
            likelihood_t += llh.item()

            if self.local_mu is None:
                self.local_mu = mu.cpu().detach().numpy()
                self.local_log_var = log_var.cpu().detach().numpy()
            else:
                self.local_mu = np.concatenate((self.local_mu, mu.cpu().detach().numpy()))
                self.local_log_var = np.concatenate((self.local_log_var, log_var.cpu().detach().numpy()))

        return copy.deepcopy(self.local_mu), copy.deepcopy(self.local_log_var)

    def test_one_epoch(self, dataloader):
        '''
        Test for one epoch
        Inputs:
            dataloader : Train dataset with form of dataloader

        Attributes:
            local_mu      : Concatenation of mu with total dataset
            local_log_var : Concatenation of log_var with total dataset
        '''
        self.eval()
        local_mu = None
        local_log_var = None
        loss_t, kl_loss_t, likelihood_t = 0, 0, 0
        with torch.no_grad():
            test_bar = tqdm(iterable=dataloader)
            for i, (data, _) in enumerate(test_bar):
                test_bar.set_description(f'Testing stage: ')
                test_bar.set_postfix(loss=loss_t / (i + 1), KL_loss=kl_loss_t / (i + 1), likelihood=likelihood_t / (i + 1))
                data = data.view(data.size(0), self.in_dim).to(self.device)
                recon_x, mu, log_var = self.forward(data)
                llh, kl_loss = self.compute_loss(recon_x, data, mu, log_var)
                loss_t += (llh.item() + kl_loss.item())
                kl_loss_t += kl_loss.item()
                likelihood_t += llh.item()

                if local_mu is None:
                    local_mu = mu.cpu().detach().numpy()
                    local_log_var = log_var.cpu().detach().numpy()
                else:
                    local_mu = np.concatenate((local_mu, mu.cpu().detach().numpy()))
                    local_log_var = np.concatenate((local_log_var, log_var.cpu().detach().numpy()))

        return local_mu, local_log_var

    def save(self, model_path: str = '../save_models'):
        """
        save model
        Inputs:
            model_path : [str] the path to save the model, default '../save_models/VAE.pth';
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

class VAE_Encoder(nn.Module):
    def __init__(self, in_dim: int, hid_dims: list, z_dim: int, device: str='cpu'):
        super(VAE_Encoder, self).__init__()
        self.in_dim = in_dim
        self.hid_dims = hid_dims
        self.z_dim = z_dim
        self.device = device
        self.num_layers = len(self.hid_dims)
        self.fc_encoder = nn.ModuleList()
        self.fc_mu = nn.Linear(self.hid_dims[-1], self.z_dim).to(self.device)
        self.fc_var = nn.Linear(self.hid_dims[-1], self.z_dim).to(self.device)
        for layer_index in range(self.num_layers):
            if layer_index == 0:
                self.fc_encoder.append(nn.Linear(self.in_dim, self.hid_dims[layer_index]).to(device))
            else:
                self.fc_encoder.append(nn.Linear(self.hid_dims[layer_index - 1], self.hid_dims[layer_index]).to(device))

    def encoder(self, x):
        """
        Encode x to latent distribution
        Inputs:
            x : [tensor] the input tensor;
        Outputs:
            mu : [tensor] mean of posterior distribution;
            log_var : [tensor] log variance of posterior distribution;
        """
        for layer_index in range(self.num_layers):
            if layer_index == 0:
                x = F.relu(self.fc_encoder[layer_index](x))
            else:
                x = F.relu(self.fc_encoder[layer_index](x))
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        return mu, var  # mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Gaussian re_parameterization
        Inputs:
            mu : [tensor] mean of posterior distribution;
            log_var : [tensor] log variance of posterior distribution;
        Outputs:
            z_sample : [tensor] sample from the distribution
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, x):
        """
        Forward process of VAE_Encoder
        Inputs:
            x : [tensor] input tensor;
        Outputs:
            z : [tensor] latent variable of x
            mu : [tensor] mean of posterior distribution;
            log_var : [tensor] log variance of posterior distribution;
        """
        mu, log_var = self.encoder(x.view(x.shape[0],-1))
        z = self.reparameterize(mu, log_var)

        return z, mu, log_var


class VAE_Decoder(nn.Module):
    def __init__(self, in_dim, hid_dims, z_dim, device: str='cpu'):
        super(VAE_Decoder, self).__init__()
        self.in_dim = in_dim
        self.hid_dims = hid_dims
        self.z_dim = z_dim
        self.num_layers = len(self.hid_dims)
        self.device = device
        self.fc_decoder = nn.ModuleList()
        for layer_index in range(self.num_layers):
            if layer_index == 0:
                self.fc_decoder.append(nn.Linear(self.z_dim, self.hid_dims[layer_index]).to(device))
            else:
                self.fc_decoder.append(nn.Linear(self.hid_dims[layer_index - 1], self.hid_dims[layer_index]).to(device))
        self.fc_decoder.append(nn.Linear(self.hid_dims[-1], self.in_dim).to(device))


    def decoder(self, z):
        """
        Reconstruct from the z
        Inputs:
            z : [tensor] latent variable of x
        Outputs:
            recon_x : [tensor] the reconstruction of x
        """
        for layer_index in range(self.num_layers):
            z = F.relu(self.fc_decoder[layer_index](z))
        recon_x = torch.sigmoid(self.fc_decoder[-1](z))
        return recon_x  # recon_x

    def forward(self, z):
        """
        Forward process of VAE_Decoder
        Inputs:
            z : [tensor] latent variable of x;
        Outputs:
            recon_x : [tensor] reconstruction of x
        """
        recon_x = self.decoder(z)
        return recon_x