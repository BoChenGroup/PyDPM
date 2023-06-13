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

import os
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, in_dim: int, z_dim: int, encoder_hid_dims: list, decoder_hid_dims: list, device: str='cpu'):
        """
        The basic model for VAE
        Inputs:
            in_dim : [int] dimension of input
            encoder_hid_dims : [list] list of dimension in encoder
            decoder_hid_dims : [list] list of dimension in decoder
            z_dim : [int] dimension of the latent variable
            device : [str] 'cpu' or 'gpu';
        """
        super(VAE, self).__init__()
        setattr(self, '_model_name', 'VAE')
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.encoder_hid_dims = encoder_hid_dims
        self.decoder_hid_dims = decoder_hid_dims
        self.device = device

        self.vae_encoder = VAE_Encoder(in_dim=self.in_dim, hid_dims=self.encoder_hid_dims, z_dim=self.z_dim, device=self.device)
        self.vae_decoder = VAE_Decoder(in_dim=self.in_dim, hid_dims=self.decoder_hid_dims, z_dim=self.z_dim, device=self.device)

    # forward
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

    # loss
    def compute_loss(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE, KLD

    # train and test
    def train_one_epoch(self, dataloader, model_opt, epoch_idx, args):
        '''
        Train for one epoch
        Inputs:
            model_opt  : Optimizer for model
            dataloader : Train dataset with form of dataloader
            epoch_idx  : Current epoch on training stage
            args       : Argument dict

        Attributes:
            local_mu      : Concatenation of mu with total dataset
            local_log_var : Concatenation of log_var with total dataset
        '''
        local_mu = None
        local_log_var = None
        loss_t, kl_loss_t, likelihood_t = 0, 0, 0

        self.train()
        train_bar = tqdm(iterable=dataloader)
        for batch_idx, (batch_x, _) in enumerate(train_bar):
            # forward
            batch_x = batch_x.view(batch_x.shape[0], self.in_dim).to(self.device)
            recon_x, mu, log_var = self.forward(batch_x)
            llh, kl_loss = self.compute_loss(recon_x, batch_x, mu, log_var)
            loss = llh + kl_loss

            # backward
            loss.backward()
            model_opt.step()
            model_opt.zero_grad()

            # accumulate loss
            loss_t += loss.item()
            kl_loss_t += kl_loss.item()
            likelihood_t += llh.item()

            # collect output
            if local_mu is None:
                local_mu = mu.cpu().detach().numpy()
                local_log_var = log_var.cpu().detach().numpy()
            else:
                local_mu = np.concatenate((local_mu, mu.cpu().detach().numpy()))
                local_log_var = np.concatenate((local_log_var, log_var.cpu().detach().numpy()))

            # tqdm
            train_bar.set_description(f'Epoch [{epoch_idx}/{args.num_epochs}]')
            train_bar.set_postfix(loss=loss_t / (batch_idx + 1), KL_loss=kl_loss_t / (batch_idx + 1), likelihood=likelihood_t / (batch_idx + 1))

        return copy.deepcopy(local_mu), copy.deepcopy(local_log_var)

    def test_one_epoch(self, dataloader):
        '''
        Test for one epoch
        Inputs:
            dataloader : Train dataset with form of dataloader

        Attributes:
            local_mu      : Concatenation of mu with total dataset
            local_log_var : Concatenation of log_var with total dataset
        '''
        local_mu = None
        local_log_var = None
        loss_t, kl_loss_t, likelihood_t = 0, 0, 0

        self.eval()
        with torch.no_grad():
            test_bar = tqdm(iterable=dataloader)
            for i, (batch_x, _) in enumerate(test_bar):
                test_bar.set_description(f'Testing stage: ')
                test_bar.set_postfix(loss=loss_t / (i + 1), KL_loss=kl_loss_t / (i + 1), likelihood=likelihood_t / (i + 1))

                # forward
                batch_x = batch_x.view(batch_x.shape[0], self.in_dim).to(self.device)
                recon_x, mu, log_var = self.forward(batch_x)
                llh, kl_loss = self.compute_loss(recon_x, batch_x, mu, log_var)

                # accumulate loss
                loss_t += (llh.item() + kl_loss.item())
                kl_loss_t += kl_loss.item()
                likelihood_t += llh.item()

                # collect output
                if local_mu is None:
                    local_mu = mu.cpu().detach().numpy()
                    local_log_var = log_var.cpu().detach().numpy()
                else:
                    local_mu = np.concatenate((local_mu, mu.cpu().detach().numpy()))
                    local_log_var = np.concatenate((local_log_var, log_var.cpu().detach().numpy()))

        return local_mu, local_log_var

    # save and load
    def save(self, model_path: str = '../save_models'):
        """
        save model
        Inputs:
            model_path : [str] the path to save the model, default '../save_models/VAE.pth';
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

    # visualization
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

    # def show(self):
    #     """
    #     Show the learned latent variables' effect
    #     Outputs:
    #         recon_x : [tensor] reconstruction of x
    #     """
    #     x, y = torch.meshgrid([torch.arange(-3, 3, 0.5), torch.arange(-3, 3, 0.5)])
    #     z = torch.stack((x, y), 2).view(x.shape[0]**2, 2).to(self.device)
    #     # z = torch.randn(batch_size, self.z_dim).to(self.device)
    #     recon_x = self.vae_decoder(z)
    #     # print(recon_x.shape)
    #     return recon_x

class VAE_Encoder(nn.Module):
    def __init__(self, in_dim: int, hid_dims: list, z_dim: int, device: str='cpu'):
        super(VAE_Encoder, self).__init__()

        self.in_dim = in_dim
        self.hid_dims = hid_dims
        self.z_dim = z_dim
        self.device = device
        self.num_layers = len(self.hid_dims)

        self.fc_encoder = nn.ModuleList()
        self.fc_mu = nn.Linear(self.hid_dims[-1], self.z_dim, device=self.device)
        self.fc_var = nn.Linear(self.hid_dims[-1], self.z_dim, device=self.device)

        for layer_index in range(self.num_layers):
            if layer_index == 0:
                self.fc_encoder.append(nn.Linear(self.in_dim, self.hid_dims[layer_index]).to(device))
            else:
                self.fc_encoder.append(nn.Linear(self.hid_dims[layer_index - 1], self.hid_dims[layer_index]).to(device))

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
        x = x.view(x.shape[0], -1)
        for layer_index in range(self.num_layers):
            if layer_index == 0:
                x = F.relu(self.fc_encoder[layer_index](x))
            else:
                x = F.relu(self.fc_encoder[layer_index](x))
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        z = self.reparameterize(mu, log_var)

        return z, mu, log_var


class VAE_Decoder(nn.Module):
    def __init__(self, in_dim: int, hid_dims: list, z_dim: int, device: str='cpu'):
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

    def forward(self, z):
        """
        Forward process of VAE_Decoder
        Inputs:
            z : [tensor] latent variable of x;
        Outputs:
            recon_x : [tensor] reconstruction of x
        """
        for layer_index in range(self.num_layers):
            z = F.relu(self.fc_decoder[layer_index](z))
        recon_x = torch.sigmoid(self.fc_decoder[-1](z))

        return recon_x