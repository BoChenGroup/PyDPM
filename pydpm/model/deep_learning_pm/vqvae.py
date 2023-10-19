"""
===========================================
VQ-VAE
Neural Discrete Representation Learning
Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
Publihsed in 2017

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
from torch.autograd import Function, Variable


class VQVAE(nn.Module):
    def __init__(self, embed_dim: int, num_embed: int, in_dim: int, z_dim: int, encoder_hid_dims: list, decoder_hid_dims: list,
                 device='cpu'):
        """
        The basic model for VQVAE
        Inputs:
            embed_dim : [int] dimension of codebook embedding
            num_embed : [int] the number of codebook vectors
            in_dim : [int] dimension of input
            encoder_hid_dims : [list] list of dimension in encoder
            decoder_hid_dims : [list] list of dimension in decoder
            z_dim : [int] dimension of the latent variable
            device : [str] 'cpu' or 'gpu';
        """
        super(VQVAE, self).__init__()
        setattr(self, '_model_name', 'VQVAE')
        self.z_dim = z_dim
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_embed = num_embed
        self.encoder_hid_dims = encoder_hid_dims
        self.decoder_hid_dims = decoder_hid_dims
        self.device = device

        self.emb = NearestEmbed(self.num_embed, self.embed_dim).to(self.device)
        self.vq_coef = 0.2
        self.comit_coef = 0.4
        self.ce_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0
        self.vqvae_encoder = VQVAE_Encoder(in_dim=self.in_dim, hid_dims=self.encoder_hid_dims,
                                         z_dim=self.z_dim, device=self.device)
        self.vqvae_decoder = VQVAE_Decoder(in_dim=self.in_dim, hid_dims=self.decoder_hid_dims,
                                         z_dim=self.z_dim, device=self.device)

    def compute_loss(self,  x, recon_x, z_e, emb):
        """
        Compute loss of VQVAE
        Inputs:
            x : [tensor] input tensor;
            recon_x : [tensor] reconstruction of x;
            z_e : [tensor] the output of encoder;
            emb : [tensor] the quantization of z_e without gradient;
        Outputs:
            loss : [tensor] sum of reconstruct loss, vq_loss and commit loss;
        """
        self.ce_loss = F.binary_cross_entropy(recon_x, x.view(-1, self.in_dim))
        self.vq_loss = F.mse_loss(emb, z_e.detach())
        self.commit_loss = F.mse_loss(z_e, emb.detach())
        return self.ce_loss + self.vq_coef*self.vq_loss + self.comit_coef*self.commit_loss

    def forward(self, x):
        """
        Forward process of VQVAE
        Inputs:
            x : [tensor] input tensor;
        Outputs:
            recon_x : [tensor] reconstruction of x;
            z_e : [tensor] the output of encoder;
            emb : [tensor] the quantization of z_e without gradient;
        """
        z_e = self.vqvae_encoder(x)
        z_q, _ = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        recon_x = self.vqvae_decoder(z_q)
        return recon_x, z_e, emb

    def train_one_epoch(self, dataloader, model_opt, epoch_idx, args):
        '''
        Train for one epoch
        Inputs:
            model_opt  : Optimizer for model
            dataloader : Train dataset with form of dataloader
            epoch_idx  : Current epoch on training stage
            args       : Argument dict
        '''
        loss_t, kl_loss_t, likelihood_t = 0, 0, 0

        self.train()
        train_bar = tqdm(iterable=dataloader)
        for batch_idx, (batch_x, _) in enumerate(train_bar):
            # forward
            batch_x = batch_x.view(batch_x.shape[0], self.in_dim).to(self.device)
            recon_x, ze, emb = self.forward(batch_x)
            loss = self.compute_loss(batch_x, recon_x, ze, emb)

            # backward
            loss.backward()
            model_opt.step()
            model_opt.zero_grad()

            # accumulate loss
            loss_t += loss.item()

            # tqdm
            train_bar.set_description(f'Epoch [{epoch_idx}/{args.num_epochs}]')
            train_bar.set_postfix(loss=loss_t / (batch_idx + 1))


    def test_one_epoch(self, dataloader):
        '''
        Test for one epoch
        Inputs:
            dataloader : Train dataset with form of dataloader
        '''
        loss_t, kl_loss_t, likelihood_t = 0, 0, 0
        self.eval()
        with torch.no_grad():
            test_bar = tqdm(iterable=dataloader)
            for i, (batch_x, _) in enumerate(test_bar):
                test_bar.set_description(f'Testing stage: ')
                test_bar.set_postfix(loss=loss_t / (i + 1))

                # forward
                batch_x = batch_x.view(batch_x.shape[0], self.in_dim).to(self.device)
                recon_x, ze, emb = self.forward(batch_x)
                loss = self.compute_loss(batch_x, recon_x, ze, emb)

                # accumulate loss
                loss_t += loss.item()

    # save and load
    def save(self, model_path: str = '../save_models'):
        """
        save model
        Inputs:
            model_path : [str] the path to save the model, default '../save_models/VQVAE.pth';
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

    def sample(self, batch_size):
        """
        Sample from generator
        Inputs:
            batch_size : [int] number of img which you want;
        Outputs:
            recon_x : [tensor] reconstruction of x
        """
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        emb, _ = self.emb(z)
        recon_x = self.vqvae_decoder(emb)
        return recon_x

class VQVAE_Encoder(nn.Module):
    def __init__(self, in_dim: int, hid_dims: list, z_dim: int, device: str='cpu'):
        super(VQVAE_Encoder, self).__init__()

        self.in_dim = in_dim
        self.hid_dims = hid_dims
        self.z_dim = z_dim
        self.device = device
        self.num_layers = len(self.hid_dims)

        self.fc_encoder = nn.ModuleList()
        self.fc_z = nn.Linear(self.hid_dims[-1], self.z_dim, device=self.device)

        for layer_index in range(self.num_layers):
            if layer_index == 0:
                self.fc_encoder.append(nn.Linear(self.in_dim, self.hid_dims[layer_index]).to(device))
            else:
                self.fc_encoder.append(nn.Linear(self.hid_dims[layer_index - 1], self.hid_dims[layer_index]).to(device))

    def forward(self, x):
        """
        Forward process of VQVAE_Encoder
        Inputs:
            x : [tensor] input tensor;
        Outputs:
            z : [tensor] latent variable of x
        """
        x = x.view(x.shape[0], -1)
        for layer_index in range(self.num_layers):
            if layer_index == 0:
                x = F.relu(self.fc_encoder[layer_index](x))
            else:
                x = F.relu(self.fc_encoder[layer_index](x))
        z = self.fc_z(x)
        return z


class VQVAE_Decoder(nn.Module):
    def __init__(self, in_dim: int, hid_dims: list, z_dim: int, device: str='cpu'):
        super(VQVAE_Decoder, self).__init__()

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
        Forward process of VQVAE_Decoder
        Inputs:
            z : [tensor] latent variable of x;
        Outputs:
            recon_x : [tensor] reconstruction of x
        """
        for layer_index in range(self.num_layers):
            z = F.relu(self.fc_decoder[layer_index](z))
        recon_x = torch.sigmoid(self.fc_decoder[-1](z))

        return recon_x


class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)
                                      ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)