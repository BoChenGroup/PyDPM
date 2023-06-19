'''
===========================================
DDPM
Denoising Diffusion Probabilistic Models
Jonathan Ho, Ajay Jain, Pieter Abbeel
Published in NIPS 2020

Sample implementation of Denoising Diffusion Probabilistic Models in PyTorch
===========================================
'''

# Author: Xinyang Liu <lxy771258012@163.com>, Muyao Wang <flare935694542@163.com>
# License: BSD-3-Clause

import os
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPM(nn.Module):
    def __init__(self, T, net_cfg, ddpm_cfg, device='cuda:0'):
        '''
        The basic model for DDPM
        Inputs:
            T           : [int] Number of time steps in DDPM
            net_cfg     : [dict] Config of net
            ddpm_cfg    : [dict] Config of DDPM
            device      : [str] 'cpu' or 'gpu';
        '''
        super(DDPM, self).__init__()
        setattr(self, '_model_name', 'DDPM')

        self.T = T

        # set net model config
        self.beta_1 = ddpm_cfg['beta_1']
        self.beta_T = ddpm_cfg['beta_T']
        self.device = device

        # initial net model
        self.net = self.net_inital(self.T, net_cfg=net_cfg)

        # initial ddpm model
        self.ddpm_trainer = GaussianDiffusionTrainer(self.net, self.beta_1, self.beta_T, self.T)
        self.ddpm_sampler = GaussianDiffusionSampler(self.net, self.beta_1, self.beta_T, self.T)

    def net_inital(self, T, net_cfg):
        '''
        Initial the model of net in reverse process of DDPM
        Inputs:
            T       : Number of time steps in DDPM
            net_cfg : Config of net
        Output:
            net     : The model of net
        '''
        self.T = T
        self.in_channel = net_cfg['in_channel']
        self.channel = net_cfg['channel']
        self.channel_mult = net_cfg['channel_mult']
        self.attn = net_cfg['attn']
        self.num_res_blocks = net_cfg['num_res_blocks']
        self.dropout = net_cfg['dropout']

        net = UNet(T=self.T, in_channel=self.in_channel, channel=self.channel, channel_mult=self.channel_mult, attn=self.attn,
                   num_res_blocks=self.num_res_blocks, dropout=self.dropout).to(self.device)

        return net

    def train_one_epoch(self, dataloader, optim, epoch_index, args):
        '''
        Inputs:
            dataloader  : dataloader of train dataset
            optim       : Optimizer for model
            epoch_index : [int] Current epoch on training stage
            args        : Hyper-parameters
        '''
        trainer = GaussianDiffusionTrainer(
            self.net, self.beta_1, self.beta_T, self.T).to(self.device)

        # start training
        train_bar = tqdm(iterable=dataloader)
        for i, (images, labels) in enumerate(train_bar):

            x_0 = images.to(self.device)
            loss = trainer(x_0).sum() / 1000.

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), args.grad_clip)
            optim.step()

            train_bar.set_postfix(ordered_dict={
                "epoch": epoch_index,
                "loss": loss.item(),
                "image shape": x_0.shape,
            })

    def test_one_epoch(self, net_model, args):
        '''
        Sampling from ddpm
        Inputs:
            net_model   : The net in reverse process of DDPM
            args        : Hyper-parameters
        Outputs:
            noisy       : The noise sampled from N(0, 1)
            sampled_img : The image sampled via reverse process of DDPM
        '''
        # load model and evaluate
        with torch.no_grad():
            print("model load weight done.")
            net_model.eval()
            self.ddpm_sampler = GaussianDiffusionSampler(net_model, self.beta_1, self.beta_T, self.T).to(self.device)

            # Sampled from standard normal distribution
            noisy_img = torch.randn(
                size=[args.batch_size, self.in_channel, 32, 32], device=self.device)
            noisy = torch.clamp(noisy_img * 0.5 + 0.5, 0, 1)
            sampled_img = self.ddpm_sampler(noisy_img)

            return noisy, sampled_img

    def load(self, model_path):
        """
        load model
        Inputs:
            model_path : [str] the path to load the model;
        """
        assert os.path.exists(model_path), 'Path Error: can not find the path to load the model'
        # Load the model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])

    def save(self, model_path: str = '../../save_models'):
        """
        save model
        Inputs:
            model_path : [str] the path to save the model, default '../save_models/DDPM.pth';
        """
        # Save the model
        torch.save({'state_dict': self.state_dict()}, model_path + '/' + self._model_name + '.pth')
        print('model has been saved by ' + model_path + '/' + self._model_name + '.pth')


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        '''
        Guassian DDPM Trainer
        Inputs:
            model   : Network in reverse process of DDPM
            beta_1  : The level of noise in first steps of forward process
            beta_T  : The level of noise in T-th steps of forward process
            T       : Number of time steps in DDPM

        Attributes:
            TODO
        '''
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        '''
        Guassian DDPM Trainer
        Inputs:
            model   : Network in reverse process of DDPM
            beta_1  : The level of noise in first step of forward process
            beta_T  : The level of noise in T-th step of forward process
            T       : Number of time steps in DDPM

        Attributes:
            TODO
        '''
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            if time_step % 100 == 0:
                print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

# the network structure of Unet in diffusion model
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        '''
        Get time embedding
        '''
        super().__init__()

        assert d_model % 2 == 0

        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]

        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]

        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, temb):
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_channel):
        '''
        Attention Block
        '''
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channel)
        self.proj_q = nn.Conv2d(in_channel, in_channel, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_channel, in_channel, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_channel, in_channel, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_channel, in_channel, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    '''
    Residual Block
    '''
    def __init__(self, in_channel, out_channel, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            Swish(),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_channel),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channel),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
        )

        if in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channel, out_channel, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

        if attn:
            self.attn = AttnBlock(out_channel)
        else:
            self.attn = nn.Identity()

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, in_channel, channel, channel_mult, attn, num_res_blocks, dropout):
        '''
        The network of DDPM
        Inputs:
            T               : [int] Number of time steps in DDPM
            channel         : [int] Number of channels in the input image
            channel_mult    : [list] Number of mult-channels
            attn            : [list] Number of attention-blocks
            num_res_block   : [int] Number of residual-blocks
            dropout         : [float] Dropout

        Attributes:
            TODO
        '''
        super().__init__()
        assert all([i < len(channel_mult) for i in attn]), 'attn index out of bound'
        tdim = channel * 4
        self.time_embedding = TimeEmbedding(T, channel, tdim)
        self.in_channel = in_channel
        self.head = nn.Conv2d(self.in_channel, channel, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        channels = [channel]  # record output channelannel when dowmsample for upsample
        now_channel = channel
        for i, mult in enumerate(channel_mult):
            out_channel = channel * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_channel=now_channel, out_channel=out_channel, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_channel = out_channel
                channels.append(now_channel)

            if i != len(channel_mult) - 1:
                self.downblocks.append(DownSample(now_channel))
                channels.append(now_channel)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_channel, now_channel, tdim, dropout, attn=True),
            ResBlock(now_channel, now_channel, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_channel = channel * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_channel=channels.pop() + now_channel, out_channel=out_channel, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_channel = out_channel

            if i != 0:
                self.upblocks.append(UpSample(now_channel))
        assert len(channels) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_channel),
            Swish(),
            nn.Conv2d(now_channel, self.in_channel, 3, stride=1, padding=1)
        )

        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)

        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h

