"""
===========================================
RealNVP for images
DENSITY ESTIMATION USING REAL NVP
Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
Publihsed in ICLR 2017

Reference code can be found in https://github.com/fmu2/realNVP
===========================================
"""

# Author: Bufeng Ge <20009100138@stu.xidian.edu.cn>, Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
import numpy as np
from tqdm import tqdm
import torch.distributions as distributions
import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms

class WeightNormConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, 
        bias=True, weight_norm=True, scale=False):
        """Intializes a Conv2d augmented with weight normalization.

        (See torch.nn.utils.weight_norm for detail.)

        Args:
            in_dim: number of input channels.
            out_dim: number of output channels.
            kernel_size: size of convolving kernel.
            stride: stride of convolution.
            padding: zero-padding added to both sides of input.
            bias: True if include learnable bias parameters, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            scale: True if include magnitude parameters, False otherwise.
        """
        super(WeightNormConv2d, self).__init__()

        if weight_norm:
            self.conv = nn.utils.weight_norm(
                nn.Conv2d(in_dim, out_dim, kernel_size, 
                    stride=stride, padding=padding, bias=bias))
            if not scale:
                self.conv.weight_g.data = torch.ones_like(self.conv.weight_g.data)
                self.conv.weight_g.requires_grad = False    # freeze scaling
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, 
                stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, bottleneck, weight_norm):
        """Initializes a ResidualBlock.

        Args:
            dim: number of input and output features.
            bottleneck: True if use bottleneck, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
        """
        super(ResidualBlock, self).__init__()
        
        self.in_block = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU())
        if bottleneck:
            self.res_block = nn.Sequential(
                WeightNormConv2d(dim, dim, (1, 1), stride=1, padding=0, 
                    bias=False, weight_norm=weight_norm, scale=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1, 
                    bias=False, weight_norm=weight_norm, scale=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                WeightNormConv2d(dim, dim, (1, 1), stride=1, padding=0, 
                    bias=True, weight_norm=weight_norm, scale=True))
        else:
            self.res_block = nn.Sequential(
                WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1, 
                    bias=False, weight_norm=weight_norm, scale=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1, 
                    bias=True, weight_norm=weight_norm, scale=True))

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return x + self.res_block(self.in_block(x))

class ResidualModule(nn.Module):
    def __init__(self, in_dim, dim, out_dim, 
        res_blocks, bottleneck, skip, weight_norm):
        """Initializes a ResidualModule.

        Args:
            in_dim: number of input features.
            dim: number of features in residual blocks.
            out_dim: number of output features.
            res_blocks: number of residual blocks to use.
            bottleneck: True if use bottleneck, False otherwise.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
        """
        super(ResidualModule, self).__init__()
        self.res_blocks = res_blocks
        self.skip = skip
        
        if res_blocks > 0:
            self.in_block = WeightNormConv2d(in_dim, dim, (3, 3), stride=1, 
                padding=1, bias=True, weight_norm=weight_norm, scale=False)
            self.core_block = nn.ModuleList(
                [ResidualBlock(dim, bottleneck, weight_norm) 
                for _ in range(res_blocks)])
            self.out_block = nn.Sequential(
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                WeightNormConv2d(dim, out_dim, (1, 1), stride=1, padding=0, 
                    bias=True, weight_norm=weight_norm, scale=True))
        
            if skip:
                self.in_skip = WeightNormConv2d(dim, dim, (1, 1), stride=1, 
                    padding=0, bias=True, weight_norm=weight_norm, scale=True)
                self.core_skips = nn.ModuleList(
                    [WeightNormConv2d(
                        dim, dim, (1, 1), stride=1, padding=0, bias=True, 
                        weight_norm=weight_norm, scale=True) 
                    for _ in range(res_blocks)])
        else:
            if bottleneck:
                self.block = nn.Sequential(
                    WeightNormConv2d(in_dim, dim, (1, 1), stride=1, padding=0, 
                        bias=False, weight_norm=weight_norm, scale=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                    WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1, 
                        bias=False, weight_norm=weight_norm, scale=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                    WeightNormConv2d(dim, out_dim, (1, 1), stride=1, padding=0, 
                        bias=True, weight_norm=weight_norm, scale=True))
            else:
                self.block = nn.Sequential(
                    WeightNormConv2d(in_dim, dim, (3, 3), stride=1, padding=1, 
                        bias=False, weight_norm=weight_norm, scale=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                    WeightNormConv2d(dim, out_dim, (3, 3), stride=1, padding=1, 
                        bias=True, weight_norm=weight_norm, scale=True))

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        if self.res_blocks > 0:
            x = self.in_block(x)
            if self.skip:
                out = self.in_skip(x)
            for i in range(len(self.core_block)):
                x = self.core_block[i](x)
                if self.skip:
                    out = out + self.core_skips[i](x)
            if self.skip:
                x = out
            return self.out_block(x)
        else:
            return self.block(x)

class AbstractCoupling(nn.Module):
    def __init__(self, mask_config, hps):
        """Initializes an AbstractCoupling.

        Args:
            mask_config: mask configuration (see build_mask() for more detail).
            hps: the set of hyperparameters.
        """
        super(AbstractCoupling, self).__init__()
        self.mask_config = mask_config
        self.res_blocks = hps.res_blocks
        self.bottleneck = hps.bottleneck
        self.skip = hps.skip
        self.weight_norm = hps.weight_norm
        self.coupling_bn = hps.coupling_bn

    def build_mask(self, size, config=1.):
        """Builds a binary checkerboard mask.

        (Only for constructing masks for checkerboard coupling layers.)

        Args:
            size: height/width of features.
            config: mask configuration that determines which pixels to mask up.
                    if 1:        if 0:
                        1 0         0 1
                        0 1         1 0
        Returns:
            a binary mask (1: pixel on, 0: pixel off).
        """
        mask = np.arange(size).reshape(-1, 1) + np.arange(size)
        mask = np.mod(config + mask, 2)
        mask = mask.reshape(-1, 1, size, size)
        return torch.tensor(mask.astype('float32'))

    def batch_stat(self, x):
        """Compute (spatial) batch statistics.

        Args:
            x: input minibatch.
        Returns:
            batch mean and variance.
        """
        mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        var = torch.mean((x - mean)**2, dim=(0, 2, 3), keepdim=True)
        return mean, var

class CheckerboardAdditiveCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, mid_dim, size, mask_config, hps):
        """Initializes a CheckerboardAdditiveCoupling.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            size: height/width of features.
            mask_config: mask configuration (see build_mask() for more detail).
            hps: the set of hyperparameters.
        """
        super(CheckerboardAdditiveCoupling, self).__init__(mask_config, hps)
        
        self.mask = self.build_mask(size, config=mask_config).cuda()
        self.in_bn = nn.BatchNorm2d(in_out_dim)
        self.block = nn.Sequential(
            nn.ReLU(),
            ResidualModule(2*in_out_dim+1, mid_dim, in_out_dim, 
                self.res_blocks, self.bottleneck, self.skip, self.weight_norm))
        self.out_bn = nn.BatchNorm2d(in_out_dim, affine=False)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        [B, _, _, _] = list(x.size())
        mask = self.mask.repeat(B, 1, 1, 1)
        x_ = self.in_bn(x * mask)
        x_ = torch.cat((x_, -x_), dim=1)
        x_ = torch.cat((x_, mask), dim=1)     # 2C+1 channels
        shift = self.block(x_) * (1. - mask)

        log_diag_J = torch.zeros_like(x)     # unit Jacobian determinant
        # See Eq(3) and Eq(4) in NICE and Section 3.7 in real NVP
        if reverse:
            if self.coupling_bn:
                mean, var = self.out_bn.running_mean, self.out_bn.running_var
                mean = mean.reshape(-1, 1, 1, 1).transpose(0, 1)
                var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
                x = x * torch.exp(0.5 * torch.log(var + 1e-5) * (1. - mask)) \
                    + mean * (1. - mask)
            x = x - shift
        else:
            x = x + shift
            if self.coupling_bn:
                if self.training:
                    _, var = self.batch_stat(x)
                else:
                    var = self.out_bn.running_var
                    var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
                x = self.out_bn(x) * (1. - mask) + x * mask
                log_diag_J = log_diag_J - 0.5 * torch.log(var + 1e-5) * (1. - mask)
        return x, log_diag_J

class CheckerboardAffineCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, mid_dim, size, mask_config, hps):
        """Initializes a CheckerboardAffineCoupling.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            size: height/width of features.
            mask_config: mask configuration (see build_mask() for more detail).
            hps: the set of hyperparameters.
        """
        super(CheckerboardAffineCoupling, self).__init__(mask_config, hps)

        self.mask = self.build_mask(size, config=mask_config).cuda()
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.in_bn = nn.BatchNorm2d(in_out_dim)
        self.block = nn.Sequential(        # 1st half of resnet: shift
            nn.ReLU(),                    # 2nd half of resnet: log_rescale
            ResidualModule(2*in_out_dim+1, mid_dim, 2*in_out_dim, 
                self.res_blocks, self.bottleneck, self.skip, self.weight_norm))
        self.out_bn = nn.BatchNorm2d(in_out_dim, affine=False)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        [B, C, _, _] = list(x.size())
        mask = self.mask.repeat(B, 1, 1, 1)
        x_ = self.in_bn(x * mask)
        x_ = torch.cat((x_, -x_), dim=1)
        x_ = torch.cat((x_, mask), dim=1)    # 2C+1 channels
        (shift, log_rescale) = self.block(x_).split(C, dim=1)
        log_rescale = self.scale * torch.tanh(log_rescale) + self.scale_shift
        shift = shift * (1. - mask)
        log_rescale = log_rescale * (1. - mask)
        
        log_diag_J = log_rescale     # See Eq(6) in real NVP 
        # See Eq(7) and Eq(8) and Section 3.7 in real NVP
        if reverse:
            if self.coupling_bn:
                mean, var = self.out_bn.running_mean, self.out_bn.running_var
                mean = mean.reshape(-1, 1, 1, 1).transpose(0, 1)
                var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
                x = x * torch.exp(0.5 * torch.log(var + 1e-5) * (1. - mask)) \
                    + mean * (1. - mask)
            x = (x - shift) * torch.exp(-log_rescale)
        else:
            x = x * torch.exp(log_rescale) + shift
            if self.coupling_bn:
                if self.training:
                    _, var = self.batch_stat(x)
                else:
                    var = self.out_bn.running_var
                    var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
                x = self.out_bn(x) * (1. - mask) + x * mask
                log_diag_J = log_diag_J - 0.5 * torch.log(var + 1e-5) * (1. - mask)
        return x, log_diag_J

class CheckerboardCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, size, mask_config, hps):
        """Initializes a CheckerboardCoupling.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            size: height/width of features.
            mask_config: mask configuration (see build_mask() for more detail).
            hps: the set of hyperparameters.
        """
        super(CheckerboardCoupling, self).__init__()

        if hps.affine:
            self.coupling = CheckerboardAffineCoupling(
                in_out_dim, mid_dim, size, mask_config, hps)
        else:
            self.coupling = CheckerboardAdditiveCoupling(
                in_out_dim, mid_dim, size, mask_config, hps)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        return self.coupling(x, reverse)

class ChannelwiseAdditiveCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, mid_dim, mask_config, hps):
        """Initializes a ChannelwiseAdditiveCoupling.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            mask_config: 1 if change the top half, 0 if change the bottom half.
            hps: the set of hyperparameters.
        """
        super(ChannelwiseAdditiveCoupling, self).__init__(mask_config, hps)

        self.in_bn = nn.BatchNorm2d(in_out_dim//2)
        self.block = nn.Sequential(
            nn.ReLU(),
            ResidualModule(in_out_dim, mid_dim, in_out_dim//2, 
                self.res_blocks, self.bottleneck, self.skip, self.weight_norm))
        self.out_bn = nn.BatchNorm2d(in_out_dim//2, affine=False)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        [_, C, _, _] = list(x.size())
        if self.mask_config:
            (on, off) = x.split(C//2, dim=1)
        else:
            (off, on) = x.split(C//2, dim=1)
        off_ = self.in_bn(off)
        off_ = torch.cat((off_, -off_), dim=1)    # C channels
        shift = self.block(off_)
        
        log_diag_J = torch.zeros_like(x)    # unit Jacobian determinant
        # See Eq(3) and Eq(4) in NICE and Section 3.7 in real NVP
        if reverse:
            if self.coupling_bn:
                mean, var = self.out_bn.running_mean, self.out_bn.running_var
                mean = mean.reshape(-1, 1, 1, 1).transpose(0, 1)
                var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
                on = on * torch.exp(0.5 * torch.log(var + 1e-5)) + mean
            on = on - shift
        else:
            on = on + shift
            if self.coupling_bn:
                if self.training:
                    _, var = self.batch_stat(on)
                else:
                    var = self.out_bn.running_var
                    var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
                on = self.out_bn(on)
                log_diag_J = log_diag_J - 0.5 * torch.log(var + 1e-5)
        if self.mask_config:
            x = torch.cat((on, off), dim=1)
        else:
            x = torch.cat((off, on), dim=1)
        return x, log_diag_J

class ChannelwiseAffineCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, mid_dim, mask_config, hps):
        """Initializes a ChannelwiseAffineCoupling.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            mask_config: 1 if change the top half, 0 if change the bottom half.
            hps: the set of hyperparameters.
        """
        super(ChannelwiseAffineCoupling, self).__init__(mask_config, hps)

        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.in_bn = nn.BatchNorm2d(in_out_dim//2)
        self.block = nn.Sequential(        # 1st half of resnet: shift
            nn.ReLU(),                    # 2nd half of resnet: log_rescale
            ResidualModule(in_out_dim, mid_dim, in_out_dim, 
                self.res_blocks, self.bottleneck, self.skip, self.weight_norm))
        self.out_bn = nn.BatchNorm2d(in_out_dim//2, affine=False)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        [_, C, _, _] = list(x.size())
        if self.mask_config:
            (on, off) = x.split(C//2, dim=1)
        else:
            (off, on) = x.split(C//2, dim=1)
        off_ = self.in_bn(off)
        off_ = torch.cat((off_, -off_), dim=1)     # C channels
        out = self.block(off_)
        (shift, log_rescale) = out.split(C//2, dim=1)
        log_rescale = self.scale * torch.tanh(log_rescale) + self.scale_shift
        
        log_diag_J = log_rescale     # See Eq(6) in real NVP
        # See Eq(7) and Eq(8) and Section 3.7 in real NVP
        if reverse:
            if self.coupling_bn:
                mean, var = self.out_bn.running_mean, self.out_bn.running_var
                mean = mean.reshape(-1, 1, 1, 1).transpose(0, 1)
                var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
                on = on * torch.exp(0.5 * torch.log(var + 1e-5)) + mean
            on = (on - shift) * torch.exp(-log_rescale)
        else:
            on = on * torch.exp(log_rescale) + shift
            if self.coupling_bn:
                if self.training:
                    _, var = self.batch_stat(on)
                else:
                    var = self.out_bn.running_var
                    var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
                on = self.out_bn(on)
                log_diag_J = log_diag_J - 0.5 * torch.log(var + 1e-5)
        if self.mask_config:
            x = torch.cat((on, off), dim=1)
            log_diag_J = torch.cat((log_diag_J, torch.zeros_like(log_diag_J)), 
                dim=1)
        else:
            x = torch.cat((off, on), dim=1)
            log_diag_J = torch.cat((torch.zeros_like(log_diag_J), log_diag_J), 
                dim=1)
        return x, log_diag_J

class ChannelwiseCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, mask_config, hps):
        """Initializes a ChannelwiseCoupling.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            mask_config: 1 if change the top half, 0 if change the bottom half.
            hps: the set of hyperparameters.
        """
        super(ChannelwiseCoupling, self).__init__()

        if hps.affine:
            self.coupling = ChannelwiseAffineCoupling(
                in_out_dim, mid_dim, mask_config, hps)
        else:
            self.coupling = ChannelwiseAdditiveCoupling(
                in_out_dim, mid_dim, mask_config, hps)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        return self.coupling(x, reverse)

class RealNVP(nn.Module):
    def __init__(self, datainfo, prior, device, args):
        """Initializes a RealNVP.

        Args:
            datainfo: information of dataset to be modeled.
            prior: prior distribution over latent space Z.
            hps: the set of hyperparameters.
        """
        super(RealNVP, self).__init__()
        setattr(self, '_model_name', 'RealNVP')
        self.datainfo = datainfo
        self.prior = prior
        self.device = device
        self.hps = Hyperparameters(
                base_dim = args.base_dim,
                res_blocks = args.res_blocks,
                bottleneck = args.bottleneck,
                skip = args.skip,
                weight_norm = args.weight_norm,
                coupling_bn = args.coupling_bn,
                affine = args.affine)

        chan = datainfo.channel
        size = datainfo.size
        dim = args.base_dim

        if datainfo.name == 'cifar10':
            # architecture for CIFAR-10 (down to 16 x 16 x C)
            # SCALE 1: 3 x 32 x 32
            self.s1_ckbd = self.checkerboard_combo(chan, dim, size, self.hps)
            self.s1_chan = self.channelwise_combo(chan*4, dim, self.hps)
            self.order_matrix_1 = self.order_matrix(chan).cuda()
            chan *= 2
            size //= 2

            # SCALE 2: 6 x 16 x 16
            self.s2_ckbd = self.checkerboard_combo(chan, dim, size, self.hps, final=True)

        else: # NOTE: can construct with loop (for future edit)
            # architecture for ImageNet and CelebA (down to 4 x 4 x C)
            # SCALE 1: 3 x 32(64) x 32(64)
            self.s1_ckbd = self.checkerboard_combo(chan, dim, size, self.hps)
            self.s1_chan = self.channelwise_combo(chan*4, dim*2, self.hps)
            self.order_matrix_1 = self.order_matrix(chan).cuda()
            chan *= 2
            size //= 2
            dim *= 2

            # SCALE 2: 6 x 16(32) x 16(32)
            self.s2_ckbd = self.checkerboard_combo(chan, dim, size, self.hps)
            self.s2_chan = self.channelwise_combo(chan*4, dim*2, self.hps)
            self.order_matrix_2 = self.order_matrix(chan).cuda()
            chan *= 2
            size //= 2
            dim *= 2

            # SCALE 3: 12 x 8(16) x 8(16)
            self.s3_ckbd = self.checkerboard_combo(chan, dim, size, self.hps)
            self.s3_chan = self.channelwise_combo(chan*4, dim*2, self.hps)
            self.order_matrix_3 = self.order_matrix(chan).cuda()
            chan *= 2
            size //= 2
            dim *= 2

            if datainfo.name == 'imnet32':
                # SCALE 4: 24 x 4 x 4
                self.s4_ckbd = self.checkerboard_combo(chan, dim, size, self.hps, final=True)
            
            elif datainfo.name in ['imnet64', 'celeba']:
                # SCALE 4: 24 x 8 x 8
                self.s4_ckbd = self.checkerboard_combo(chan, dim, size, self.hps)
                self.s4_chan = self.channelwise_combo(chan*4, dim*2, self.hps)
                self.order_matrix_4 = self.order_matrix(chan).cuda()
                chan *= 2
                size //= 2
                dim *= 2

                # SCALE 5: 48 x 4 x 4
                self.s5_ckbd = self.checkerboard_combo(chan, dim, size, self.hps, final=True)

    def checkerboard_combo(self, in_out_dim, mid_dim, size, hps, final=False):
        """Construct a combination of checkerboard coupling layers.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            size: height/width of features.
            hps: the set of hyperparameters.
            final: True if at final scale, False otherwise.
        Returns:
            A combination of checkerboard coupling layers.
        """
        if final:
            return nn.ModuleList([
                CheckerboardCoupling(in_out_dim, mid_dim, size, 1., hps),
                CheckerboardCoupling(in_out_dim, mid_dim, size, 0., hps),
                CheckerboardCoupling(in_out_dim, mid_dim, size, 1., hps),
                CheckerboardCoupling(in_out_dim, mid_dim, size, 0., hps)])
        else:
            return nn.ModuleList([
                CheckerboardCoupling(in_out_dim, mid_dim, size, 1., hps), 
                CheckerboardCoupling(in_out_dim, mid_dim, size, 0., hps),
                CheckerboardCoupling(in_out_dim, mid_dim, size, 1., hps)])
        
    def channelwise_combo(self, in_out_dim, mid_dim, hps):
        """Construct a combination of channelwise coupling layers.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            hps: the set of hyperparameters.
        Returns:
            A combination of channelwise coupling layers.
        """
        return nn.ModuleList([
                ChannelwiseCoupling(in_out_dim, mid_dim, 0., hps),
                ChannelwiseCoupling(in_out_dim, mid_dim, 1., hps),
                ChannelwiseCoupling(in_out_dim, mid_dim, 0., hps)])

    def squeeze(self, x):
        """Squeezes a C x H x W tensor into a 4C x H/2 x W/2 tensor.

        (See Fig 3 in the real NVP paper.)

        Args:
            x: input tensor (B x C x H x W).
        Returns:
            the squeezed tensor (B x 4C x H/2 x W/2).
        """
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C*4, H//2, W//2)
        return x

    def undo_squeeze(self, x):
        """unsqueezes a C x H x W tensor into a C/4 x 2H x 2W tensor.

        (See Fig 3 in the real NVP paper.)

        Args:
            x: input tensor (B x C x H x W).
        Returns:
            the squeezed tensor (B x C/4 x 2H x 2W).
        """
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C//4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C//4, H*2, W*2)
        return x

    def order_matrix(self, channel):
        """Constructs a matrix that defines the ordering of variables
        when downscaling/upscaling is performed.

        Args:
          channel: number of features.
        Returns:
          a kernel for rearrange the variables.
        """
        weights = np.zeros((channel*4, channel, 2, 2))
        ordering = np.array([[[[1., 0.],
                               [0., 0.]]],
                             [[[0., 0.],
                               [0., 1.]]],
                             [[[0., 1.],
                               [0., 0.]]],
                             [[[0., 0.],
                               [1., 0.]]]])
        for i in range(channel):
            s1 = slice(i, i+1)
            s2 = slice(4*i, 4*(i+1))
            weights[s2, s1, :, :] = ordering
        shuffle = np.array([4*i for i in range(channel)]
                         + [4*i+1 for i in range(channel)]
                         + [4*i+2 for i in range(channel)]
                         + [4*i+3 for i in range(channel)])
        weights = weights[shuffle, :, :, :].astype('float32')
        return torch.tensor(weights)

    def factor_out(self, x, order_matrix):
        """Downscales and factors out the bottom half of the tensor.

        (See Fig 4(b) in the real NVP paper.)

        Args:
            x: input tensor (B x C x H x W).
            order_matrix: a kernel that defines the ordering of variables.
        Returns:
            the top half for further transformation (B x 2C x H/2 x W/2)
            and the Gaussianized bottom half (B x 2C x H/2 x W/2).
        """
        x = F.conv2d(x, order_matrix, stride=2, padding=0)
        [_, C, _, _] = list(x.size())
        (on, off) = x.split(C//2, dim=1)
        return on, off

    def restore(self, on, off, order_matrix):
        """Merges variables and restores their ordering.

        (See Fig 4(b) in the real NVP paper.)

        Args:
            on: the active (transformed) variables (B x C x H x W).
            off: the inactive variables (B x C x H x W).
            order_matrix: a kernel that defines the ordering of variables.
        Returns:
            combined variables (B x 2C x H x W).
        """
        x = torch.cat((on, off), dim=1)
        return F.conv_transpose2d(x, order_matrix, stride=2, padding=0)

    def g(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, x_off_1 = self.factor_out(z, self.order_matrix_1)

        if self.datainfo.name in ['imnet32', 'imnet64', 'celeba']:
            x, x_off_2 = self.factor_out(x, self.order_matrix_2)
            x, x_off_3 = self.factor_out(x, self.order_matrix_3)

            if self.datainfo.name in ['imnet64', 'celeba']:
                x, x_off_4 = self.factor_out(x, self.order_matrix_4)

                # SCALE 5: 4 x 4
                for i in reversed(range(len(self.s5_ckbd))):
                    x, _ = self.s5_ckbd[i](x, reverse=True)
                
                x = self.restore(x, x_off_4, self.order_matrix_4)

                # SCALE 4: 8 x 8
                x = self.squeeze(x)
                for i in reversed(range(len(self.s4_chan))):
                    x, _ = self.s4_chan[i](x, reverse=True)
                x = self.undo_squeeze(x)

            for i in reversed(range(len(self.s4_ckbd))):
                x, _ = self.s4_ckbd[i](x, reverse=True)

            x = self.restore(x, x_off_3, self.order_matrix_3)

            # SCALE 3: 8(16) x 8(16)
            x = self.squeeze(x)
            for i in reversed(range(len(self.s3_chan))):
                x, _ = self.s3_chan[i](x, reverse=True)
            x = self.undo_squeeze(x)

            for i in reversed(range(len(self.s3_ckbd))):
                x, _ = self.s3_ckbd[i](x, reverse=True)

            x = self.restore(x, x_off_2, self.order_matrix_2)

            # SCALE 2: 16(32) x 16(32)
            x = self.squeeze(x)
            for i in reversed(range(len(self.s2_chan))):
                x, _ = self.s2_chan[i](x, reverse=True)
            x = self.undo_squeeze(x)

        for i in reversed(range(len(self.s2_ckbd))):
            x, _ = self.s2_ckbd[i](x, reverse=True)

        x = self.restore(x, x_off_1, self.order_matrix_1)

        # SCALE 1: 32(64) x 32(64)
        x = self.squeeze(x)
        for i in reversed(range(len(self.s1_chan))):
            x, _ = self.s1_chan[i](x, reverse=True)
        x = self.undo_squeeze(x)

        for i in reversed(range(len(self.s1_ckbd))):
            x, _ = self.s1_ckbd[i](x, reverse=True)

        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        z, log_diag_J = x, torch.zeros_like(x)

        # SCALE 1: 32(64) x 32(64)
        for i in range(len(self.s1_ckbd)):
            z, inc = self.s1_ckbd[i](z)
            log_diag_J = log_diag_J + inc

        z, log_diag_J = self.squeeze(z), self.squeeze(log_diag_J)
        for i in range(len(self.s1_chan)):
            z, inc = self.s1_chan[i](z)
            log_diag_J = log_diag_J + inc
        z, log_diag_J = self.undo_squeeze(z), self.undo_squeeze(log_diag_J)

        z, z_off_1 = self.factor_out(z, self.order_matrix_1)
        log_diag_J, log_diag_J_off_1 = self.factor_out(log_diag_J, self.order_matrix_1)

        # SCALE 2: 16(32) x 16(32)
        for i in range(len(self.s2_ckbd)):
            z, inc = self.s2_ckbd[i](z)
            log_diag_J = log_diag_J + inc

        if self.datainfo.name in ['imnet32', 'imnet64', 'celeba']:
            z, log_diag_J = self.squeeze(z), self.squeeze(log_diag_J)
            for i in range(len(self.s2_chan)):
                z, inc = self.s2_chan[i](z)
                log_diag_J = log_diag_J + inc
            z, log_diag_J = self.undo_squeeze(z), self.undo_squeeze(log_diag_J)

            z, z_off_2 = self.factor_out(z, self.order_matrix_2)
            log_diag_J, log_diag_J_off_2 = self.factor_out(log_diag_J, self.order_matrix_2)

            # SCALE 3: 8(16) x 8(16)
            for i in range(len(self.s3_ckbd)):
                z, inc = self.s3_ckbd[i](z)
                log_diag_J = log_diag_J + inc

            z, log_diag_J = self.squeeze(z), self.squeeze(log_diag_J)
            for i in range(len(self.s3_chan)):
                z, inc = self.s3_chan[i](z)
                log_diag_J = log_diag_J + inc
            z, log_diag_J = self.undo_squeeze(z), self.undo_squeeze(log_diag_J)

            z, z_off_3 = self.factor_out(z, self.order_matrix_3)
            log_diag_J, log_diag_J_off_3 = self.factor_out(log_diag_J, self.order_matrix_3)

            # SCALE 4: 4(8) x 4(8)
            for i in range(len(self.s4_ckbd)):
                z, inc = self.s4_ckbd[i](z)
                log_diag_J = log_diag_J + inc

            if self.datainfo.name in ['imnet64', 'celeba']:
                z, log_diag_J = self.squeeze(z), self.squeeze(log_diag_J)
                for i in range(len(self.s4_chan)):
                    z, inc = self.s4_chan[i](z)
                    log_diag_J = log_diag_J + inc
                z, log_diag_J = self.undo_squeeze(z), self.undo_squeeze(log_diag_J)

                z, z_off_4 = self.factor_out(z, self.order_matrix_4)
                log_diag_J, log_diag_J_off_4 = self.factor_out(log_diag_J, self.order_matrix_4)

                # SCALE 5: 4 x 4
                for i in range(len(self.s5_ckbd)):
                    z, inc = self.s5_ckbd[i](z)
                    log_diag_J = log_diag_J + inc

                z = self.restore(z, z_off_4, self.order_matrix_4)
                log_diag_J = self.restore(log_diag_J, log_diag_J_off_4, self.order_matrix_4)

            z = self.restore(z, z_off_3, self.order_matrix_3)
            z = self.restore(z, z_off_2, self.order_matrix_2)
            log_diag_J = self.restore(log_diag_J, log_diag_J_off_3, self.order_matrix_3)
            log_diag_J = self.restore(log_diag_J, log_diag_J_off_2, self.order_matrix_2)
        
        z = self.restore(z, z_off_1, self.order_matrix_1)
        log_diag_J = self.restore(log_diag_J, log_diag_J_off_1, self.order_matrix_1)

        return z, log_diag_J

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Eq(2) and Eq(3) in the real NVP paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_diag_J = self.f(x)
        log_det_J = torch.sum(log_diag_J, dim=(1, 2, 3))
        log_prior_prob = torch.sum(self.prior.log_prob(z), dim=(1, 2, 3))
        return log_prior_prob + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        C = self.datainfo.channel
        H = W = self.datainfo.size
        z = self.prior.sample((size, C, H, W))
        return self.g(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input and sum of squares of scaling factors.
            (the latter is used in L2 regularization.)
        """
        weight_scale = None
        for name, param in self.named_parameters():
            param_name = name.split('.')[-1]
            if param_name in ['weight_g', 'scale'] and param.requires_grad:
                if weight_scale is None:
                    weight_scale = torch.pow(param, 2).sum()
                else:
                    weight_scale = weight_scale + torch.pow(param, 2).sum()
        return self.log_prob(x), weight_scale

    # train and test
    def train_one_epoch(self, dataloader, model_opt, epoch, args):
        '''
        Train for one epoch
        Inputs:
            model_opt  : Optimizer for model
            dataloader : Train dataset with form of dataloader
            epoch      : Current epoch on training stage
            args       : Argument dict

        '''
        self.train()
        loss_mean, log_ll_mean, bpd_mean = 0., 0., 0.
        train_bar = tqdm(iterable=dataloader)
        for i, data in enumerate(train_bar):
            train_bar.set_description(f'Epoch [{epoch}/{args.num_epochs}]')
            train_bar.set_postfix(loss=loss_mean / (i + 1), log_ll=log_ll_mean / (i + 1), bpd=bpd_mean)

            x, _ = data
            # log-determinant of Jacobian from the logit transform
            x, log_det = logit_transform(x)
            x = x.to(self.device)
            log_det = log_det.to(self.device)

            # log-likelihood of input minibatch
            log_ll, weight_scale = self(x)
            log_ll = (log_ll + log_det).mean()

            # add L2 regularization on scaling factors
            loss = -log_ll + args.scale_reg * weight_scale
            loss_mean = (loss_mean + loss.item()) / (i + 1)
            log_ll_mean = (log_ll_mean + log_ll.item()) / (i + 1)

            loss.backward()
            model_opt.step()
            model_opt.zero_grad()

            bpd_mean = (-log_ll_mean + np.log(256.) * args.image_size) / (args.image_size * np.log(2.))

        return log_ll_mean

    def test_one_epoch(self, dataloader, epoch, args):
        '''
        Test for one epoch
        Inputs:
            dataloader : Train dataset with form of dataloader
            epoch      : Current epoch on testing stage for saving model
            args       : Argument dict dataset
        '''
        self.eval()
        loss_mean, log_ll_mean, bpd_mean = 0., 0., 0.
        test_bar = tqdm(iterable=dataloader)
        with torch.no_grad():
            for i, data in enumerate(test_bar):
                test_bar.set_description(f'Testing stage: ')
                test_bar.set_postfix(loss=loss_mean / (i + 1), log_ll=log_ll_mean / (i + 1), bpd=bpd_mean)

                x, _ = data
                # log-determinant of Jacobian from the logit transform
                x, log_det = logit_transform(x)
                x = x.to(self.device)
                log_det = log_det.to(self.device)

                # log-likelihood of input minibatch
                log_ll, weight_scale = self(x)
                log_ll = (log_ll + log_det).mean()

                # add L2 regularization on scaling factors
                loss = -log_ll + args.scale_reg * weight_scale
                loss_mean = (loss_mean + loss.item()) / (i + 1)
                log_ll_mean = (log_ll_mean + log_ll.item()) / (i + 1)

                bpd_mean = (-log_ll_mean + np.log(256.) * args.image_size) / (args.image_size * np.log(2.))

                samples = self.sample(args.sample_size)
                samples, _ = logit_transform(samples, reverse=True)
                utils.save_image(utils.make_grid(samples),
                    '../../output/images/' + self.datainfo.name + '_ep{}.png'.format(epoch))

        return log_ll_mean

    def save(self, model_path: str = '../save_models'):
        """
        save model
        Inputs:
            model_path : [str] the path to save the model, default '../save_models/realNVP.pth';
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

class Hyperparameters():
    def __init__(self, base_dim, res_blocks, bottleneck,
        skip, weight_norm, coupling_bn, affine):
        """Instantiates a set of hyperparameters used for constructing layers.

        Args:
            base_dim: features in residual blocks of first few layers.
            res_blocks: number of residual blocks to use.
            bottleneck: True if use bottleneck, False otherwise.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if batchnorm coupling layer output, False otherwise.
            affine: True if use affine coupling, False if use additive coupling.
        """
        self.base_dim = base_dim
        self.res_blocks = res_blocks
        self.bottleneck = bottleneck
        self.skip = skip
        self.weight_norm = weight_norm
        self.coupling_bn = coupling_bn
        self.affine = affine
        
class DataInfo():
    def __init__(self, name, channel, size):
        """Instantiates a DataInfo.

        Args:
            name: name of dataset.
            channel: number of image channels.
            size: height and width of an image.
        """
        self.name = name
        self.channel = channel
        self.size = size

def logit_transform(x, constraint=0.9, reverse=False):
    '''Transforms data from [0, 1] into unbounded space.

    Restricts data into [0.05, 0.95].
    Calculates logit(alpha+(1-alpha)*x).

    Args:
        x: input tensor.
        constraint: data constraint before logit.
        reverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
        (if reverse=True, no log-determinant is returned.)
    '''
    if reverse:
        x = 1. / (torch.exp(-x) + 1.)    # [0.05, 0.95]
        x *= 2.             # [0.1, 1.9]
        x -= 1.             # [-0.9, 0.9]
        x /= constraint     # [-1, 1]
        x += 1.             # [0, 2]
        x /= 2.             # [0, 1]
        return x, 0
    else:
        [B, C, H, W] = list(x.size())
        
        # dequantization
        noise = distributions.Uniform(0., 1.).sample((B, C, H, W))
        x = (x * 255. + noise) / 256.
        
        # restrict data
        x *= 2.             # [0, 2]
        x -= 1.             # [-1, 1]
        x *= constraint     # [-0.9, 0.9]
        x += 1.             # [0.1, 1.9]
        x /= 2.             # [0.05, 0.95]

        # logit data
        logit_x = torch.log(x) - torch.log(1. - x)

        # log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(
            np.log(constraint) - np.log(1. - constraint))
        log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
            - F.softplus(-pre_logit_scale)

        return logit_x, torch.sum(log_diag_J, dim=(1, 2, 3))