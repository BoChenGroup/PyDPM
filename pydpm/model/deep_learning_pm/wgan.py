"""
===========================================
WGAN
Wasserstein GAN
Martin Arjovsky, Soumith Chintala, and Leon Bottou,
Publihsed in 2017

===========================================
"""

# Author: Bufeng Ge <20009100138@stu.xidian.edu.cn>, Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm
import os

class WGAN(nn.Module):
    def __init__(self, img_shape, g_z_dim: int=100, g_hid_dims: list=[100, 200, 400, 800], d_hid_dims: list=[200, 100], device='cuda:0'):
        """
        The basic model for GAN
        Inputs:
            img_shape : [tuple] the size of input
            g_z_dim : [int] the noise dimension
            g_latent_dim : [list] generator latent dimension;
            d_latent_dim : [list] discriminator latent dimension
            device : [str] 'cpu' or 'gpu';
        """
        super(WGAN, self).__init__()
        setattr(self, '_model_name', 'WGAN')
        self.img_shape = img_shape
        self.g_z_dim = g_z_dim
        self.g_hid_dims = g_hid_dims
        self.d_hid_dims = d_hid_dims
        self.generator = Generator(img_shape=self.img_shape, z_dim=self.g_z_dim, hid_dims=self.g_hid_dims).to(device)
        self.discriminator = Discriminator(img_shape=self.img_shape, hid_dims=self.d_hid_dims).to(device)
        self.device = device
        self.Tensor = torch.FloatTensor if self.device == 'cpu' else torch.cuda.FloatTensor


    def sample(self, batch_size):
        """
        Sample from generator
        Inputs:
            batch_size : [int] number of img which you want;
        Outputs:
            gen_imgs : [tensor] a batch of images
        """
        # Sample noise as generator input
        z = torch.tensor(np.random.normal(0, 1, (batch_size, self.g_z_dim))).to(self.device)
        # Generate a batch of images
        gen_imgs = self.generator(z)
        return gen_imgs


    def train_one_epoch(self, args, model_opt_G, model_opt_D, dataloader, epoch, n_epochs):
        '''
        Train for one epoch
        Inputs:
            model_opt_G     : Optimizer for generator
            model_opt_D     : Optimizer for discriminator
            dataloader      : Train dataset with form of dataloader
            sample_interval : interval betwen image samples while training
            epoch           : Current epoch on training stage
            n_epoch         : Total number of epochs on training stage
        '''
        G_loss_t, D_loss_t = 0, 0
        batches_done = 0
        train_bar = tqdm(iterable=dataloader)
        for i, (imgs, _) in enumerate(train_bar):
            train_bar.set_description(f'Epoch [{epoch}/{n_epochs}]')
            train_bar.set_postfix(G_loss=G_loss_t / (i / args.n_critic + 1), D_loss=D_loss_t / (i + 1))

            # Adversarial ground truths
            real_imgs = Variable(imgs.type(self.Tensor))

            model_opt_D.zero_grad()

            # Sample noise as generator input
            fake_imgs = self.sample(imgs.shape[0])
            # z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.g_z_dim))))
            # # Generate a batch of images
            # fake_imgs = self.generator(z)
            # Adversarial loss
            loss_D = -torch.mean(self.discriminator(real_imgs)) + torch.mean(self.discriminator(fake_imgs))

            loss_D.backward()
            model_opt_D.step()
            D_loss_t += -loss_D.item()

            # Train Generator
            # Sample noise as generator input
            z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.g_z_dim))))
            # Generate a batch of images
            gen_imgs = self.generator(z)

            # Clip weights of discriminator
            for p in self.discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)

            if i % args.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

                model_opt_G.zero_grad()

                # Generate a batch of images
                gen_imgs = self.generator(z)
                # Adversarial loss
                loss_G = -torch.mean(self.discriminator(gen_imgs))

                loss_G.backward()
                model_opt_G.step()
                G_loss_t += -loss_G.item()

            if batches_done % args.sample_interval == 0:
                save_image(gen_imgs.data[:25], "../../output/images/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += 1



    def save(self, model_path: str = '../save_models'):
        """
        save model
        Inputs:
            model_path : [str] the path to save the model, default '../save_models/GAN.pth';
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

class Generator(nn.Module):
    def __init__(self, img_shape, z_dim: int=100, hid_dims: list=[100, 200, 400, 800]):
        """
        The basic model for generator
        Inputs:
            img_shape : [tuple] the size of input
            latent_dim : [int] generator latent dimension;
            z_dim : [int] the noise dimension
        """
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.z_dim = z_dim
        self.hid_dims = hid_dims
        self.num_layers = len(self.hid_dims)
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.Block = nn.ModuleList()
        for layer_index in range(self.num_layers):
            if layer_index == 0:
                latent_layers = block(self.z_dim, self.hid_dims[layer_index], normalize=False)
            else:
                latent_layers = block(self.hid_dims[layer_index - 1], self.hid_dims[layer_index])
            for i in range(len(latent_layers)):
                self.Block.append(latent_layers[i])

        self.model = nn.Sequential(
            *self.Block,
            nn.Linear(self.hid_dims[-1], int(np.prod(self.img_shape))),
            nn.Tanh(),
        )


    def forward(self, z):
        """
        Forward process of generator
        Inputs:
            z : [tuple] the size of input
        Outputs；
            img : [tensor] return the generated image
        """
        z = z.float()
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape, hid_dims: list=[512, 256]):
        """
        The basic model for discriminator
        Inputs:
            img_shape : [tuple] the size of input
            latent_dim : [int] discriminator latent dimension;
        """
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.hid_dims = hid_dims
        self.num_layers = len(self.hid_dims)
        self.Block = nn.ModuleList()

        for layer_index in range(self.num_layers):
            if layer_index == 0:
                self.Block.append(nn.Linear(int(np.prod(self.img_shape)), self.hid_dims[layer_index]))
            else:
                self.Block.append(nn.Linear(self.hid_dims[layer_index - 1], self.hid_dims[layer_index]))
            self.Block.append(nn.LeakyReLU(0.2, inplace=True))

        self.Block.append(nn.Linear(self.hid_dims[-1], 1))

        self.model = nn.Sequential(
            *self.Block,
        )

    def forward(self, img):
        """
        Forward process of discriminator
        Inputs:
            img : [tensor] input image
        Outputs；
            validity : [tensor] return the validity of the input image
        """
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity