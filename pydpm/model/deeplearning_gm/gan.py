"""
===========================================
GAN
Generative Adversarial Networks
IJ Goodfellow，J Pouget-Abadie，M Mirza，B Xu，D Warde-Farley，S Ozair，A Courville，Y Bengio
Publihsed in 2014

===========================================
"""

# Author: Muyao Wang <flare935694542@163.com>, Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm
import os

class GAN(nn.Module):
    def __init__(self, img_shape, g_latent_dim=100, z_dim=128, d_latent_dim=512, device='cuda:0'):
        """
        The basic model for GAN
        Inputs:
            img_shape : [tuple] the size of input
            g_latent_dim : [int] generator latent dimension;
            z_dim : [int] the noise dimension
            d_latent_dim : [int] discriminator latent dimension
            device : [str] 'cpu' or 'gpu';
        """
        super(GAN, self).__init__()
        setattr(self, '_model_name', 'GAN')
        self.img_shape = img_shape
        self.g_latent_dim = g_latent_dim
        self.d_latent_dim = d_latent_dim
        self.generator = Generator(latent_dim = g_latent_dim, z_dim = z_dim, img_shape = self.img_shape).to(device)
        self.discriminator = Discriminator(latent_dim = d_latent_dim, img_shape = self.img_shape).to(device)
        self.adversarial_loss = torch.nn.BCELoss().to(device)
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
        z = torch.tensor(np.random.normal(0, 1, (batch_size, self.g_latent_dim))).to(self.device)
        # Generate a batch of images
        gen_imgs = self.generator(z)
        return gen_imgs


    def train_one_epoch(self, model_opt_G, model_opt_D, dataloader, sample_interval, epoch, n_epochs):
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
        g_loss_interval = 4
        train_bar = tqdm(iterable=dataloader)
        for i, (imgs, _) in enumerate(train_bar):
            train_bar.set_description(f'Epoch [{epoch}/{n_epochs}]')
            train_bar.set_postfix(loss=G_loss_t / (i / g_loss_interval + 1), KL_loss=D_loss_t / (i + 1))

            # Adversarial ground truths
            valid = Variable(self.Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)
            fake = Variable(self.Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False).to(self.device)
            real_imgs = Variable(imgs.type(self.Tensor))

            # Train Generator
            # Sample noise as generator input
            z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.g_latent_dim))))
            # Generate a batch of images
            gen_imgs = self.generator(z)
            if (i % g_loss_interval == 0):
                model_opt_G.zero_grad()
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
                g_loss.backward()
                model_opt_G.step()
                G_loss_t += g_loss.item()

            # Train Discriminator
            model_opt_D.zero_grad()
            real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
            fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            model_opt_D.step()
            D_loss_t += d_loss.item()

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], "../output/images/%d.png" % batches_done, nrow=5, normalize=True)


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
    def __init__(self, img_shape, latent_dim = 100, z_dim = 128):
        """
        The basic model for generator
        Inputs:
            img_shape : [tuple] the size of input
            latent_dim : [int] generator latent dimension;
            z_dim : [int] the noise dimension
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.z_dim = z_dim
        self.img_shape = img_shape
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, self.z_dim, normalize=False),
            *block(self.z_dim, self.z_dim*2),
            *block(self.z_dim*2, self.z_dim*4),
            *block(self.z_dim*4, self.z_dim*8),
            nn.Linear(self.z_dim*8, int(np.prod(self.img_shape))),
            nn.Tanh()
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
    def __init__(self, img_shape, latent_dim = 256):
        """
        The basic model for discriminator
        Inputs:
            img_shape : [tuple] the size of input
            latent_dim : [int] discriminator latent dimension;
        """
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), self.latent_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_dim*2, self.latent_dim ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid(),
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