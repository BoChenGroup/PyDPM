"""
===========================================
VAE
Auto-Encoding Variational Bayes
Diederik P. Kingma， Max Welling
Publihsed in 2014
===========================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, device='cpu'):
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
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.device = device
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1).to(self.device)
        self.fc2 = nn.Linear(h_dim1, h_dim2).to(self.device)
        self.fc31 = nn.Linear(h_dim2, z_dim).to(self.device)
        self.fc32 = nn.Linear(h_dim2, z_dim).to(self.device)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2).to(self.device)
        self.fc5 = nn.Linear(h_dim2, h_dim1).to(self.device)
        self.fc6 = nn.Linear(h_dim1, x_dim).to(self.device)

    def encoder(self, x):
        """
        Encode x to latent distribution
        Inputs:
            x : [tensor] the input tensor;
        Outputs:
            mu : [tensor] mean of posterior distribution;
            log_var : [tensor] log variance of posterior distribution;
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
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

    def decoder(self, z):
        """
        Reconstruct from the z
        Inputs:
            z : [tensor] the sample of posterior distribution;
        Outputs:
            recon_x : [tensor] the reconstruction of x
        """
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))  # recon_x

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
        mu, log_var = self.encoder(x.view(x.shape[0],-1))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    def sample(self, batch_size):
        """
        Sample from generator
        Inputs:
            batch_size : [int] number of img which you want;
        Outputs:
            recon_x : [tensor] reconstruction of x
        """
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        recon_x = self.decoder(z)
        return recon_x
    def show(self):
        """
        Sample from generator
        Inputs:
            batch_size : [int] number of img which you want;
        Outputs:
            recon_x : [tensor] reconstruction of x
        """
        x, y = torch.meshgrid([torch.arange(-3, 3, 0.5), torch.arange(-3, 3, 0.5)])
        z = torch.stack((x, y), 2).view(x.shape[0]**2, 2).to(self.device)
        # z = torch.randn(batch_size, self.z_dim).to(self.device)
        recon_x = self.decoder(z)
        # print(recon_x.shape)
        return recon_x
    def save(self, epoch, checkpoints):
        """
        save model
        Inputs:
            epoch     : [int] train epoch;
            checkpoints : [str] trained model path;
        """
        '======>>saving'
        torch.save({'epoch': epoch + 1,
                    'state_dict': self.state_dict()},
                   checkpoints + '_epochs{}.pth'.format(epoch + 1))

    def load(self, start_ep, checkpoints):
        """
        load model
        Inputs:
            start_ep : [int] the epoch of checkpoint;
            checkpoints : [str] trained model path;
        """
        try:
            print("Loading Chechpoint from ' {} '".format(checkpoints + '_epochs{}.pth'.format(start_ep)))
            checkpoint = torch.load(checkpoints + '_epochs{}.pth'.format(start_ep))
            self.start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['state_dict'])
            print("Resuming Training From Epoch {}".format(self.start_epoch))
            self.start_epoch = 0
        except:
            print("No Checkpoint Exists At '{}'".format(checkpoints))
            self.start_epoch = 0