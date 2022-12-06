"""
===========================================
RBM
A Practical Guide to Training
Restricted Boltzmann Machines
Geoffrey Hinton
Publihsed in 2010
===========================================
"""
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class RBM(nn.Module):
    def __init__(self, n_vis=784, n_hin=500, k=5):
        """
        The basic model for VAE
        Inputs:
            n_vis : [int] number of visible units;
            n_hin : [int] number of latent units;
            k : [int] layers of RBM;
        """
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin, n_vis) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k

    def sample_from_p(self, p):
        """
        Sample from p distribution
        Inputs:
            p : [tensor] distribution of p;
        Outputs:
            sample of p :[tensor] sample of p;
        """
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))

    def v_to_h(self, v):
        """
        propagation from v to h
        Inputs:
            v : [tensor] distribution of v;
        Outputs:
            p_h : [tensor] prediction of h;
            sample_h : [tensor] sample of h;
        """
        p_h = F.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        """
        propagation from h to v
        Inputs:
            h : [tensor] distribution of h;
        Outputs:
            p_v : [tensor] prediction of v;
            sample_v : [tensor] sample of v;
        """
        p_v = F.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, v):
        """
         Forward process of RBM
         Inputs:
             v : [tensor] input of data;
         Outputs:
             v : [tensor] input of data;
             v_ : [tensor] prediction of v;
         """
        pre_h1, h1 = self.v_to_h(v)

        h_ = h1
        for _ in range(self.k):
            pre_v_, v_ = self.h_to_v(h_)
            pre_h_, h_ = self.v_to_h(v_)

        return v, v_

    def free_energy(self, v):
        """
        Free energy of RBM
        Inputs:
            v : [tensor] distribution of v;
        Outputs:
            free_energy : [tensor] free energy of whole system
        """
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

    def save(self, epoch, checkpoints):
        """
        save model
        Inputs:
            epoch     : [int] train epoch;
            checkpoints : [str] trained model path;
        """
        '======>>saving'
        torch.save({'epoch': epoch ,
                    'state_dict': self.state_dict()},
                   checkpoints + '_epochs{}.pth'.format(epoch))

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