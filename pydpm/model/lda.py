"""
===========================================
Latent Dirichlet Allocation
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import numpy as np
from pydpm.utils import Model_Sampler_CPU
import time
from pydpm.utils.Metric import *

class LDA(object):
    def __init__(self, K, device='gpu'):

        self.K = K
        if device == 'gpu':
            self.device = 'gpu'
            from pydpm.utils import Model_Sampler_GPU
            from pydpm import distribution
            self.Multrnd_Matrix = Model_Sampler_GPU.Multrnd_Matrix_GPU
            self.Crt_Matrix = Model_Sampler_GPU.Crt_Matrix_GPU
            self.Crt_Multirnd_Matrix = Model_Sampler_GPU.Crt_Multirnd_Matrix_GPU
            self.gamma=distribution.gamma
            self.dirichlect=distribution.dirichlet
        else:
            raise Exception('device type error')

    def initial(self, data):
        self.data=data
        self.V = data.shape[0]
        self.N = data.shape[1]
        self.Phi=np.random.rand(self.V, self.K)
        self.Phi=self.Phi/np.sum(self.Phi,axis=0)
        self.Theta=np.ones([self.K, self.N]) / self.K
        self.eta = 0.05
        self.r_k= np.ones([self.K,1])*50/self.K

    def train(self, iter_all=100):
        import time
        for iter in range(iter_all):
            start_time = time.time()
            ZSDS, WSZS = self.Multrnd_Matrix(self.data, self.Phi, self.Theta)
            if iter <= 50:
                self.Theta = self.gamma(ZSDS + self.r_k.repeat(self.N, axis=1), 0.5)
                self.Phi = np.transpose(self.dirichlect(np.transpose(WSZS + self.eta)))
            else:
                self.Theta = np.transpose(self.dirichlect(np.transpose(ZSDS + 50 / self.K)))
                self.Phi = np.transpose(self.dirichlect(np.transpose(WSZS + self.eta)))

            end_time=time.time()
            print("Epoch {:3d} takes {:8.2f} seconds".format(iter, end_time - start_time))