"""
===========================================
Layer
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

from pydpm.utils import Model_Sampler_CPU
from pydpm.utils.Metric import *


realmin= 2.2e-10

class data_base():

    def __init__(self,path):
        self.load_data(path)

    def load_data(self,path):
        import numpy as np
        import scipy.io as sio
        train_data = sio.loadmat(path)
        self.data = np.array(np.ceil(train_data['train_mnist'] * 5), order='C')[:, 0:999]


class base_layer():

    formerlayer=None
    backlayer=None


    def initial(self,device='cpu'):
        raise InterruptedError

    def feedforward(self):
        raise InterruptedError

    def backforward(self):
        raise InterruptedError


class prob_layer(base_layer):

    def __init__(self, K):

        self.K = K
        self.T = 0

    def initial(self,device='cpu'):

        if device=='cpu':
            self.Multrnd_Matrix = Model_Sampler_CPU.Multrnd_Matrix
            self.Crt_Matrix = Model_Sampler_CPU.Crt_Matrix
            self.Crt_Multirnd_Matrix = Model_Sampler_CPU.Crt_Multirnd_Matrix
        elif device=='gpu':
            from ..utils import Model_Sampler_GPU
            self.Multrnd_Matrix = Model_Sampler_GPU.Multrnd_Matrix_GPU
            self.Crt_Matrix = Model_Sampler_GPU.Crt_Matrix_GPU
            self.Crt_Multirnd_Matrix = Model_Sampler_GPU.Crt_Multirnd_Matrix_GPU
        self.device=device
        Supara = {}
        Supara['a0pj'] = 0.01
        Supara['b0pj'] = 0.01
        Supara['e0cj'] = 1
        Supara['f0cj'] = 1
        Supara['eta'] = 0.01

        if self.T==1:
            self.V = self.formerlayer.data.shape[0]
            self.N = self.formerlayer.data.shape[1]
            self.Phi = 0.2 + 0.8 * np.random.rand(self.V, self.K)
        else:
            self.N = self.formerlayer.Theta.shape[1]
            self.Phi=0.2 + 0.8 * np.random.rand(self.formerlayer.K, self.K)

        self.Eta = Supara['eta']
        self.Phi = self.Phi / np.maximum(realmin, self.Phi.sum(0))

        self.c_j = np.ones([1, self.N])
        self.p_j = np.ones([1, self.N])

        self.r_k = np.ones([self.K, 1]) / self.K
        self.Theta=np.ones([self.K, self.N]) / self.K

        self.Supara = Supara
        self.Likelihood=[]
        self.Reconstruct_Error=[]

    def forward_prob2prob(self):

        self.Xt_to_t1 = np.zeros(self.Theta.shape)
        self.WSZS = np.zeros(self.Phi.shape)
        # ======================== Upward Pass ======================== #
        # Update Phi

        self.Xt_to_t1, self.WSZS = self.Crt_Multirnd_Matrix(self.formerlayer.Xt_to_t1, self.Phi, self.Theta)
        self.Phi = self.Update_Phi(self.WSZS, self.Eta)

        # ======================== Downward Pass ======================== #

        if self.backlayer == None:

            self.c_j[0, :] = np.random.gamma(self.r_k.sum(0) + self.Supara['e0cj'], 1) / \
                             (self.formerlayer.Theta.sum(0) + self.Supara['f0cj'])

        else:

            self.c_j[0, :] = np.random.gamma(self.Theta.sum(0) + self.Supara['e0cj'], 1) / \
                             (self.formerlayer.Theta.sum(0) + self.Supara['f0cj'])

        tmp = -np.log(np.maximum(1 - self.formerlayer.p_j, realmin))
        self.p_j = tmp / (tmp + self.c_j)

        # if self.backlayer==None:
        #     self.c_j[0, :] = np.random.gamma(self.Theta.sum(0) + self.Supara['e0cj'], 1) / (
        #         self.formerlayer.Theta.sum(0) + self.Supara['f0cj'])
        #     self.c_j_last[0, :] = np.random.gamma(self.r_k.sum(0) + self.Supara['e0cj'], 1) / (
        #         self.formerlayer.Theta.sum(0) + self.Supara['f0cj'])
        #
        # else:
        #     self.c_j[0, :] = np.random.gamma(self.Theta.sum(0) + self.Supara['e0cj'], 1) / (
        #         self.formerlayer.Theta.sum(0) + self.Supara['f0cj'])
        #
        # p_j_tmp = self.Calculate_pj(self.c_j, self.T)
        # self.p_j = p_j_tmp

    def forward_data2prob(self):

        self.Xt_to_t1 = np.zeros(self.Theta.shape)
        self.WSZS = np.zeros(self.Phi.shape)
        # ======================== Upward Pass ======================== #
        # Update Phi
        self.Xt_to_t1, self.WSZS = self.Multrnd_Matrix(self.formerlayer.data, self.Phi, self.Theta)
        self.Phi = self.Update_Phi(self.WSZS, self.Eta)

        # ======================== Downward Pass ======================== #
        # Update c_j, p_j
        # Update c_j_2, p_j_2
        if self.backlayer != None:
            self.p_j[0, :] = np.random.beta(self.Xt_to_t1.sum(0) + self.Supara['a0pj'],
                                            self.backlayer.Theta.sum(0) + self.Supara['b0pj'])

        else:
            self.p_j[0, :] = np.random.beta(self.Xt_to_t1.sum(0) + self.Supara['a0pj'],
                                            self.r_k.sum(0) + self.Supara['b0pj'])

        self.p_j = np.minimum(np.maximum(self.p_j, realmin), 1 - realmin)  # make sure p_j is not too large or small
        self.c_j = (1 - self.p_j) / self.p_j

    def backward_lastlayer(self):

        shape = np.repeat(self.r_k, self.N, axis=1)
        if self.T==1:
            self.Theta = self.Update_Theta(self.Xt_to_t1, shape, self.c_j, (1 - np.exp(-1)) * np.ones([1, self.N]))
        else:
            self.Theta = self.Update_Theta(self.Xt_to_t1, shape, self.c_j, self.formerlayer.p_j)
        if self.T == 1:
            self.Likelihood.append(Poisson_Likelihood(self.formerlayer.data, np.dot(self.Phi, self.Theta)) / self.N)
            self.Reconstruct_Error.append(
                Reconstruct_Error(self.formerlayer.data, np.dot(self.Phi, self.Theta)) / self.N)
            print(self.Likelihood[-1], self.Reconstruct_Error[-1])


    def backward_prob2prob(self):

        shape = np.dot(self.backlayer.Phi, self.backlayer.Theta)
        if self.T==1:
            self.Theta = self.Update_Theta(self.Xt_to_t1, shape, self.c_j, (1 - np.exp(-1)) * np.ones([1, self.N]))
        else:
            self.Theta = self.Update_Theta(self.Xt_to_t1, shape, self.c_j, self.formerlayer.p_j)

        if self.T == 1:

            self.Likelihood.append(Poisson_Likelihood(self.formerlayer.data, np.dot(self.Phi, self.Theta)) / self.N)
            self.Reconstruct_Error.append(Reconstruct_Error(self.formerlayer.data, np.dot(self.Phi, self.Theta)) / self.N)

    def feedforward(self):

        if type(self.formerlayer) == prob_layer:
            self.forward_prob2prob()
        if type(self.formerlayer) == data_base:
            self.forward_data2prob()

    def feedbackward(self):

        if self.backlayer == None:
            self.backward_lastlayer()
        if type(self.backlayer) == prob_layer:
            self.backward_prob2prob()

    def Update_Phi(self, WSZS_t, Eta_t):

        # update Phi_t
        Phi_t_shape = WSZS_t + Eta_t
        Phi_t = np.random.gamma(Phi_t_shape, 1)
        Phi_t = Phi_t / Phi_t.sum(0)

        return Phi_t

    def Update_Theta(self, Xt_to_t1_t, shape, c_j_t1, p_j_t):

        # update Theta_t
        Theta_t_shape = Xt_to_t1_t + shape
        Theta_t = np.random.gamma(Theta_t_shape, 1) / (c_j_t1[0, :] - np.log(np.maximum(realmin, 1 - p_j_t[0, :])))

        return Theta_t


class model():

    def __init__(self,layerlist,device='cpu'):

        self.layerlist=layerlist
        for i,v in enumerate(layerlist):
            v.T=i
            if i!=0:
                v.formerlayer=layerlist[i-1]
                v.initial(device)
            if i!=len(layerlist)-1:
                v.backlayer=layerlist[i+1]

    def train(self,iter):
        for i in range(iter):
            print("epoch:",i)
            for t in self.layerlist[1:]:
                t.feedforward()
            for t in range(len(self.layerlist)-1,0,-1):
                self.layerlist[t].feedbackward()






