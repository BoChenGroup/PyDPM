"""
===========================================
Poisson Factor Analysis
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import numpy as np
from ..utils import Model_Sampler_CPU
import time
from ..utils.Metric import *


realmin = 2.2e-10

class PFA(object):

    def __init__(self, K, device='cpu'):

        self.K = K

        if device == 'cpu':
            self.device = 'cpu'
            self.Multrnd_Matrix = Model_Sampler_CPU.Multrnd_Matrix
            self.Crt_Matrix = Model_Sampler_CPU.Crt_Matrix
            self.Crt_Multirnd_Matrix = Model_Sampler_CPU.Crt_Multirnd_Matrix

        elif device == 'gpu':
            self.device = 'gpu'
            from pydpm.utils import Model_Sampler_GPU
            self.Multrnd_Matrix = Model_Sampler_GPU.Multrnd_Matrix_GPU
            self.Crt_Matrix = Model_Sampler_GPU.Crt_Matrix_GPU
            self.Crt_Multirnd_Matrix = Model_Sampler_GPU.Crt_Multirnd_Matrix_GPU

    def initial(self, data):

        self.data = data
        self.V = data.shape[0]
        self.N = data.shape[1]

        Supara = {}
        Supara['a0pj'] = 0.01
        Supara['b0pj'] = 0.01
        Supara['e0cj'] = 1
        Supara['f0cj'] = 1
        # Supara['eta'] = np.ones(self.T) * 0.01
        self.Eta = 0.01
        Eta = []
        self.Phi = 0.2 + 0.8 * np.random.rand(self.V, self.K)
        self.Phi = self.Phi / np.maximum(realmin, self.Phi.sum(0))

        r_k = np.ones([self.K, 1])/self.K

        self.Theta = np.ones([self.K, self.N]) / self.K
        c_j = np.ones([1, self.N])
        p_j = np.ones([1, self.N])

        self.Supara = Supara
        self.r_k = r_k
        self.c_j = c_j
        self.p_j = p_j


    def train(self, iter_all=200):

        Xt_to_t1 = np.zeros(self.Theta.shape)
        WSZS = np.zeros(self.Phi.shape)
        data = self.data
        self.Likelihood = []
        self.Reconstruct_Error = []
        for iter in range(iter_all):

            start_time = time.time()

            # ======================== Upward Pass ======================== #
            # Update Phi
            Xt_to_t1, WSZS = self.Multrnd_Matrix(data, self.Phi, self.Theta)

            self.Phi = self.Update_Phi(WSZS, self.Eta)

            # ======================== Downward Pass ======================== #
            # Update c_j, p_j
            if iter >= 0:
                # Update c_j_2, p_j_2
                self.p_j[0, :] = np.random.beta(Xt_to_t1.sum(0) + self.Supara['a0pj'], self.r_k.sum(0) + self.Supara['b0pj'])

                self.p_j = np.minimum(np.maximum(self.p_j, realmin), 1 - realmin)  # make sure p_j is not too large or small
                self.c_j = (1 - self.p_j) / self.p_j

                # Update c_j_3_T+1, p_j_3_T+1

            # Update Theta
                shape = np.repeat(self.r_k, self.N, axis=1)
                self.Theta = self.Update_Theta(Xt_to_t1, shape, self.c_j, self.p_j)

            end_time = time.time()

            print("Epoch {:3d} takes {:8.2f} seconds".format(iter, end_time-start_time))
            self.Likelihood.append(Poisson_Likelihood(data, np.dot(self.Phi, self.Theta)) / self.N)
            self.Reconstruct_Error.append(Reconstruct_Error(data, np.dot(self.Phi, self.Theta)) / self.N)

            # print("Likelihood: {:<8.2f}".format(Poisson_Likelihood(data, np.dot(self.Phi[0], self.Theta[0])) / self.N))
            # print("Reconstruct: {:<8.2f}".format(Reconstruct_Error(data, np.dot(self.Phi[0], self.Theta[0])) / self.N))


    def Calculate_pj(self, c_j, T):

        # calculate p_j from layer 1 to T+1
        p_j = []
        N = c_j[1].size
        p_j.append((1 - np.exp(-1)) * np.ones([1, N]))  # p_j_1
        p_j.append(1 / (1 + c_j[1]))                    # p_j_2

        for t in [i for i in range(T + 1) if i > 1]:    # p_j_3_T+1; only T>=2 works
            tmp = -np.log(np.maximum(1 - p_j[t - 1], realmin))
            p_j.append(tmp / (tmp + c_j[t]))

        return p_j

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

