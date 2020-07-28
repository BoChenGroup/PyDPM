"""
===========================================
Poisson Gamma Belief Network
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import numpy as np
from pydpm.utils import Model_Sampler_CPU
import time
from pydpm.utils.Metric import *


class PGBN(object):

    def __init__(self, shape, device='cpu'):

        self.K = np.array(shape)
        self.T = self.K.size

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

        self.data=data
        self.V = data.shape[0]
        self.N = data.shape[1]

        Supara = {}
        Supara['a0pj'] = 0.01
        Supara['b0pj'] = 0.01
        Supara['e0cj'] = 1
        Supara['f0cj'] = 1
        Supara['eta'] = np.ones(self.T) * 0.01

        Phi = []
        Eta = []
        for t in range(self.T):  # from layer 1 to T
            Eta.append(Supara['eta'][t])
            if t == 0:
                Phi.append(0.2 + 0.8 * np.random.rand(self.V, self.K[t]))
            else:
                Phi.append(0.2 + 0.8 * np.random.rand(self.K[t-1], self.K[t]))
            Phi[t] = Phi[t] / np.maximum(realmin, Phi[t].sum(0))

        r_k = np.ones([self.K[self.T-1], 1])/self.K[self.T-1]

        Theta = []
        c_j = []
        for t in range(self.T):  # from layer 1 to T
            Theta.append(np.ones([self.K[t], self.N]) / self.K[t])
            c_j.append(np.ones([1, self.N]))
        c_j.append(np.ones([1, self.N]))  #
        p_j = self.Calculate_pj(c_j, self.T)


        self.Supara = Supara
        self.Phi = Phi
        self.Eta = Eta
        self.Theta = Theta
        self.r_k = r_k
        self.c_j = c_j
        self.p_j = p_j

    def train(self, iter_all=200):

        data = self.data
        Xt_to_t1 = []
        WSZS = []
        for t in range(self.T):
            Xt_to_t1.append(np.zeros(self.Theta[t].shape))
            WSZS.append(np.zeros(self.Phi[t].shape))

        self.Likelihood = []
        self.Reconstruct_Error = []
        for iter in range(iter_all):

            start_time = time.time()

            # ======================== Upward Pass ======================== #
            # Update Phi
            for t in range(self.T):  # from layer 1 to T
                if t == 0:
                    Xt_to_t1[t], WSZS[t] = self.Multrnd_Matrix(data, self.Phi[t], self.Theta[t])
                else:
                    Xt_to_t1[t], WSZS[t] = self.Crt_Multirnd_Matrix(Xt_to_t1[t-1], self.Phi[t], self.Theta[t])

                self.Phi[t] = self.Update_Phi(WSZS[t], self.Eta[t])

            # ======================== Downward Pass ======================== #
            # Update c_j, p_j
            if iter >= 0:
                # Update c_j_2, p_j_2
                if self.T > 1:
                    self.p_j[1][0, :] = np.random.beta(Xt_to_t1[0].sum(0) + self.Supara['a0pj'], self.Theta[1].sum(0) + self.Supara['b0pj'])

                else:
                    self.p_j[1][0, :] = np.random.beta(Xt_to_t1[0].sum(0) + self.Supara['a0pj'], self.r_k.sum(0) + self.Supara['b0pj'])

                self.p_j[1] = np.minimum(np.maximum(self.p_j[1], realmin), 1 - realmin)  # make sure p_j is not too large or small
                self.c_j[1] = (1 - self.p_j[1]) / self.p_j[1]

                # Update c_j_3_T+1, p_j_3_T+1
                for t in [i for i in range(self.T + 1) if i > 1]:  # from layer 3 to layer T+1
                    if t == self.T:
                        self.c_j[t][0, :] = np.random.gamma(self.r_k.sum(0) + self.Supara['e0cj'], 1) / (self.Theta[t-1].sum(0) + self.Supara['f0cj'])

                    else:
                        self.c_j[t][0, :] = np.random.gamma(self.Theta[t].sum(0) + self.Supara['e0cj'], 1) / (self.Theta[t-1].sum(0) + self.Supara['f0cj'])

                p_j_tmp = self.Calculate_pj(self.c_j, self.T)
                self.p_j[2:] = p_j_tmp[2:]

            # Update Theta
            for t in range(self.T - 1, -1, -1):  # from layer T to 1
                if t == self.T - 1:
                    shape = np.repeat(self.r_k, self.N, axis=1)
                else:
                    shape = np.dot(self.Phi[t + 1], self.Theta[t + 1])
                self.Theta[t] = self.Update_Theta(Xt_to_t1[t], shape, self.c_j[t + 1], self.p_j[t])

            end_time = time.time()

            print("Epoch {:3d} takes {:8.2f} seconds".format(iter, end_time-start_time))
            self.Likelihood.append(Poisson_Likelihood(data, np.dot(self.Phi[0], self.Theta[0])) / self.N)
            self.Reconstruct_Error.append(Reconstruct_Error(data, np.dot(self.Phi[0], self.Theta[0])) / self.N)

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

