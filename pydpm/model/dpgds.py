"""
===========================================
Deep Poisson Gamma Dynamical Systems
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0


import time
import numpy as np
from pydpm.utils.Metric import *
from pydpm.utils import Model_Sampler_CPU
import scipy

def log_max(x):
    return np.log(np.maximum(x, 0.000001))

class DPGDS(object):

    def __init__(self,K, device='cpu'):
        self.K = K
        self.L = len(K)
        if device == 'cpu':
            self.device = 'cpu'
            self.Multrnd_Matrix = Model_Sampler_CPU.Multrnd_Matrix
            self.Multrnd_Matrix_CPU = Model_Sampler_CPU.Multrnd_Matrix
            self.Crt_Matrix = Model_Sampler_CPU.Crt_Matrix

        elif device == 'gpu':
            self.device = 'gpu'
            from pydpm.utils import Model_Sampler_GPU
            self.Multrnd_Matrix = Model_Sampler_GPU.Multrnd_Matrix_GPU
            self.Multrnd_Matrix_CPU = Model_Sampler_CPU.Multrnd_Matrix
            self.Crt_Matrix = Model_Sampler_CPU.Crt_Matrix

    def initial(self, data):

        self.data = data
        self.V, self.T = self.data.shape
        L=self.L
        K=self.K

        #setting
        self.Setting={}
        self.Setting['Stationary'] = 1
        self.Setting['NSample'] = 1
        self.Setting['Step'] = 10

        ## self.Supara
        self.Supara = {}
        self.Supara['tao0'] = 1
        self.Supara['gamma0'] = 100  ##??
        self.Supara['eta0'] = 0.1
        self.Supara['epilson0'] = 0.1
        self.Supara['c'] = 1

        ## self.Para
        self.Para = {}
        self.Para['Phi'] = [0] * L
        self.Para['Pi'] = [0] * L
        self.Para['Xi'] = [0] * L
        self.Para['V'] = [0] * L
        self.Para['beta'] = [0] * L
        self.Para['q'] = [0] * L
        self.Para['h'] = [0] * L
        self.Para['n'] = [0] * L
        self.Para['rou'] = [0] * L

        self.Para['Piprior'] = [0] * L
        self.Para['Theta'] = [0] * L
        self.Para['delta'] = [0] * L
        self.Para['Zeta'] = [0] * L
        self.Para['L_dotkt'] = [0] * L
        self.Para['L_kdott'] = [0] * L
        self.Para['A_KT'] = [0] * L
        self.Para['A_VK'] = [0] * L

        self.Para['L_KK'] = [0] * L
        self.Para['prob1'] = [0] * L
        self.Para['prob2'] = [0] * L
        self.Para['Xt_to_t1'] = [0] * (L + 1)
        self.Para['X_layer_split1'] = [0] * L
        self.Para['X_layer'] = [0] * L

        ## initial
        for l in range(L):
            if l == 0:
                self.Para['Phi'][l] = np.random.rand(self.V, K[l])
                self.Para['A_VK'][l] = np.zeros((self.V, K[l]))
            else:
                self.Para['Phi'][l] = np.random.rand(K[l - 1], K[l])
                self.Para['A_VK'][l] = np.zeros((K[l - 1], K[l]))

            self.Para['Phi'][l] = self.Para['Phi'][l] / np.sum(self.Para['Phi'][l], axis=0)
            self.Para['Pi'][l] = np.eye(K[l])
            self.Para['Xi'][l] = 1
            self.Para['V'][l] = np.ones((K[l], 1))
            self.Para['beta'][l] = 1
            self.Para['h'][l] = np.zeros((K[l], K[l]))
            self.Para['n'][l] = np.zeros((K[l], 1))
            self.Para['rou'][l] = np.zeros((K[l], 1))

            self.Para['Theta'][l] = np.ones((K[l], self.T)) / K[l]
            self.Para['delta'][l] = np.ones((self.T, 1))
            self.Para['Zeta'][l] = np.zeros((self.T + 1, 1))
            self.Para['L_dotkt'][l] = np.zeros((K[l], self.T + 1))
            self.Para['A_KT'][l] = np.zeros((K[l], self.T))
            self.Para['L_kdott'][l] = np.zeros((K[l], self.T + 1))
            self.Para['X_layer'][l] = np.zeros((K[l], self.T, 2))


    def train(self, iter_all=200):

        starttime=time.time()
        self.Setting['Burnin'] = int(iter_all / 2)
        self.Setting['Collection'] = iter_all - self.Setting['Burnin']
        self.Likelihood = []
        self.Reconstruct_Error = []
        for i in range(self.Setting['Burnin'] + self.Setting['Collection']):
            print(i-1,":",time.time()-starttime)
            starttime = time.time()
            for l in range(self.L):

                # checked
                self.Para['L_KK'][l] = np.zeros((self.K[l], self.K[l]))

                if l == 0:
                    X_train = np.array(self.data, dtype=np.double, order='C')
                    [self.Para['A_KT'][l], self.Para['A_VK'][l]] = self.Multrnd_Matrix(X_train, self.Para['Phi'][l], self.Para['Theta'][l])

                else:
                    [self.Para['A_KT'][l], self.Para['A_VK'][l]] = self.Multrnd_Matrix(self.Para['Xt_to_t1'][l], self.Para['Phi'][l], self.Para['Theta'][l])


                if l == self.L - 1:

                    for t in range(self.T - 1, 0, -1):  # T-1 : 1
                        tmp1 = self.Para['A_KT'][l][:, t:t + 1] + self.Para['L_dotkt'][l][:, t + 1:t + 2]

                        self.Para['L_kdott'][l][:, t:t + 1] = self.Crt_Matrix(tmp1.astype('double'),
                                                                              self.Supara['tao0'] * np.dot(self.Para['Pi'][l], self.Para['Theta'][l][:,t - 1:t]))

                        start_time = time.time()
                        [self.Para['L_dotkt'][l][:, t:t + 1], tmp] = self.Multrnd_Matrix_CPU(np.array(self.Para['L_kdott'][l][:, t:t + 1], dtype=np.double, order='C'),
                                                                                         self.Para['Pi'][l],
                                                                                         np.array(self.Para['Theta'][l][:, t - 1:t], dtype=np.double, order='C'))


                        self.Para['L_KK'][l] = self.Para['L_KK'][l] + tmp
                else:
                    self.Para['prob1'][l] = self.Supara['tao0'] * np.dot(self.Para['Pi'][l], self.Para['Theta'][l])
                    self.Para['prob2'][l] = self.Supara['tao0'] * np.dot(self.Para['Phi'][l + 1], self.Para['Theta'][l + 1])
                    self.Para['X_layer'][l] = np.zeros((self.K[l], self.T, 2))
                    self.Para['Xt_to_t1'][l + 1] = np.zeros((self.K[l], self.T))
                    self.Para['X_layer_split1'][l] = np.zeros((self.K[l], self.T))

                    for t in range(self.T - 1, 0, -1):
                        start_time=time.time()
                        self.Para['L_kdott'][l][:, t:t + 1] = self.Crt_Matrix(self.Para['A_KT'][l][:, t:t + 1] + self.Para['L_dotkt'][l][:, t + 1:t + 2],
                                                                              self.Supara['tao0'] * (np.dot(self.Para['Phi'][l + 1], self.Para['Theta'][l + 1][:, t:t + 1]) + np.dot(self.Para['Pi'][l], self.Para['Theta'][l][:, t - 1:t])))
                        #print((self.Para['A_KT'][l][:, t:t + 1] + self.Para['L_dotkt'][l][:, t + 1:t + 2]).size)

                        # split layer2 count
                        tmp_input = np.array(self.Para['L_kdott'][l][:, t:t + 1], dtype=np.float64, order='C')
                        tmp1 = self.Para['prob1'][l][:, t - 1:t]
                        tmp2 = self.Para['prob2'][l][:, t:t + 1]
                        [tmp, self.Para['X_layer'][l][:, t, :]] = self.Multrnd_Matrix_CPU(tmp_input, np.concatenate((tmp1, tmp2), axis=1), np.ones((2, 1)))
                        self.Para['X_layer_split1'][l][:, t] = np.reshape(self.Para['X_layer'][l][:, t, 0], -1)  # ????
                        self.Para['Xt_to_t1'][l + 1][:, t] = np.reshape(self.Para['X_layer'][l][:, t, 1], -1)  # ?????
                        # sample split1
                        tmp_input = np.array(self.Para['X_layer_split1'][l][:, t:t + 1], dtype=np.float64, order='C')
                        [self.Para['L_dotkt'][l][:, t:t + 1], tmp] = self.Multrnd_Matrix_CPU(tmp_input, self.Para['Pi'][l],np.array(self.Para['Theta'][l][:,t - 1:t], order='C'))
                        self.Para['L_KK'][l] = self.Para['L_KK'][l] + tmp

                    self.Para['L_kdott'][l][:, 0:1] = self.Crt_Matrix(self.Para['A_KT'][l][:, 0:1] + self.Para['L_dotkt'][l][:, 1:2],
                                                                      self.Supara['tao0'] * np.dot(self.Para['Phi'][l + 1], self.Para['Theta'][l + 1][:, 0:1]))
                    self.Para['Xt_to_t1'][l + 1][:, 0:1] = self.Para['L_kdott'][l][:, 0:1]

                # Sample Phi
                self.Para['Phi'][l] = Model_Sampler_CPU.Sample_Pi(self.Para['A_VK'][l], self.Supara['eta0'])

                # Sample Pi
                self.Para['Piprior'][l] = np.dot(self.Para['V'][l], np.transpose(self.Para['V'][l]))
                self.Para['Piprior'][l][np.arange(self.Para['Piprior'][l].shape[0]), np.arange(self.Para['Piprior'][l].shape[1])] = 0

                self.Para['Piprior'][l] = self.Para['Piprior'][l] + np.diag(np.reshape(self.Para['Xi'][l] * self.Para['V'][l], [self.Para['V'][l].shape[0], 1]))
                self.Para['Pi'][l] = Model_Sampler_CPU.Sample_Pi(self.Para['L_KK'][l], self.Para['Piprior'][l])


            # Calculate Zeta   
            if (self.Setting['Stationary'] == 1):
                for l in range(self.L):
                    if (l == 0):
                        for t in range(self.T - 1, -1, -1):
                            self.Para['Zeta'][l][t] = np.log(1 + self.Para['Zeta'][l][t + 1] + self.Para['delta'][0][t])

                    else:
                        for t in range(self.T - 1, -1, -1):
                            self.Para['Zeta'][l][t] = np.log(1 + self.Para['Zeta'][l][t + 1] + self.Para['Zeta'][l - 1][t])

            # self.Para['L_dotkt'][l][:,T:T+1] = np.random.poisson(self.Para['Zeta'][l][0] * self.Supara['tao0'] * self.Para['Theta'][l] [:,T-1:T])

            # Sample Theta  checked
            for l in range(self.L - 1, -1, -1):  # L-1 : 0

                if l == self.L - 1:

                    for t in range(self.T):
                        if t == 0:
                            shape = self.Para['A_KT'][l][:, t:t + 1] + self.Para['L_dotkt'][l][:, t + 1:t + 2] + self.Supara['tao0'] * self.Para['V'][l]
                        else:
                            shape = self.Para['A_KT'][l][:, t:t + 1] + self.Para['L_dotkt'][l][:, t + 1:t + 2] + self.Supara['tao0'] * np.dot(self.Para['Pi'][l], self.Para['Theta'][l][:, t - 1:t])

                        if (l == 0):
                            scale = self.Para['delta'][0][t] + self.Supara['tao0'] + self.Supara['tao0'] * self.Para['Zeta'][l][t + 1];
                        else:
                            scale = self.Para['Zeta'][l - 1][t] + self.Supara['tao0'] + self.Supara['tao0'] * self.Para['Zeta'][l][t + 1];

                        self.Para['Theta'][l][:, t:t + 1] = np.random.gamma(shape) / scale

                else:
                    for t in range(self.T):
                        if t == 0:
                            shape = self.Para['A_KT'][l][:, t:t + 1] + self.Para['L_dotkt'][l][:, t + 1:t + 2] + self.Supara['tao0'] * np.dot(self.Para['Phi'][l + 1], self.Para['Theta'][l + 1][:, t:t + 1])
                        else:
                            shape = self.Para['A_KT'][l][:, t:t + 1] + self.Para['L_dotkt'][l][:, t + 1:t + 2] + self.Supara['tao0'] * (np.dot(self.Para['Phi'][l + 1], self.Para['Theta'][l + 1][:, t:t + 1]) + np.dot(self.Para['Pi'][l], self.Para['Theta'][l][:, t - 1:t]))

                        if (l == 0):
                            scale = self.Para['delta'][0][t] + self.Supara['tao0'] + self.Supara['tao0'] * self.Para['Zeta'][0][t + 1];
                        else:
                            scale = self.Para['Zeta'][l - 1][t] + self.Supara['tao0'] + self.Supara['tao0'] * self.Para['Zeta'][l][t + 1];

                        self.Para['Theta'][l][:, t:t + 1] = np.random.gamma(shape) / scale

            # Sample Beta check
            for l in range(self.L):
                shape = self.Supara['epilson0'] + self.Supara['gamma0']
                scale = self.Supara['epilson0'] + np.sum(self.Para['V'][l])
                self.Para['beta'][l] = np.random.gamma(shape) / scale

                # Sample q checked
                a = np.sum(self.Para['L_dotkt'][l], axis=1, keepdims=1)
                a[a == 0] = 1e-10
                b = self.Para['V'][l] * (self.Para['Xi'][l] + np.repeat(np.sum(self.Para['V'][l]), self.K[l], axis=0).reshape([self.K[l], 1]) - self.Para['V'][l])
                b[b == 0] = 1e-10  # 
                self.Para['q'][l] = np.maximum(np.random.beta(b, a), 0.00001)
                # Sample h checked
                for k1 in range(self.K[l]):
                    for k2 in range(self.K[l]):
                        self.Para['h'][l][k1:k1 + 1, k2:k2 + 1] = Model_Sampler_CPU.Crt_Matrix(
                            self.Para['L_KK'][l][k1:k1 + 1, k2:k2 + 1], self.Para['Piprior'][l][k1:k1 + 1, k2:k2 + 1])
                # Sample Xi checked
                shape = self.Supara['gamma0'] / self.K[l] + np.trace(self.Para['h'][l])
                scale = self.Para['beta'][l] - np.dot(np.transpose(self.Para['V'][l]), log_max(self.Para['q'][l]))
                self.Para['Xi'][l] = np.random.gamma(shape) / scale

            # Sample V # check
            for k in range(self.K[self.L - 1]):
                self.Para['L_kdott'][self.L - 1][k:k + 1, 0:1] = Model_Sampler_CPU.Crt_Matrix(
                    self.Para['A_KT'][self.L - 1][k:k + 1, 0:1] + self.Para['L_dotkt'][self.L - 1][k:k + 1, 1:2],
                    np.reshape(self.Supara['tao0'] * self.Para['V'][self.L - 1][k], (1, 1)))
                self.Para['n'][self.L - 1][k] = np.sum(self.Para['h'][self.L - 1][k, :] + np.transpose(self.Para['h'][self.L - 1][:, k])) \
                                                - self.Para['h'][self.L - 1][k, k] + self.Para['L_kdott'][self.L - 1][k:k + 1, 0:1]
                self.Para['rou'][self.L - 1][k] = - log_max(self.Para['q'][self.L - 1][k]) * (self.Para['Xi'][self.L - 1] + np.sum(self.Para['V'][self.L - 1]) - self.Para['V'][self.L - 1][k]) \
                                                  - np.dot(np.transpose(log_max(self.Para['q'][self.L - 1])), self.Para['V'][self.L - 1]) \
                                                  + log_max(self.Para['q'][self.L - 1][k]) * self.Para['V'][self.L - 1][k] + self.Para['Zeta'][self.L - 1][0]
            shape_top = self.Supara['gamma0'] / self.K[self.L - 1] + self.Para['n'][self.L - 1]
            scale_top = self.Para['beta'][self.L - 1] + self.Para['rou'][self.L - 1]
            self.Para['V'][self.L - 1] = np.random.gamma(shape_top) / scale_top

            # Sample V[1] V[L] check
            if (self.L > 1):
                for l in range(self.L - 1):
                    for k in range(self.K[l]):
                        self.Para['n'][l][k] = np.sum(self.Para['h'][l][k, :] + np.transpose(self.Para['h'][l][:, k])) - self.Para['h'][l][k, k]
                        self.Para['rou'][l][k] = - log_max(self.Para['q'][l][k]) * (self.Para['Xi'][l] + np.sum(self.Para['V'][l]) - self.Para['V'][l][k]) \
                                                 - np.dot(log_max(np.transpose(self.Para['q'][l])), self.Para['V'][l]) + log_max(self.Para['q'][l][k] * self.Para['V'][l][k])
                        # self.Para['rou'][l][k] = 1

                    shape = self.Supara['gamma0'] / self.K[l] + self.Para['n'][l]
                    scale = self.Para['beta'][l] + self.Para['rou'][l]
                    self.Para['V'][l] = np.random.gamma(shape) / scale

            # Likelihood
            deltatmp = np.zeros([self.K[0], self.T])
            deltatmp[0:self.K[0] - 1, :] = np.transpose(self.Para['delta'][0])
            Theta_hat = deltatmp * self.Para['Theta'][0]
            lambd = np.dot(self.Para['Phi'][0], self.Para['Theta'][0])

            like = np.sum(X_train * np.log(lambd) - lambd) / self.V / self.T
            print("Likelihood:", like)
            self.Likelihood.append(like)