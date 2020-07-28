"""
===========================================
Poisson Gamma Dynamical Systems
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0


import time
import numpy as np
from pydpm.utils.Metric import *
from pydpm.utils import Model_Sampler_CPU
import scipy

class PGDS(object):

    def __init__(self, K, device='cpu'):

        self.K = K
        self.L = 1
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
        L = self.L
        K = self.K

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

        self.Para['L_KK'] = [0] * L
        self.Para['prob1'] = [0] * L
        self.Para['prob2'] = [0] * L
        self.Para['Xt_to_t1'] = [0] * (L + 1)
        self.Para['X_layer_split1'] = [0] * L
        self.Para['X_layer'] = [0] * L

        ## initial
  
        self.Para['Phi'] = np.random.rand(self.V, K)
        self.Para['A_VK'] = np.zeros((self.V, K))

        self.Para['Phi'] = self.Para['Phi'] / np.sum(self.Para['Phi'], axis=0)
        self.Para['Pi'] = np.eye(K)
        self.Para['Xi'] = 1
        self.Para['V'] = np.ones((K, 1))
        self.Para['beta'] = 1
        self.Para['h'] = np.zeros((K, K))
        self.Para['n'] = np.zeros((K, 1))
        self.Para['rou'] = np.zeros((K, 1))

        self.Para['Theta'] = np.ones((K, self.T)) / K
        self.Para['delta'] = np.ones((self.T, 1))
        self.Para['Zeta'] = np.zeros((self.T + 1, 1))
        self.Para['L_dotkt'] = np.zeros((K, self.T + 1))
        self.Para['A_KT'] = np.zeros((K, self.T))
        self.Para['L_kdott'] = np.zeros((K, self.T + 1))
        self.Para['X_layer'] = np.zeros((K, self.T, 2))


    def train(self, iter_all=200):

        starttime=time.time()
        self.Setting['Burnin'] = int(iter_all/2)
        self.Setting['Collection'] = iter_all-self.Setting['Burnin']
        self.Likelihood = []
        self.Reconstruct_Error = []
        for i in range(self.Setting['Burnin'] + self.Setting['Collection']):

            print(i-1,":",time.time()-starttime)
            starttime = time.time()

            self.Para['L_KK'] = np.zeros((self.K, self.K))

            X_train = np.array(self.data, dtype=np.double, order='C')
            [self.Para['A_KT'], self.Para['A_VK']] = self.Multrnd_Matrix(X_train, self.Para['Phi'], self.Para['Theta'])

            for t in range(self.T - 1, 0, -1):  # T-1 : 1

                tmp1 = self.Para['A_KT'][:, t:t + 1] + self.Para['L_dotkt'][:, t + 1:t + 2]

                self.Para['L_kdott'][:, t:t + 1] = self.Crt_Matrix(tmp1.astype('double'), self.Supara['tao0'] * np.dot(self.Para['Pi'], self.Para['Theta'][:,t - 1:t]))


                [self.Para['L_dotkt'][:, t:t + 1], tmp] = self.Multrnd_Matrix_CPU(np.array(self.Para['L_kdott'][:, t:t + 1], dtype=np.double, order='C'),
                                                                                 self.Para['Pi'],
                                                                                 np.array(self.Para['Theta'][:, t - 1:t], dtype=np.double, order='C'))


                self.Para['L_KK'] = self.Para['L_KK'] + tmp

            # Sample Phi
            self.Para['Phi'] = Model_Sampler_CPU.Sample_Pi(self.Para['A_VK'], self.Supara['eta0'])

            # Sample Pi
            self.Para['Piprior'] = np.dot(self.Para['V'], np.transpose(self.Para['V']))
            self.Para['Piprior'][np.arange(self.Para['Piprior'].shape[0]), np.arange(self.Para['Piprior'].shape[1])] = 0

            self.Para['Piprior'] = self.Para['Piprior'] + np.diag(np.reshape(self.Para['Xi'] * self.Para['V'], [self.Para['V'].shape[0], 1]))
            self.Para['Pi'] = Model_Sampler_CPU.Sample_Pi(self.Para['L_KK'], self.Para['Piprior'])


            # Calculate Zeta   
            if (self.Setting['Stationary'] == 1):
                for t in range(self.T - 1, -1, -1):
                    self.Para['Zeta'][t] = np.log(1 + self.Para['Zeta'][t + 1] + self.Para['delta'][t])


            # self.Para['L_dotkt'][:,T:T+1] = np.random.poisson(self.Para['Zeta'][0] * self.Supara['tao0'] * self.Para['Theta'] [:,T-1:T])

            # Sample Theta  checked
            for t in range(self.T):
                if t == 0:
                    shape = self.Para['A_KT'][:, t:t + 1] + self.Para['L_dotkt'][:, t + 1:t + 2] + self.Supara['tao0'] * self.Para['V']
                else:
                    shape = self.Para['A_KT'][:, t:t + 1] + self.Para['L_dotkt'][:, t + 1:t + 2] + self.Supara['tao0'] * np.dot(self.Para['Pi'], self.Para['Theta'][:, t - 1:t])

                scale = self.Para['delta'][t] + self.Supara['tao0'] + self.Supara['tao0'] * self.Para['Zeta'][t + 1];
                self.Para['Theta'][:, t:t + 1] = np.random.gamma(shape) / scale

            # Sample Beta check
            shape = self.Supara['epilson0'] + self.Supara['gamma0']
            scale = self.Supara['epilson0'] + np.sum(self.Para['V'])
            self.Para['beta'] = np.random.gamma(shape) / scale

            # Sample q checked
            a = np.sum(self.Para['L_dotkt'], axis=1, keepdims=1)
            a[a == 0] = 1e-10
            b = self.Para['V'] * (self.Para['Xi'] + np.repeat(np.sum(self.Para['V']), self.K, axis=0).reshape([self.K, 1]) - self.Para['V'])
            b[b == 0] = 1e-10  # 
            self.Para['q'] = np.maximum(np.random.beta(b, a), 0.00001)
            # Sample h checked
            for k1 in range(self.K):
                for k2 in range(self.K):
                    self.Para['h'][k1:k1 + 1, k2:k2 + 1] = Model_Sampler_CPU.Crt_Matrix(
                        self.Para['L_KK'][k1:k1 + 1, k2:k2 + 1], self.Para['Piprior'][k1:k1 + 1, k2:k2 + 1])
            # Sample Xi checked
            shape = self.Supara['gamma0'] / self.K + np.trace(self.Para['h'])
            scale = self.Para['beta'] - np.dot(np.transpose(self.Para['V']), log_max(self.Para['q']))
            self.Para['Xi'] = np.random.gamma(shape) / scale

            # Sample V # check
            for k in range(self.K):
                self.Para['L_kdott'][k:k + 1, 0:1] = Model_Sampler_CPU.Crt_Matrix(
                    self.Para['A_KT'][k:k + 1, 0:1] + self.Para['L_dotkt'][k:k + 1, 1:2],
                    np.reshape(self.Supara['tao0'] * self.Para['V'][k], (1, 1)))
                self.Para['n'][k] = np.sum(self.Para['h'][k, :] + np.transpose(self.Para['h'][:, k])) \
                                    - self.Para['h'][k, k] + self.Para['L_kdott'][k:k + 1, 0:1]

                self.Para['rou'][k] = - log_max(self.Para['q'][k]) * (self.Para['Xi'] + np.sum(self.Para['V']) - self.Para['V'][k]) \
                                      - np.dot(np.transpose(log_max(self.Para['q'])), self.Para['V']) \
                                      + log_max(self.Para['q'][k]) * self.Para['V'][k] + self.Para['Zeta'][0]
            shape_top = self.Supara['gamma0'] / self.K + self.Para['n']
            scale_top = self.Para['beta'] + self.Para['rou']
            self.Para['V'] = np.random.gamma(shape_top) / scale_top

            # Likelihood
            deltatmp = np.zeros([self.K, self.T])
            deltatmp[0:self.K - 1, :] = np.transpose(self.Para['delta'])
            Theta_hat = deltatmp * self.Para['Theta']
            lambd = np.dot(self.Para['Phi'], self.Para['Theta'])

            like = np.sum(X_train * np.log(lambd) - lambd) / self.V / self.T
            print("Likelihood:", like)
            self.Likelihood.append(like)