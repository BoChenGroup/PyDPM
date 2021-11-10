"""
===========================================
Deep Poisson Gamma Dynamical Systems
Dandan Guo, Bo Chen and Hao Zhang
Published in Neural Information Processing Systems 2018

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>; Wei Zhao <13279389260@163.com>
# License: BSD-3-Clause


import os
import copy
import time
import numpy as np

from ._basic_model import Basic_Model
from .._sampler import Basic_Sampler
from .._utils import *

class DPGDS(Basic_Model):
    def __init__(self, K: int, device='gpu'):
        """
        The basic model for DPGDS
        Inputs:
            K      : [int] number of topics in DPGDS;
            device : [str] 'cpu' or 'gpu';

        Attributes:
            @public:
                global_params : [Params] the global parameters of the probabilistic model
                local_params  : [Params] the local parameters of the probabilistic model

            @private:
                _model_setting : [Params] the model settings of the probabilistic model
                _hyper_params  : [Params] the hyper parameters of the probabilistic model

        """
        super(DPGDS, self).__init__()
        setattr(self, '_model_name', 'DPGDS')

        self._model_setting.K = K
        self._model_setting.L = len(K)
        self._model_setting.device = device

        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''

        self._sampler = Basic_Sampler(self._model_setting.device)

        # self.Crt_Matrix = Model_Sampler_CPU.Crt_Matrix
        # self.Multrnd_Matrix_CPU = Model_Sampler_CPU.Multrnd_Matrix
        # self.Sample_Pi = Model_Sampler_CPU.Sample_Pi
        # self.Multrnd_Matrix = self._sampler.multi_aug


    def initial(self, data: np.ndarray):
        '''
        Inintial the parameters of DPGDS with the input documents
        Inputs:
            data : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary length of V

        Attributes:
            @public:
                global_params.Phi  : [np.ndarray] V*K matrix, K topics with a vocabulary length of V
                local_params.Theta : [np.ndarray] N*K matrix, the topic propotions of N documents

            @private:
                _model_setting.V         : [int] scalar, the length of the vocabulary
                _model_setting.Stationary: [int] scalar,
                _hyper_params.Supara     : [dict] scalar,
                _hyper_params.Para       : [dict] scalar,

        '''
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'
        self._model_setting.V = data.shape[0]
        self._model_setting.Stationary = 1

        self.global_params.Phi = [0] * self._model_setting.L
        self.global_params.Pi = [0] * self._model_setting.L
        self.global_params.Xi = [0] * self._model_setting.L
        self.global_params.V = [0] * self._model_setting.L
        self.global_params.beta = [0] * self._model_setting.L
        self.global_params.q = [0] * self._model_setting.L
        self.global_params.h = [0] * self._model_setting.L
        self.global_params.n = [0] * self._model_setting.L
        self.global_params.rou = [0] * self._model_setting.L

        for l in range(self._model_setting.L):
            if l == 0:
                self.global_params.Phi[l] = np.random.rand(self._model_setting.V, self._model_setting.K[l])
            else:
                self.global_params.Phi[l] = np.random.rand(self._model_setting.K[l - 1], self._model_setting.K[l])
            self.global_params.Phi[l] = self.global_params.Phi[l] / np.sum(self.global_params.Phi[l], axis=0)

            self.global_params.Pi [l] = np.eye(self._model_setting.K[l])
            self.global_params.Xi[l] = 1
            self.global_params.V[l] = np.ones((self._model_setting.K[l], 1))
            self.global_params.beta[l] = 1
            self.global_params.h[l] = np.zeros((self._model_setting.K[l], self._model_setting.K[l]))
            self.global_params.n[l] = np.zeros((self._model_setting.K[l], 1))
            self.global_params.rou[l] = np.zeros((self._model_setting.K[l], 1))

        self._hyper_params.tao0 = 1
        self._hyper_params.gamma0 = 100
        self._hyper_params.eta0 = 0.1
        self._hyper_params.epilson0 = 0.1

    def train(self, iter_all: int, data: np.ndarray, is_train: bool = True):
        '''
        Inputs:
            iter_all   : [int] scalar, the iterations of sampling
            train_data : [np.ndarray] V*N_train matrix, N_train bag-of-words vectors with a vocabulary length of V
            is_train   : [bool] True or False, whether to update the global params in the probabilistic model

        Attributes:
            @public:
                local_params.Theta : [np.ndarray] N_train*K matrix, the topic propotions of N_train documents

            @private:
                _model_setting.N         : [int] scalar, the number of the documents in the corpus
                _model_setting.Burnin    : [int] scalar, the burunin iterations of sampling
                _model_setting.Collection: [int] scalar, the Collection iterations of sampling
                _hyper_params.Para       : [dict] scalar,
        Outputs:
                local_params  : [Params] the local parameters of the probabilistic model

        '''
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'
        self._model_setting.T = data.shape[1]

        # local params
        self.local_params.Theta = [0] * self._model_setting.L
        self.local_params.delta = [0] * self._model_setting.L
        self.local_params.Zeta = [0] * self._model_setting.L

        for l in range(self._model_setting.L):
            self.local_params.Theta[l] = np.ones((self._model_setting.K[l], self._model_setting.T)) / self._model_setting.K[l]
            self.local_params.Zeta[l] = np.zeros((self._model_setting.T + 1, 1))
            self.local_params.delta[l] = np.ones((self._model_setting.T, 1))

        # temporary params
        A_KT = [0] * self._model_setting.L
        A_VK = [0] * self._model_setting.L
        L_dotkt = [0] * self._model_setting.L
        L_kdott = [0] * self._model_setting.L
        L_KK = [0] * self._model_setting.L
        prob1 = [0] * self._model_setting.L
        prob2 = [0] * self._model_setting.L
        Xt_to_t1 = [0] * (self._model_setting.L + 1)
        X_layer_split1 = [0] * self._model_setting.L
        X_layer = [0] * self._model_setting.L
        Piprior = [0] * self._model_setting.L

        for l in range(self._model_setting.L):
            if l == 0:
                A_VK[l] = np.zeros((self._model_setting.V, self._model_setting.K[l]))
            else:
                A_VK[l] = np.zeros((self._model_setting.K[l - 1], self._model_setting.K[l]))

            A_KT[l] = np.zeros((self._model_setting.K[l], self._model_setting.T))
            L_dotkt[l] = np.zeros((self._model_setting.K[l], self._model_setting.T + 1))
            L_kdott[l] = np.zeros((self._model_setting.K[l], self._model_setting.T + 1))
            X_layer[l] = np.zeros((self._model_setting.K[l], self._model_setting.T, 2))

        for iter in range(iter_all):

            start_time = time.time()
            for l in range(self._model_setting.L):

                L_KK[l] = np.zeros((self._model_setting.K[l], self._model_setting.K[l]))

                if l == 0:
                    A_KT[l], A_VK[l] = self._sampler.multi_aug(data, self.global_params.Phi[l], self.local_params.Theta[l])
                else:
                    A_KT[l], A_VK[l] = self._sampler.multi_aug(Xt_to_t1[l], self.global_params.Phi[l], self.local_params.Theta[l])

                if l == self._model_setting.L - 1:
                    for t in range(self._model_setting.T - 1, 0, -1):  # T-1 : 1
                        tmp1 = A_KT[l][:, t:t + 1] + L_dotkt[l][:, t + 1:t + 2]
                        L_kdott[l][:, t:t + 1] = self._sampler.crt(tmp1, self._hyper_params.tao0 * np.dot(self.global_params.Pi[l], self.local_params.Theta[l][:, t - 1:t]))
                        L_dotkt[l][:, t:t + 1], tmp = self._sampler.multi_aug(L_kdott[l][:, t:t + 1], self.global_params.Pi[l], self.local_params.Theta[l][:, t - 1:t])
                        L_KK[l] = L_KK[l] + tmp

                else:
                    prob1[l] = self._hyper_params.tao0 * np.dot(self.global_params.Pi[l], self.local_params.Theta[l])
                    prob2[l] = self._hyper_params.tao0 * np.dot(self.global_params.Phi[l + 1], self.local_params.Theta[l + 1])
                    X_layer[l] = np.zeros((self._model_setting.K[l], self._model_setting.T, 2))
                    Xt_to_t1[l + 1] = np.zeros((self._model_setting.K[l], self._model_setting.T))
                    X_layer_split1[l] = np.zeros((self._model_setting.K[l], self._model_setting.T))

                    for t in range(self._model_setting.T - 1, 0, -1):
                        L_kdott[l][:, t:t + 1] = self._sampler.crt(A_KT[l][:, t:t + 1] + L_dotkt[l][:, t + 1:t + 2], self._hyper_params.tao0 * (np.dot(self.global_params.Phi[l + 1], self.local_params.Theta[l + 1][:, t:t + 1]) + np.dot(self.global_params.Pi[l], self.local_params.Theta[l][:, t - 1:t])))

                        # split layer2 count
                        tmp_input = np.array(L_kdott[l][:, t:t + 1], dtype=np.float64, order='C')
                        tmp1 = prob1[l][:, t - 1:t]
                        tmp2 = prob2[l][:, t:t + 1]
                        [tmp, X_layer[l][:, t, :]] = self._sampler.multi_aug(tmp_input, np.concatenate((tmp1, tmp2), axis=1), np.ones((2, 1)))
                        X_layer_split1[l][:, t] = np.reshape(X_layer[l][:, t, 0], -1)  # ????
                        Xt_to_t1[l + 1][:, t] = np.reshape(X_layer[l][:, t, 1], -1)  # ?????
                        # sample split1
                        tmp_input = np.array(X_layer_split1[l][:, t:t + 1], dtype=np.float64, order='C')
                        L_dotkt[l][:, t:t + 1], tmp = self._sampler.multi_aug(tmp_input, self.global_params.Pi[l], np.array(self.local_params.Theta[l][:, t - 1:t], order='C'))
                        L_KK[l] = L_KK[l] + tmp

                    L_kdott[l][:, 0:1] = self._sampler.crt(A_KT[l][:, 0:1] + L_dotkt[l][:, 1:2], self._hyper_params.tao0 * np.dot(self.global_params.Phi[l + 1], self.local_params.Theta[l + 1][:, 0:1]))
                    Xt_to_t1[l + 1][:, 0:1] = L_kdott[l][:, 0:1]


                if is_train:
                    # Sample Phi
                    self.global_params.Phi[l] = self.Sample_Pi(A_VK[l], self._hyper_params.eta0)

                    # Sample Pi
                    Piprior[l] = np.dot(self.global_params.V[l], np.transpose(self.global_params.V[l]))
                    Piprior[l][np.arange(Piprior[l].shape[0]), np.arange(Piprior[l].shape[1])] = 0
                    Piprior[l] = Piprior[l] + np.diag(np.reshape(self.global_params.Xi[l] * self.global_params.V[l], [self.global_params.V[l].shape[0]]))
                    self.global_params.Pi[l] = self.Sample_Pi(L_KK[l], Piprior[l])

            # Calculate Zeta
            if (self._model_setting.Stationary == 1):
                for l in range(self._model_setting.L):
                    if (l == 0):
                        for t in range(self._model_setting.T - 1, -1, -1):
                            self.local_params.Zeta[l][t] = np.log(1 + self.local_params.Zeta[l][t + 1] + self.local_params.delta[0][t])

                    else:
                        for t in range(self._model_setting.T - 1, -1, -1):
                            self.local_params.Zeta[l][t] = np.log(1 + self.local_params.Zeta[l][t + 1] + self.local_params.Zeta[l - 1][t])

            # Sample Theta  checked
            for l in range(self._model_setting.L - 1, -1, -1):  # L-1 : 0
                if l == self._model_setting.L - 1:
                    for t in range(self._model_setting.T):
                        if t == 0:
                            shape = A_KT[l][:, t:t + 1] + L_dotkt[l][:, t + 1:t + 2] + self._hyper_params.tao0 * self.global_params.V[l]
                        else:
                            shape = A_KT[l][:, t:t + 1] + L_dotkt[l][:, t + 1:t + 2] + self._hyper_params.tao0 * np.dot(self.global_params.Pi[l], self.local_params.Theta[l][:, t - 1:t])
                        if (l == 0):
                            scale = self.local_params.delta[0][t] + self._hyper_params.tao0 + self._hyper_params.tao0 * self.local_params.Zeta[l][t + 1]
                        else:
                            scale = self.local_params.Zeta[l - 1][t] + self._hyper_params.tao0 + self._hyper_params.tao0 * self.local_params.Zeta[l][t + 1]

                        self.local_params.Theta[l][:, t:t + 1] = self._sampler.gamma(shape) / scale
                else:
                    for t in range(self._model_setting.T):
                        if t == 0:
                            shape = A_KT[l][:, t:t + 1] + L_dotkt[l][:, t + 1:t + 2] + self._hyper_params.tao0 * np.dot(self.global_params.Phi[l + 1], self.local_params.Theta[l + 1][:, t:t + 1])
                        else:
                            shape = A_KT[l][:, t:t + 1] + L_dotkt[l][:, t + 1:t + 2] + self._hyper_params.tao0 * (np.dot(self.global_params.Phi[l + 1], self.local_params.Theta[l + 1][:, t:t + 1]) + np.dot(self.global_params.Pi[l], self.local_params.Theta[l][:, t - 1:t]))
                        if (l == 0):
                            scale = self.local_params.delta[0][t] + self._hyper_params.tao0 + self._hyper_params.tao0 * self.local_params.Zeta[0][t + 1]
                        else:
                            scale = self.local_params.Zeta[l - 1][t] + self._hyper_params.tao0 + self._hyper_params.tao0 * self.local_params.Zeta[l][t + 1]

                        self.local_params.Theta[l][:, t:t + 1] = self._sampler.gamma(shape) / scale

            if is_train:
                # Sample Beta check
                for l in range(self._model_setting.L):
                    shape = self._hyper_params.epilson0 + self._hyper_params.gamma0
                    scale = self._hyper_params.epilson0 + np.sum(self.global_params.V[l])
                    self.global_params.beta[l] = self._sampler.gamma(shape) / scale

                    # Sample q checked
                    a = np.sum(L_dotkt[l], axis=1, keepdims=1)
                    a[a == 0] = 1e-10
                    b = self.global_params.V[l] * (self.global_params.Xi[l] + np.repeat(np.sum(self.global_params.V[l]), self._model_setting.K[l], axis=0).reshape([self._model_setting.K[l], 1]) - self.global_params.V[l])
                    b[b == 0] = 1e-10  #
                    self.global_params.q[l] = np.maximum(self._sampler.beta(b, a), realmin)
                    # Sample h checked
                    for k1 in range(self._model_setting.K[l]):
                        for k2 in range(self._model_setting.K[l]):
                            self.global_params.h[l][k1:k1 + 1, k2:k2 + 1] = self._sampler.crt(L_KK[l][k1:k1 + 1, k2:k2 + 1], Piprior[l][k1:k1 + 1, k2:k2 + 1])
                    # Sample Xi checked
                    shape = self._hyper_params.gamma0 / self._model_setting.K[l] + np.trace(self.global_params.h[l])
                    scale = self.global_params.beta[l] - np.dot(np.transpose(self.global_params.V[l]), log_max(self.global_params.q[l]))
                    self.global_params.Xi[l] = self._sampler.gamma(shape) / scale

                # Sample V # check
                for k in range(self._model_setting.K[self._model_setting.L - 1]):
                    L_kdott[self._model_setting.L - 1][k:k + 1, 0:1] = self._sampler.crt(A_KT[self._model_setting.L - 1][k:k + 1, 0:1] + L_dotkt[self._model_setting.L - 1][k:k + 1, 1:2],
                                                                                       np.reshape(self._hyper_params.tao0 * self.global_params.V[self._model_setting.L - 1][k], (1, 1)))
                    self.global_params.n[self._model_setting.L - 1][k] = np.sum(self.global_params.h[self._model_setting.L - 1][k, :] + np.transpose(self.global_params.h[self._model_setting.L - 1][:, k])) \
                                                                         - self.global_params.h[self._model_setting.L - 1][k, k] + L_kdott[self._model_setting.L - 1][k:k + 1, 0:1]
                    self.global_params.rou[self._model_setting.L - 1][k] = - log_max(self.global_params.q[self._model_setting.L - 1][k]) * (self.global_params.Xi[self._model_setting.L - 1] + np.sum(self.global_params.V[self._model_setting.L - 1]) - self.global_params.V[self._model_setting.L - 1][k]) \
                                                                           - np.dot(np.transpose(log_max(self.global_params.q[self._model_setting.L - 1])), self.global_params.V[self._model_setting.L - 1]) \
                                                                           + log_max(self.global_params.q[self._model_setting.L - 1][k]) * self.global_params.V[self._model_setting.L - 1][k] + self.local_params.Zeta[self._model_setting.L - 1][0]
                shape_top = self._hyper_params.gamma0 / self._model_setting.K[self._model_setting.L - 1] + self.global_params.n[self._model_setting.L - 1]
                scale_top = self.global_params.beta[self._model_setting.L - 1] + self.global_params.rou[self._model_setting.L - 1]
                self.global_params.V[self._model_setting.L - 1] = self._sampler.gamma(shape_top) / scale_top

                # Sample V[1] V[L] check
                if (self._model_setting.L > 1):
                    for l in range(self._model_setting.L - 1):
                        for k in range(self._model_setting.K[l]):
                            self.global_params.n[l][k] = np.sum(self.global_params.h[l][k, :] + np.transpose(self.global_params.h[l][:, k])) - self.global_params.h[l][k, k]
                            self.global_params.rou[l][k] = - log_max(self.global_params.q[l][k]) * (self.global_params.Xi[l] + np.sum(self.global_params.V[l]) - self.global_params.V[l][k]) \
                                                     - np.dot(log_max(np.transpose(self.global_params.q[l])), self.global_params.V[l]) + log_max(self.global_params.q[l][k] * self.global_params.V[l][k])

                        shape = self._hyper_params.gamma0 / self._model_setting.K[l] + self.global_params.n[l]
                        scale = self.global_params.beta[l] + self.global_params.rou[l]
                        self.global_params.V[l] = self._sampler.gamma(shape) / scale

            # Likelihood
            deltatmp = np.zeros([self._model_setting.K[0], self._model_setting.T])
            deltatmp[0:self._model_setting.K[0] - 1, :] = np.transpose(self.local_params.delta[0])
            Theta_hat = deltatmp * self.local_params.Theta[0]
            lambd = np.dot(self.global_params.Phi[0], self.local_params.Theta[0])

            like = np.sum(data * np.log(lambd) - lambd) / self._model_setting.V / self._model_setting.T

            end_time = time.time()
            stages = 'Training' if is_train else 'Testing'
            print(f'{stages} Stage: ',
                  f'epoch {iter:3d} takes {end_time - start_time:.2f} seconds. Likelihood:{like:8.3f}')

        return copy.deepcopy(self.local_params)


    def test(self, iter_all: int, data: np.ndarray):
        '''
        Inputs:
            iter_all   : [int] scalar, the iterations of sampling
            data       : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary of length V

        Outputs:
            local_params  : [Params] the local parameters of the probabilistic model

        '''
        local_params = self.train(iter_all, data, is_train=False)

        return local_params

    def save(self, model_path: str = './save_models'):
        '''
        Save the model to the specified directory.
        Inputs:
            model_path : [str] the directory path to save the model, default './save_models/DPGDS.npy'
        '''
        # create the directory path
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        # save the model
        model = {}
        for params in ['global_params', 'local_params', '_model_setting', '_hyper_params']:
            if params in dir(self):
                model[params] = getattr(self, params)

        np.save(model_path + '/' + self._model_name + '.npy', model)
        print('model have been saved by ' + model_path + '/' + self._model_name + '.npy')

    def load(self, model_path: str):
        '''
        Load the model parameters from the specified directory
        Inputs:
            model_path : [str] the directory path to load the model.

        '''
        assert os.path.exists(model_path), 'Path Error: can not find the path to load the model'
        model = np.load(model_path, allow_pickle=True).item()

        for params in ['global_params', 'local_params', '_model_setting', '_hyper_params']:
            if params in model:
                setattr(self, params, model[params])

        self.Para = self._hyper_params.Para

    def Sample_Pi(self, WSZS, Eta):
        Phi = self._sampler.gamma(WSZS + Eta)
        tmp = np.sum(Phi, axis=0)
        temp_dex = np.where(tmp > 0)
        temp_dex_no = np.where(tmp <= 0)
        Phi[:, temp_dex] = Phi[:, temp_dex] / (tmp[temp_dex] + realmin)
        Phi[:, temp_dex_no] = 0
        return Phi



