"""
===========================================
Poisson Gamma Dynamical Systems
Aaron Schein, Hanna Wallach and Mingyuan Zhou
Published in Neural Information Processing Systems 2016

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

class PGDS(Basic_Model):
    def __init__(self, K: int, device='gpu'):
        """
        The basic model for PGDS
        Inputs:
            K      : [int] number of topics in PGDS;
            device : [str] 'cpu' or 'gpu';

        Attributes:
            @public:
                global_params : [Params] the global parameters of the probabilistic model
                local_params  : [Params] the local parameters of the probabilistic model

            @private:
                _model_setting : [Params] the model settings of the probabilistic model
                _hyper_params  : [Params] the hyper parameters of the probabilistic model

        """
        super(PGDS, self).__init__()
        setattr(self, '_model_name', 'PGDS')

        self._model_setting.K = K
        self._model_setting.L = 1
        self._model_setting.device = device

        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''

        self._sampler = Basic_Sampler(self._model_setting.device)


    def initial(self, data: np.ndarray):
        '''
        Inintial the parameters of PGDS with the input documents
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

        self.global_params.Phi = np.random.rand(self._model_setting.V, self._model_setting.K)
        self.global_params.Phi = self.global_params.Phi / np.sum(self.global_params.Phi, axis=0)
        self.global_params.Pi = np.eye(self._model_setting.K)
        self.global_params.Xi = 1
        self.global_params.V = np.ones((self._model_setting.K, 1))
        self.global_params.beta = 1
        self.global_params.h = np.zeros((self._model_setting.K, self._model_setting.K))
        self.global_params.n = np.zeros((self._model_setting.K, 1))
        self.global_params.rou = np.zeros((self._model_setting.K, 1))

        self._hyper_params.tao0 = 1
        self._hyper_params.gamma0 = 100
        self._hyper_params.eta0 = 0.1
        self._hyper_params.epilson0 = 0.1

    def train(self, iter_all: int, data: np.ndarray, is_train: bool = True):
        '''
        Inputs:
            iter_all   : [int] scalar, the iterations of gibbs sampling
            data       : [np.ndarray] V*N_train matrix, N_train bag-of-words vectors with a vocabulary length of V
            is_train   : [bool] True or False, whether to update the global params in the probabilistic model

        Attributes:
            @public:
                local_params.Theta : [np.ndarray] N_train*K matrix, the topic propotions of N_train documents

            @private:
                _model_setting.T         : [int] scalar, the number of the documents in the corpus
                _model_setting.Burnin    : [int] scalar, the burunin iterations of sampling
                _model_setting.Collection: [int] scalar, the Collection iterations of sampling
                _hyper_params.Para       : [dict] scalar,
        Outputs:
                local_params  : [Params] the local parameters of the probabilistic model

        '''
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'

        self._model_setting.T = data.shape[1]

        self.local_params.Theta = np.ones((self._model_setting.K, self._model_setting.T)) / self._model_setting.K
        self.local_params.Zeta = np.zeros((self._model_setting.T + 1, 1))
        self.local_params.delta = np.ones((self._model_setting.T, 1))


        A_VK = np.zeros((self._model_setting.V, self._model_setting.K))
        A_KT = np.zeros((self._model_setting.K, self._model_setting.T))
        L_dotkt = np.zeros((self._model_setting.K, self._model_setting.T + 1))
        L_kdott = np.zeros((self._model_setting.K, self._model_setting.T + 1))


        for iter in range(iter_all):

            start_time = time.time()

            A_KT, A_VK = self._sampler.multi_aug(data, self.global_params.Phi, self.local_params.Theta)

            L_KK = np.zeros((self._model_setting.K, self._model_setting.K))
            for t in range(self._model_setting.T - 1, 0, -1):  # T-1 : 1
                tmp1 = A_KT[:, t:t + 1] + L_dotkt[:, t + 1:t + 2]
                L_kdott[:, t:t + 1] = self._sampler.crt(tmp1, self._hyper_params.tao0 * np.dot(self.global_params.Pi, self.local_params.Theta[:, t - 1:t]))
                L_dotkt[:, t:t + 1], tmp = self._sampler.multi_aug(L_kdott[:, t:t + 1], self.global_params.Pi, self.local_params.Theta[:, t - 1:t])
                L_KK = L_KK + tmp

            if is_train:
                # Sample Phi
                self.global_params.Phi = self.Sample_Pi(A_VK, self._hyper_params.eta0)

                # Sample Pi
                Piprior = np.dot(self.global_params.V, np.transpose(self.global_params.V))
                Piprior[np.arange(Piprior.shape[0]), np.arange(Piprior.shape[1])] = 0
                Piprior = Piprior + np.diag(np.reshape(self.global_params.Xi * self.global_params.V, [self.global_params.V.shape[0]]))
                self.global_params.Pi = self.Sample_Pi(L_KK, Piprior)

            # Calculate Zeta   
            if (self._model_setting.Stationary == 1):
                for t in range(self._model_setting.T - 1, -1, -1):
                    self.local_params.Zeta[t] = np.log(1 + self.local_params.Zeta[t + 1] + self.local_params.delta[t])

            # Sample Theta  checked
            for t in range(self._model_setting.T):
                if t == 0:
                    shape = A_KT[:, t:t + 1] + L_dotkt[:, t + 1:t + 2] + self._hyper_params.tao0 * self.global_params.V
                else:
                    shape = A_KT[:, t:t + 1] + L_dotkt[:, t + 1:t + 2] + self._hyper_params.tao0 * np.dot(self.global_params.Pi, self.local_params.Theta[:, t - 1:t])

                scale = self.local_params.delta[t] + self._hyper_params.tao0 + self._hyper_params.tao0 * self.local_params.Zeta[t + 1];
                self.local_params.Theta[:, t:t + 1] = self._sampler.gamma(shape) / scale

            if is_train:
                # Sample Beta check
                shape = self._hyper_params.epilson0 + self._hyper_params.gamma0
                scale = self._hyper_params.epilson0 + np.sum(self.global_params.V)
                self.global_params.beta = self._sampler.gamma(shape) / scale

                # Sample q checked
                a = np.sum(L_dotkt, axis=1, keepdims=1)
                a[a == 0] = 1e-10
                b = self.global_params.V * (self.global_params.Xi + np.repeat(np.sum(self.global_params.V), self._model_setting.K, axis=0).reshape([self._model_setting.K, 1]) - self.global_params.V)
                b[b == 0] = 1e-10  #
                self.global_params.q = np.maximum(self._sampler.beta(b, a), 0.00001)

                # Sample h checked
                for k1 in range(self._model_setting.K):
                    for k2 in range(self._model_setting.K):
                        self.global_params.h[k1:k1 + 1, k2:k2 + 1] = self._sampler.crt(L_KK[k1:k1 + 1, k2:k2 + 1], Piprior[k1:k1 + 1, k2:k2 + 1])

                # Sample Xi checked
                shape = self._hyper_params.gamma0 / self._model_setting.K + np.trace(self.global_params.h)
                scale = self.global_params.beta - np.dot(np.transpose(self.global_params.V), log_max(self.global_params.q))
                self.global_params.Xi = self._sampler.gamma(shape) / scale

                # Sample V # check
                for k in range(self._model_setting.K):
                    L_kdott[k:k + 1, 0:1] = self._sampler.crt(A_KT[k:k + 1, 0:1] + L_dotkt[k:k + 1, 1:2],
                                                            np.reshape(self._hyper_params.tao0 * self.global_params.V[k], (1, 1)))
                    self.global_params.n[k] = np.sum(self.global_params.h[k, :] + np.transpose(self.global_params.h[:, k])) \
                                               - self.global_params.h[k, k] + L_kdott[k:k + 1, 0:1]

                    self.global_params.rou[k] = - log_max(self.global_params.q[k]) * (self.global_params.Xi + np.sum(self.global_params.V) - self.global_params.V[k]) \
                                                 - np.dot(np.transpose(log_max(self.global_params.q)), self.global_params.V) \
                                                 + log_max(self.global_params.q[k]) * self.global_params.V[k] + self.local_params.Zeta[0]

                shape_top = self._hyper_params.gamma0 / self._model_setting.K + self.global_params.n
                scale_top = self.global_params.beta + self.global_params.rou
                self.global_params.V = self._sampler.gamma(shape_top) / scale_top

            # Likelihood
            deltatmp = np.zeros([self._model_setting.K, self._model_setting.T])
            deltatmp[0:self._model_setting.K - 1, :] = np.transpose(self.local_params.delta)
            Theta_hat = deltatmp * self.local_params.Theta
            lambd = np.dot(self.global_params.Phi, self.local_params.Theta)
            # lambd = np.dot(self._hyper_params.Para['Phi'], self._hyper_params.Para['Theta'])

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
            model_path : [str] the directory path to save the model, default './save_models/PGDS.npy'
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

    def Sample_Pi(self, WSZS, Eta):
        Phi = self._sampler.gamma(WSZS + Eta)
        tmp = np.sum(Phi, axis=0)
        temp_dex = np.where(tmp > 0)
        temp_dex_no = np.where(tmp <= 0)
        Phi[:, temp_dex] = Phi[:, temp_dex] / (tmp[temp_dex] + realmin)
        Phi[:, temp_dex_no] = 0
        return Phi

