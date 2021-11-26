"""
===========================================
Poisson Factor Analysis
Beta-Negative Binomial Process and Poisson Factor Analysis
Mingyuan Zhou, Lauren Hannah, David Dunson, Lawrence Carin
Publihsed in International Conference on Artificial Intelligence and Statistic 2012

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

class PFA(Basic_Model):
    def __init__(self, K: int, device='cpu'):
        """
        The basic model for PFA
        Inputs:
            K      : [int] number of topics in PFA;
            device : [str] 'cpu' or 'gpu';

        Attributes:
            @public:
                global_params : [Params] the global parameters of the probabilistic model
                local_params  : [Params] the local parameters of the probabilistic model

            @private:
                _model_setting : [Params] the model settings of the probabilistic model
                _hyper_params  : [Params] the hyper parameters of the probabilistic model

        """
        super(PFA, self).__init__()
        setattr(self, '_model_name', 'PFA')

        self._model_setting.K = K
        self._model_setting.device = device

        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''

        self._sampler = Basic_Sampler(self._model_setting.device)


    def initial(self, data: np.ndarray):
        '''
        Inintial the parameters of PFA with the input documents
        Inputs:
            data : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary length of V

        Attributes:
            @public:
                global_params.Phi  : [np.ndarray] V*K matrix, K topics with a vocabulary length of V
                local_params.Theta : [np.ndarray] N*K matrix, the topic propotions of N documents

            @private:
                _model_setting.V        : [int] scalar, the length of the vocabulary
                _hyper_params.Phi_eta   : [int] scalar, the parameter in the prior of Phi
                _hyper_params.Theta_r_k : [int] scalar, the parameter in the prior of Theta
                _hyper_params.p_j_a0    : [int] scalar, the parameter in the prior of p_j
                _hyper_params.p_j_b0    : [int] scalar, the parameter in the prior of p_j
                _hyper_params.c_j_a0    : [int] scalar, the parameter in the prior of c_j
                _hyper_params.c_j_b0    : [int] scalar, the parameter in the prior of c_j

        '''
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'
        self._model_setting.V = data.shape[0]

        self.global_params.Phi = 0.2 + 0.8 * np.random.rand(self._model_setting.V, self._model_setting.K)
        self.global_params.Phi = self.global_params.Phi / np.maximum(realmin, np.sum(self.global_params.Phi, axis=0))

        self._hyper_params.Phi_eta = 0.01
        self._hyper_params.Theta_r_k = np.ones([self._model_setting.K, 1]) / self._model_setting.K
        self._hyper_params.p_j_a0 = 0.01
        self._hyper_params.p_j_b0 = 0.01
        self._hyper_params.c_j_a0 = 1
        self._hyper_params.c_j_b0 = 1


    def train(self, iter_all: int, data: np.ndarray, is_train: bool = True):
        '''
        Inputs:
            iter_all   : [int] scalar, the iterations of gibbs sampling
            data       : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary of length V
            is_train   : [bool] True or False, whether to update the global params in the probabilistic model

        Attributes:
            @public:
                local_params.Theta    : [np.ndarray] N_train*K matrix, the topic propotions of N_train documents
                local_params.c_j      : [np.ndarray] scalar,
                local_params.p_j      : [np.ndarray] scalar,

            @private:
                _model_setting.N         : [int] scalar, the number of the documents in the corpus
                _model_setting.Iteration : [int] scalar, the iterations of gibbs sampling

        Outputs:
                local_params  : [Params] the local parameters of the probabilistic model

        '''
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'
        self._model_setting.N = data.shape[1]
        self._model_setting.Iteration = iter_all

        # initial local parameters
        self.local_params.Theta = np.ones([self._model_setting.K, self._model_setting.N]) / self._model_setting.K
        self.local_params.c_j = np.ones([1, self._model_setting.N])
        self.local_params.p_j = np.ones([1, self._model_setting.N])

        # gibbs sampling
        for iter in range(self._model_setting.Iteration):
            start_time = time.time()

            # update Phi
            Xt_to_t1, WSZS = self._sampler.multi_aug(data, self.global_params.Phi, self.local_params.Theta)

            if is_train:
                self.global_params.Phi = self._update_Phi(WSZS, self._hyper_params.Phi_eta)

            # Update c_j, p_j
            self.local_params.p_j[0, :] = self._sampler.beta(Xt_to_t1.sum(0) + self._hyper_params.p_j_a0, self._hyper_params.Theta_r_k.sum(0) + self._hyper_params.p_j_b0)
            self.local_params.p_j = np.minimum(np.maximum(self.local_params.p_j, realmin), 1 - realmin)  # make sure p_j is not too large or small
            self.local_params.c_j = (1 - self.local_params.p_j) / self.local_params.p_j

            # Update Theta
            shape = np.repeat(self._hyper_params.Theta_r_k, self._model_setting.N, axis=1)
            self.local_params.Theta = self._update_Theta(Xt_to_t1, shape, self.local_params.c_j, self.local_params.p_j)

            end_time = time.time()
            stages = 'Training' if is_train else 'Testing'
            print(f'{stages} Stage: ',
                  f'epoch {iter:3d} takes {end_time - start_time:.2f} seconds')

        return copy.deepcopy(self.local_params)


    def test(self, iter_all: int, data: np.ndarray):
        '''
        Inputs:
            iter_all   : [int] scalar, the iterations of gibbs sampling
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
            model_path : [str] the directory path to save the model, default './save_models/PFA.npy'
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


    def _update_Phi(self, WSZS_t, Eta_t):
        '''
        update Phi_t at layer t
        Inputs:
            WSZS_t  : [np.ndarray]  (K_t-1)*(K_t) count matrix appearing in the likelihood of Phi_t
            Eta_t   : [np.ndarray]  scalar, the variables in the prior of Phi_t
        Outputs:
            Phi_t   : [np.ndarray]  (K_t-1)*(K_t), topic matrix at layer t

        '''
        Phi_t_shape = WSZS_t + Eta_t
        Phi_t = self._sampler.gamma(Phi_t_shape, 1)
        Phi_t = Phi_t / (Phi_t.sum(0) + realmin)

        return Phi_t


    def _update_Theta(self, Xt_to_t1_t, shape, c_j_t1, p_j_t):
        '''
        update Theta_t at layer t
        Inputs:
            Xt_to_t1_t : [np.ndarray]  (K_t-1)*(K_t) count matrix appearing in the likelihood of Phi_t
            shape      : [np.ndarray]  scalar, the variables in the prior of Phi_t
            c_j_t1     : [np.ndarray]  N * 1 vector, the variables in the scale parameter in the Theta_t+1
            p_j_t      : [np.ndarray]  N * 1 vector, the variables in the scale parameter in the Theta_t
        Outputs:
            Theta_t   : [np.ndarray]  (K_t-1)*(K_t), topic proportion matrix at layer t

        '''
        Theta_t_shape = Xt_to_t1_t + shape
        Theta_t = self._sampler.gamma(Theta_t_shape, 1) / (c_j_t1[0, :] - np.log(np.maximum(realmin, 1 - p_j_t[0, :])))

        return Theta_t

