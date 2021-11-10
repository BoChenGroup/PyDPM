"""
===========================================
Latent Dirichlet Allocation
David M.Blei  Andrew Y.Ng  and  Michael I.Jordan
Published in Journal of Machine Learning 2003

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>; Wei Zhao <13279389260@163.com>
# License: BSD-3-Claus


import os
import copy
import time
import numpy as np

from ._basic_model import Basic_Model
from .._sampler import Basic_Sampler
from .._utils import *


class LDA(Basic_Model):
    def __init__(self, K: int, device='gpu'):
        """
        The basic model for LDA
        Inputs:
            K      : [int] number of topics in LDA;
            device : [str] 'cpu' or 'gpu';

        Attributes:
            @public:
                global_params : [Params] the global parameters of the probabilistic model
                local_params  : [Params] the local parameters of the probabilistic model

            @private:
                _model_setting : [Params] the model settings of the probabilistic model
                _hyper_params  : [Params] the hyper parameters of the probabilistic model

        """
        super(LDA, self).__init__()
        setattr(self, '_model_name', 'LDA')

        self._model_setting.K = K
        self._model_setting.device = device

        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''

        self._sampler = Basic_Sampler(self._model_setting.device)


    def initial(self, data: np.ndarray):
        '''
        Inintial the parameters of LDA with the input documents
        Inputs:
            data : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary length of V

        Attributes:
            @public:
                global_params.Phi  : [np.ndarray] V*K matrix, K topics with a vocabulary length of V

            @private:
                _model_setting.V        : [int] scalar, the length of the vocabulary
                _hyper_params.Phi_eta   : [float] scalar,
                _hyper_params.Theta_r_k : [float] scalar,

        '''
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'
        self._model_setting.V = data.shape[0]

        self._hyper_params.Phi_eta = 0.05
        self._hyper_params.Theta_r_k = np.ones([self._model_setting.K, 1])*50 / self._model_setting.K

        self.global_params.Phi = np.random.rand(self._model_setting.V, self._model_setting.K)
        self.global_params.Phi = self.global_params.Phi / np.sum(self.global_params.Phi, axis=0)


    def train(self, iter_all: int, data: np.ndarray, is_train: bool = True):
        '''
        Inputs:
            iter_all   : [int] scalar, the iterations of gibbs sampling
            data       : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary of length V
            is_train   : [bool] True or False, whether to update the global params in the probabilistic model

        Attributes:
            @public:
                local_params.Theta : [np.ndarray] N_train*K matrix, the topic propotions of N_train documents

            @private:
                _model_setting.N_        : [int] scalar, the number of the documents in the corpus
                _model_setting.Iteration : [int] scalar, the iterations of gibbs sampling

        Outputs:
                local_params  : [Params] the local parameters of the probabilistic model

        '''
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'
        self._model_setting.N = data.shape[1]
        self._model_setting.Iteration = iter_all

        # initial local parameters
        self.local_params.Theta = np.ones([self._model_setting.K, self._model_setting.N]) / self._model_setting.K

        # gibbs sampling
        for iter in range(self._model_setting.Iteration):
            start_time = time.time()

            ZSDS, WSZS = self._sampler.multi_aug(data, self.global_params.Phi, self.local_params.Theta)
            # update local params
            if iter <= 50:
                self.local_params.Theta = self._sampler.gamma(ZSDS + np.repeat(self._hyper_params.Theta_r_k, self._model_setting.N, axis=1), 0.5)
            else:
                self.local_params.Theta = np.transpose(self._sampler.dirichlet(np.transpose(ZSDS + 50 / self._model_setting.K)))

            # update global params
            if is_train:
                self.global_params.Phi = np.transpose(self._sampler.dirichlet(np.transpose(WSZS + self._hyper_params.Phi_eta)))

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


    def load(self, model_path: str):
        '''
        Load the model parameters from the specified directory
        Inputs:
            model_path : [str] the directory path to load the model;

        '''
        assert os.path.exists(model_path), 'Path Error: can not find the path to load the model'
        model = np.load(model_path, allow_pickle=True).item()

        for params in ['global_params', 'local_params', '_model_setting', '_hyper_params']:
            if params in model:
                setattr(self, params, model[params])


    def save(self, model_path: str = './save_models'):
        '''
        Save the model to the specified directory.
        Inputs:
            model_path : [str] the directory path to save the model;
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



