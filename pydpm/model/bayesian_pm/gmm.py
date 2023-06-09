"""
===========================================
Gaussian Mixture Model
===========================================

"""

# Author: Xinyang Liu <lxy771258012@163.com>;
# License: BSD-3-Claus

import os
import copy
import time
import numpy as np
from scipy.stats import multivariate_normal

from ..basic_model import Basic_Model, Params
from ...sampler import Basic_Sampler
from ...utils import *


class GMM(Basic_Model):
    def __init__(self, K: int, device='gpu'):
        """
        The basic model for GMM
        Inputs:
            K      : [int] the number of component in GMM;
            device : [str] 'cpu' or 'gpu';

        Attributes:
            @public:
                global_params : [Params] the global parameters of the probabilistic model

            @private:
                _model_setting : [Params] the model settings of the probabilistic model

        """
        super(GMM, self).__init__()
        setattr(self, '_model_name', 'GMM')

        self._model_setting.K = K
        self._model_setting.device = device

        assert self._model_setting.device in ['cpu',
                                              'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''

        self._sampler = Basic_Sampler(self._model_setting.device)

    def initial(self, data: np.ndarray):
        """
        Inintial the parameters of GMM with the input data
        Inputs:
            data : [np.ndarray] N*D matrix

        Attributes:
            @public:
                global_params.Phi  : [np.ndarray] 1*N matrix
                global_params.Sigma : [np.ndarray] D*D*N matrix
                global_params.Mu :   [np.ndarray] D*k matrix
                global_params.Weight :  [np.ndarray] N*K matrix
            @private:
                _model_setting.N        : [int] scalar, the number of sample with the input data
        """
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'
        # initialise global params
        [self._model_setting.N, self._model_setting.D] = data.shape

        self.global_params.Mu = np.zeros((self._model_setting.D, self._model_setting.K))
        self.global_params.Phi = np.zeros((1, self._model_setting.K))
        self.global_params.Sigma = np.zeros((self._model_setting.D, self._model_setting.D, self._model_setting.K))
        self.global_params.Weight = np.zeros(
            (self._model_setting.N, self._model_setting.K), dtype='double') / self._model_setting.N

        rn_index = list(range(self._model_setting.N))
        np.random.shuffle(rn_index)  # random index N samples
        self.global_params.Mu = data[rn_index[0:self._model_setting.K], :].transpose()  # generate K random centroid

        x2 = np.sum(data ** 2, axis=1)
        x2.shape = (x2.shape[0], 1)
        d_mat = np.tile(x2, (1, self._model_setting.K)) + \
                np.tile(np.sum(self.global_params.Mu ** 2, axis=0), (self._model_setting.N, 1)) - \
                2 * np.dot(data, self.global_params.Mu)
        labels = d_mat.argmin(axis=1)
        for k in range(self._model_setting.K):
            bool_list = (labels == k).tolist()
            index_list = list(enumerate(bool_list))
            index_list = [i for i, x in index_list if x is True]
            data_k = data[index_list, :]
            self.global_params.Phi[0][k] = float(np.size(data_k, 0)) / self._model_setting.N
            self.global_params.Sigma[:, :, k] = np.cov(np.transpose(data_k))

    def train(self, data: np.ndarray, num_epochs: int, is_train: bool = True, is_initial_local: bool=True):
        """
        Inputs:
            data       : [np.ndarray] N*D matrix
            num_epoch   : [int] scalar, number of epochs

        Outputs:
                cluster : [list]
                global_params : [Params] the global parameters of the probabilistic model
        """

        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'
        for a in range(num_epochs):
            # e_step
            self.e_step(data)
            # m_step
            self.m_step(data)

        cluster = self.global_params.Weight.argmax(axis=1)

        return cluster, self.global_params

    def e_step(self, data):
        for i in range(self._model_setting.K):
            self.global_params.Weight[:, i] = multivariate_normal(
                self.global_params.Mu[:, i], self.global_params.Sigma[:, :, i]
            , allow_singular=True).pdf(data) * self.global_params.Phi[0][i]
        for i in range(self._model_setting.N):
            self.global_params.Weight[i, :] = self.global_params.Weight[i, :] / np.sum(
                self.global_params.Weight[i, :])

    def m_step(self, data):
        temp_sigma = np.zeros((self._model_setting.D, self._model_setting.D, self._model_setting.K))
        for i in range(self._model_setting.K):
            self.global_params.Phi[0][i] = np.sum(self.global_params.Weight[:, i]) / self._model_setting.N
            self.global_params.Mu[:, i] = np.array(np.matrix(self.global_params.Weight[:, i]) * np.matrix(
                data) / np.sum(self.global_params.Weight[:, i]))[0, :]
            temp_sigma[:, :, i] = np.dot(self.global_params.Weight[:, i] * (data - self.global_params.Mu[:, i]).T,
                                         (data - self.global_params.Mu[:, i])) / np.sum(self.global_params.Weight[:, i])
        self.global_params.Sigma = temp_sigma

    def test(self, data: np.ndarray, num_epochs: int, is_initial_local: bool=True):
        """
        Inputs:
            data       : [np.ndarray] N*D matrix
            num_epoch   : [int] scalar, number of epochs

        Outputs:
            local_params  : [Params] the local parameters of the probabilistic model

        """
        cluster, global_params = self.train(data, num_epochs=num_epochs, is_train=False, is_initial_local=is_initial_local)

        return cluster, global_params

    def load(self, model_path: str):
        """
        Load the model parameters from the specified directory
        Inputs:
            model_path : [str] the directory path to load the model;

        """
        assert os.path.exists(model_path), 'Path Error: can not find the path to load the model'
        model = np.load(model_path, allow_pickle=True).item()

        for params in ['global_params', 'local_params', '_model_setting', '_hyper_params']:
            if params in model:
                setattr(self, params, model[params])

    def save(self, model_path: str = './save_models'):
        """
        Save the model to the specified directory.
        Inputs:
            model_path : [str] the directory path to save the model;
        """
        # create the directory path
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        # save the model
        model = {}
        for params in ['global_params', 'local_params', '_model_setting', '_hyper_params']:
            if params in dir(self):
                model[params] = getattr(self, params)

        np.save(model_path + '/' + self._model_name + '.npy', model)
