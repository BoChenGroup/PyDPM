"""
===========================================
Convolutional Poisson Gamma Belief Network
Chaojie Wang  Sucheng Xiao  Bo Chen  and  Mingyuan Zhou
Published in International Conference on Machine Learning 2019

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>
# License: BSD-3-Clause

import time
import os
import copy
import numpy as np

from ..basic_model import Basic_Model, Params
from ...sampler import Basic_Sampler
from ...utils import *


class CPGBN(Basic_Model):

    def __init__(self, K: list, device='gpu'):
        """
        The basic model for CPGBN
        Inputs:
            K      : [list] number of topics at diffrent layers in CPGBN;
            device : [str] 'cpu' or 'gpu';

        Attributes:
            @public:
                global_params : [Params] the global parameters of the probabilistic model
                local_params  : [Params] the local parameters of the probabilistic model

            @private:
                _model_setting   : [Params] the model settings of the probabilistic model
                _hyper_params    : [Params] the hyper parameters of the probabilistic model
                _model_setting.T : [int] the network depth

        """
        super(CPGBN, self).__init__()
        setattr(self, '_model_name', 'CPGBN')

        self._model_setting.K = K
        self._model_setting.T = len(K)
        self._model_setting.device = device

        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''
        assert self._model_setting.T > 1, 'Network Structure Error: the network depth should be larger than 1'

        self._sampler = Basic_Sampler(self._model_setting.device)


    def initial(self, data:list, is_sparse: bool=False):
        '''
        Inintial the parameters of CPGBN with the input documents
        Inputs:
            dense matrix representation
                args[0] : [np.ndarray] N*V*L matrix, N documents represented as V*L sparse matrices

            sparse matrix representation
                args[0] : [list] N documents under the sparse representation, batch.rows, batch.cols, batch.values
                args[1] : [list] the shape of the input document matrix, N*V*L

        Attributes:
            @public:
                global_params.D_k  : [np.ndarray] K*1*S3*S4, K probabilistic convolutional kernels, where the size of each kernel is S3*S4

            @private:
                _model_setting.V        : [int] scalar, the length of the vocabulary
                _model_setting.L        : [int] scalar, the max length of the document
                _hyper_params.D_k_eta   : [float] scalar, parameter in the prior of D_k
                _hyper_params.W_nk_gamma: [float] scalar, parameter in the prior of W_nk
                _hyper_params.p_n_a0    : [float] scalar, parameter in the prior of p_n
                _hyper_params.p_n_b0    : [float] scalar, parameter in the prior of p_n
                _hyper_params.c_n_e0    : [float] scalar, parameter in the prior of c_n
                _hyper_params.c_n_f0    : [float] scalar, parameter in the prior of c_n
                _hyper_params.p_j_a0    : [float] scalar, parameter in the prior of p_j
                _hyper_params.p_j_b0    : [float] scalar, parameter in the prior of p_j
                _hyper_params.c_j_e0    : [float] scalar, parameter in the prior of c_j
                _hyper_params.c_j_f0    : [float] scalar, parameter in the prior of c_j
        '''
        # assert len(args) <= 2, 'Data type error: the input dataset should be a 3-D np.ndarray or two lists to store the input dataset under ' \
        #                       'the sparse representation'
        # if len(args) == 1:
        #     _, self._model_setting.V, self._model_setting.L = args[0].shape
        # elif len(args) == 2:
        #     _, self._model_setting.V, self._model_setting.L = args[1]

        if not is_sparse:
            assert len(data) == 1, Warning('The length of dense data list should be 1')
            _, self._model_setting.V, self._model_setting.L = data[0].shape
        else:
            assert len(data) == 2, Warning('The length of sparse data list should be 2')
            _, self._model_setting.V, self._model_setting.L = data[1]

        self._model_setting._structure = Params()
        _structure = self._model_setting._structure
        _structure.K_V1 = self._model_setting.V
        _structure.K_V2 = self._model_setting.L + 2  # padding
        _structure.K_S3 = self._model_setting.V
        _structure.K_S4 = 3
        _structure.K_S1 = _structure.K_V1 + 1 - _structure.K_S3
        _structure.K_S2 = _structure.K_V2 + 1 - _structure.K_S4

        # convolutional layers
        self._hyper_params.D_k_eta = 0.05
        self._hyper_params.W_nk_gamma = 0.1 * np.ones([self._model_setting.K[0], _structure.K_S1, _structure.K_S2])
        self._hyper_params.p_n_a0 = 0.1
        self._hyper_params.p_n_b0 = 0.1
        self._hyper_params.c_n_e0 = 0.1
        self._hyper_params.c_n_f0 = 0.1

        self.global_params.D_k = np.random.rand(self._model_setting.K[0], 1, _structure.K_S3, _structure.K_S4)
        for k in range(self._model_setting.K[0]):
            self.global_params.D_k[k, :, :, :] = self.global_params.D_k[k, :, :, :] / np.sum(self.global_params.D_k[k, :, :, :])

        # fully-connected layers
        self.global_params.Phi = []
        self._hyper_params.Phi_eta = []
        self._hyper_params.Theta_r_k = np.ones([self._model_setting.K[self._model_setting.T - 1], 1]) / self._model_setting.K[self._model_setting.T - 1]
        self._hyper_params.p_j_a0 = 0.1
        self._hyper_params.p_j_b0 = 0.1
        self._hyper_params.c_j_e0 = 0.1
        self._hyper_params.c_j_f0 = 0.1

        for t_phi in range(self._model_setting.T - 1):
            self._hyper_params.Phi_eta.append(0.05)
            self.global_params.Phi.append(0.2 + 0.8 * np.random.rand(self._model_setting.K[t_phi], self._model_setting.K[t_phi + 1]))
            self.global_params.Phi[t_phi] = self.global_params.Phi[t_phi] / np.maximum(realmin, self.global_params.Phi[t_phi].sum(0))


    def train(self, data:list, is_sparse: bool=False, iter_all: int=1, is_train: bool=True, is_initial_local: bool=True):
        '''
        Inputs:
            iter_all   : [int] scalar, the iterations of gibbs sampling
            dense matrix representation
                args[0] : [np.ndarray] N*V*L matrix, N documents represented as V*L sparse matrices

            sparse matrix representation
                args[0] : [list] N documents under the sparse representation, batch.rows, batch.cols, batch.values
                args[1] : [list] the shape of the input document matrix, N*V*L
            **kwargs:
                kwargs['is_train']: [bool] True or False, whether to update the global params in the probabilistic model

        Attributes:
            @public:
                local_params.W_nk     : [np.ndarray] N_train*K*N_S1*N_S2 matrix, N_train*K feature maps with a size of N_S1*N_S2
                local_params.c_n      : [np.ndarray] N_train vector, the variable in the scale parameter of W_nk
                local_params.p_n      : [np.ndarray] N_train vector, the variable in the scale parameter of W_nk
                local_params.c_j      : [np.ndarray] N_train vector, the variable in the scale parameter of Theta_nk
                local_params.p_j      : [np.ndarray] N_train vector, the variable in the scale parameter of Theta_nk

            @private:
                _model_setting.N         : [int] scalar, the number of the documents in the corpus
                _model_setting.Iteration : [int] scalar, the iterations of gibbs sampling

        Outputs:
                local_params  : [Params] the local parameters of the probabilistic model

        '''
        # assert len(args) <= 2, 'Data type error: the input dataset should be a 3-D np.ndarray or two lists to store the input dataset under ' \
        #                       'the sparse representation'
        # if len(args) == 1:
        #     self._model_setting.N, self._model_setting.V, self._model_setting.L = args[0].shape
        #     batch_file_indices, batch_rows, batch_cols = np.where(args[0])
        #     batch_values = args[0][batch_file_indices, batch_rows, batch_cols]
        # elif len(args) == 2:
        #     self._model_setting.N, self._model_setting.V, self._model_setting.L = args[1]
        #     batch_file_indices, batch_rows, batch_cols, batch_values = args[0]
        # if 'is_train' in kwargs:
        #     is_train = kwargs['is_train']
        # else:
        #     is_train = True

        if not is_sparse:
            assert len(data) == 1, Warning('The length of dense data list should be 1')
            self._model_setting.N, self._model_setting.V, self._model_setting.L = data[0].shape
            batch_file_indices, batch_rows, batch_cols = np.where(data[0])
            batch_values = data[0][batch_file_indices, batch_rows, batch_cols]
        else:
            assert len(data) == 2, Warning('The length of sparse data list should be 2')
            batch_rows, batch_cols, batch_file_indices, batch_values = data[0]
            self._model_setting.N, self._model_setting.V, self._model_setting.L = data[1]

        self._model_setting.Iteration = iter_all
        _structure = self._model_setting._structure

        # initial local parameters in convolutional layers
        if is_initial_local or not hasattr(self.local_params, 'W_nk') or hasattr(self.local_params, 'c_n') or hasattr(self.local_params, 'p_n'):
            self.local_params.W_nk = np.random.rand(self._model_setting.N, self._model_setting.K[0], _structure.K_S1, _structure.K_S2)
            self.local_params.c_n = 1 * np.ones([self._model_setting.N])
            self.local_params.p_n = 1 / (1 + self.local_params.c_n)

            # initial local parameters in fully-connected layers
            self.local_params.Theta = []
            self.local_params.c_j = []
            self.local_params.p_j = []

            for t_phi in range(self._model_setting.T - 1):
                self.local_params.Theta.append(np.ones([self._model_setting.K[t_phi+1], self._model_setting.N]) / self._model_setting.K[t_phi+1])
                self.local_params.c_j.append(1 * np.ones([self._model_setting.N]))
                if t_phi == 0:
                    tmp = -log_max(1 - self.local_params.p_n)
                else:
                    tmp = -log_max(1 - self.local_params.p_j[t_phi-1])
                self.local_params.p_j.append(tmp / (tmp + self.local_params.c_j[t_phi]))

        # gibbs sampling
        Xt_to_t1 = [0] * (self._model_setting.T - 1) #[0 for i in range(self._model_setting.T - 1)]
        WSZS = [0] * (self._model_setting.T - 1) #[0 for i in range(self._model_setting.T - 1)]

        for iter in range(self._model_setting.Iteration):
            start_time = time.time()

            # dataset augmentation
            W_nk_aug, D_k_aug = self._sampler.conv_multi_aug(batch_rows, batch_cols, batch_file_indices, batch_values,
                                                              self.global_params.D_k, self.local_params.W_nk)
            W_nk_Aug_Pool = np.transpose(np.sum(np.sum(W_nk_aug, axis=3), axis=2))  # K*N

            for t_phi in range(self._model_setting.T - 1):
                if t_phi == 0:
                    Xt_to_t1[t_phi], WSZS[t_phi] = self._sampler.crt_multi_aug(W_nk_Aug_Pool, self.global_params.Phi[t_phi], self.local_params.Theta[t_phi])
                else:
                    Xt_to_t1[t_phi], WSZS[t_phi] = self._sampler.crt_multi_aug(Xt_to_t1[t_phi - 1], self.global_params.Phi[t_phi], self.local_params.Theta[t_phi])

            # update global_params
            if is_train:
                for k in range(self._model_setting.K[0]):
                    self.global_params.D_k[k, :, :, :] = (D_k_aug[k, :, :, :] + self._hyper_params.D_k_eta) / np.sum(D_k_aug[k, :, :, :] + self._hyper_params.D_k_eta)

                for t_phi in range(self._model_setting.T - 1):
                    self.global_params.Phi[t_phi] = (WSZS[t_phi] + self._hyper_params.Phi_eta[t_phi]) / np.sum(WSZS[t_phi] + self._hyper_params.Phi_eta[t_phi], axis=0)

            # update local_params
            self.local_params.c_n = self._sampler.gamma(self._hyper_params.c_n_e0 + np.sum(self._hyper_params.W_nk_gamma))
            self.local_params.c_n = self.local_params.c_n / (self._hyper_params.c_n_f0 + np.sum(np.sum(np.sum(W_nk_aug, axis=3), axis=2), axis=1) + realmin)
            self.local_params.p_n = 1 / (self.local_params.c_n + 1)
            self.local_params.W_nk = self._sampler.gamma(W_nk_aug + self._hyper_params.W_nk_gamma) / (1 + self.local_params.c_n)[:, np.newaxis, np.newaxis, np.newaxis]

            for t_phi in range(self._model_setting.T - 1):
                if t_phi == self._model_setting.T - 2:
                    self.local_params.c_j[t_phi] = self._sampler.gamma(self._hyper_params.c_j_e0 + np.sum(self._hyper_params.Theta_r_k, axis=0))
                else:
                    self.local_params.c_j[t_phi] = self._sampler.gamma(self._hyper_params.c_j_e0 + np.sum(self.local_params.Theta[t_phi+1], axis=0))
                self.local_params.c_j[t_phi] = self.local_params.c_j[t_phi] / (self._hyper_params.c_j_f0 + np.sum(self.local_params.Theta[t_phi], axis=0) + realmin)

                if t_phi == 0:
                    tmp = -log_max(1 - self.local_params.p_n)
                else:
                    tmp = -log_max(1 - self.local_params.p_j[t_phi - 1])
                self.local_params.p_j[t_phi] = tmp / (tmp + self.local_params.c_j[t_phi])

            for t_phi in range(self._model_setting.T - 2, -1, -1):
                if t_phi == self._model_setting.T - 2:
                    shape = self._hyper_params.Theta_r_k
                else:
                    shape = np.dot(self.global_params.Phi[t_phi + 1], self.local_params.Theta[t_phi + 1])
                Theta = self._sampler.gamma(Xt_to_t1[t_phi] + shape)

                if t_phi == 0:
                    self.local_params.Theta[t_phi] = Theta / (-log_max(1 - self.local_params.p_n) + self.local_params.c_j[0])
                else:
                    self.local_params.Theta[t_phi] = Theta / (-log_max(1 - self.local_params.p_j[t_phi - 1]) + self.local_params.c_j[t_phi])

            end_time = time.time()
            stages = 'Training' if is_train else 'Testing'
            print(f'{stages} Stage: ',
                  f'epoch {iter:3d} takes {end_time - start_time:.2f} seconds')

        return copy.deepcopy(self.local_params)


    def test(self, data: list, is_sparse: bool=False, iter_all: int=1, is_initial_local: bool=True):
        '''
        Inputs:
            iter_all   : [int] scalar, the iterations of gibbs sampling
            dense matrix representation
                args[0] : [np.ndarray] N*V*L matrix, N documents represented as V*L sparse matrices

            sparse matrix representation
                args[0] : [list] N documents under the sparse representation, batch.rows, batch.cols, batch.values
                args[1] : [list] the shape of the input document matrix, N*V*L

        '''
        local_params = self.train(data, is_sparse=is_sparse, iter_all=iter_all, is_train=False, is_initial_local=is_initial_local)

        return local_params


    def save(self, model_path: str = '../save_models'):
        '''
        Save the model to the specified directory.
        Inputs:
            model_path : [str] the directory path to save the model, default '../save_models/PFA.npy'
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
        model['_model_setting'].device = self._model_setting.device
        for params in ['global_params', 'local_params', '_model_setting', '_hyper_params']:
            if params in model:
                setattr(self, params, model[params])
