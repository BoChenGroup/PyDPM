"""
===========================================
Deep Relational Topic Modeling via Graph Poisson Gamma Belief Network
Chaojie Wang, Hao Zhang, Bo Chen, Dongsheng Wang, Zhengjue Wang
Published in Advances in Neural Information Processing System 2020

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Wei Zhao <13279389260@163.com>; Jiawen Wu <wjw19960807@163.com>
# License: BSD-3-Clause

import os
import copy
import time
import numpy as np

from ._basic_model import Basic_Model
from .._sampler import Basic_Sampler
from .._utils import *


class GPGBN(Basic_Model):
    def __init__(self, K: list = [128, 64, 32], device='gpu'):
        """
        The basic model for GPGBN
        Inputs:
            K      : [list] number of topics at diffrent layers in PGBN;
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
        super(GPGBN, self).__init__()
        setattr(self, '_model_name', 'GPGBN')

        self._model_setting.K = K
        self._model_setting.T = len(K)
        self._model_setting.device = device

        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''

        self._sampler = Basic_Sampler(self._model_setting.device)

    def initial(self, data: np.ndarray):
        '''
        Initial the parameters of PGBN with the input documents
        Inputs:
            data : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary length of V

        Attributes:
            @public:
                global_params.Phi  : [list] T (K_t-1)*(K_t) factor loading matrices at different layers

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

        self.global_params.Phi = []
        self.global_params.U = []
        self._hyper_params.Phi_eta = []
        self._hyper_params.Theta_r_k = np.ones([self._model_setting.K[self._model_setting.T - 1], 1]) / self._model_setting.K[self._model_setting.T - 1]
        self._hyper_params.eta = np.ones(self._model_setting.T) * 0.01
        self._hyper_params.p_j_a0 = 0.01
        self._hyper_params.p_j_b0 = 0.01
        self._hyper_params.c_j_e0 = 1
        self._hyper_params.c_j_f0 = 1  # TODO 最后四个超参对应的代码块 update c_j 被注释掉了

        for t in range(self._model_setting.T):
            self._hyper_params.Phi_eta.append(self._hyper_params.eta[t])
            if t == 0:
                self.global_params.Phi.append(0.2 + 0.8 * np.random.rand(self._model_setting.V, self._model_setting.K[t]))
            else:
                self.global_params.Phi.append(0.2 + 0.8 * np.random.rand(self._model_setting.K[t-1], self._model_setting.K[t]))
            self.global_params.Phi[t] = self.global_params.Phi[t] / np.maximum(realmin, self.global_params.Phi[t].sum(0))
            self.global_params.U.append(self._sampler.gamma(1 * np.ones([self._model_setting.K[t], 1])) / (1 * np.ones([self._model_setting.K[t], 1])))


    def train(self, iter_all: int, data:np.ndarray, data_A:np.ndarray, is_train: bool = True):
        '''
        Inputs:
            iter_all   : [int] scalar, the iterations of gibbs sampling
            data       : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary of length V
            is_train   : [bool] True or False, whether to update the global params in the probabilistic model

        Attributes:
            @public:
                local_params.Theta : [list] T (K_t)*(N) topic proportions at different layers
                local_params.c_j   : [list] T+1 1*N vector, the variables in the scale parameter in the Theta
                local_params.p_j   : [list] T+1 1*N vector, the variables in the scale parameter in the Theta

            @private:
                _model_setting.N         : [int] scalar, the number of the documents in the corpus
                _model_setting.Iteration : [int] scalar, the iterations of gibbs sampling

        Outputs:
            local_params  : [Params] the local parameters of the probabilistic model

        '''
        # todo 1.调整注释  2.train/test  3.展示部分代码  4.cosine_simlar
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'
        self._model_setting.N = data.shape[1]
        self._model_setting.Iteration = iter_all

        # initial local params
        self.local_params.Theta = []
        self.local_params.c_j = []
        for t in range(self._model_setting.T):  # from layer 1 to T
            self.local_params.Theta.append(np.ones([self._model_setting.K[t], self._model_setting.N]) / self._model_setting.K[t])
            self.local_params.c_j.append(np.ones([1, self._model_setting.N]))
        self.local_params.c_j.append(np.ones([1, self._model_setting.N]))
        self.local_params.p_j = self._calculate_pj(self.local_params.c_j, self._model_setting.T)

        Xt_to_t1 = []
        WSZS = []

        self.local_params.Sigma = []
        for t in range(self._model_setting.T):
            Xt_to_t1.append(np.zeros(self.local_params.Theta[t].shape))
            WSZS.append(np.zeros(self.global_params.Phi[t].shape))
            self.local_params.Sigma.append(self._sampler.gamma(1 * np.ones([1, self._model_setting.N])) / (1 * np.ones([1, self._model_setting.N])))

        # gibbs sampling
        LH_list = []
        LH_graph_list = []
        U_all = np.zeros([np.sum(self._model_setting.K), self._model_setting.Iteration])

        for iter in range(self._model_setting.Iteration):
            start_time = time.time()

            # update global params: Phi
            for t in range(self._model_setting.T):  # from layer 1 to T
                if t == 0:
                    Xt_to_t1[t], WSZS[t] = self._sampler.multi_aug(data, self.global_params.Phi[t], self.local_params.Theta[t])
                else:
                    Xt_to_t1[t], WSZS[t] = self._sampler.crt_multi_aug(Xt_to_t1[t-1], self.global_params.Phi[t], self.local_params.Theta[t])

                # Xt1 = self._sampler.crt(Xt_to_t1[t-1].astype('double'), np.dot(self.global_params.Phi[t], self.local_params.Theta[t]))
                # Xt_to_t1[t], WSZS[t] = self._sampler.multi_aug(Xt1.astype('double'), self.global_params.Phi[t], self.local_params.Theta[t])

                if is_train:  # todo 源作不分train，且注释了两行
                    self.global_params.Phi[t] = self._update_Phi(WSZS[t], self._hyper_params.Phi_eta[t])


            # downward pass
            M = np.reshape(data_A, [self._model_setting.N * self._model_setting.N, 1])
            M_rate_k = np.zeros([self._model_setting.N * self._model_setting.N, np.sum(np.array(self._model_setting.K))])
            Theta_inter_k = np.zeros([self._model_setting.N * self._model_setting.N, np.sum(np.array(self._model_setting.K))])

            for t in range(self._model_setting.T):
                theta_t = self.local_params.Theta[t]  # K*N
                u_t = self.global_params.U[t]
                sigma_t = self.local_params.Sigma[t]
                start_index = int(np.sum(self._model_setting.K[:t]))
                for k in range(self._model_setting.K[t]):
                    theta_k = theta_t[k: k+1, :].copy()  # todo matmul 有没有加速？
                    theta_inter_k = np.matmul(np.transpose(theta_k * sigma_t), theta_k * sigma_t)
                    theta_inter_k[(np.arange(self._model_setting.N), np.arange(self._model_setting.N))] = 0
                    Theta_inter_k[:, start_index + k: start_index + k + 1] = np.reshape(theta_inter_k, [-1, 1])
                    M_rate_k[:, start_index + k: start_index + k + 1] = np.reshape(theta_inter_k * u_t[k, 0], [-1, 1])

                    # m_rate_k = np.matmul(np.transpose(theta_k*u_t[k, 0]), theta_k)  # theta_k' * U * theta_k
                    # m_rate_k[(np.arange(self._model_setting.N), np.arange(self._model_setting.N))] = 0  # diagnol is zeros
                    # M_rate_k[:, start_index + k: start_index + k+1] = np.reshape(m_rate_k, [-1, 1])

            M_rate_k = M_rate_k / (np.sum(M_rate_k, 1, keepdims=True) + realmin)  # (N*N)*K
            [M_kn, M_vk] = self._sampler.multi_aug(M.astype('double'), M_rate_k, np.ones([np.sum(np.array(self._model_setting.K)), 1]))
            M_ijk = np.reshape(M_vk, [self._model_setting.N, self._model_setting.N, np.sum(np.array(self._model_setting.K))])

            # update c_j todo 源码注释了
            # if iter > 10:
            #     if self._model_setting.T > 1:
            #         for n in range(self._model_setting.N):
            #             self.local_params.p_j[1][0, n] = np.random.beta(Xt_to_t1[0][:, n].sum(0) + self._hyper_params.p_j_a0,
            #                                           self.local_params.Theta[1][:, n].sum(0) + self._hyper_params.p_j_b0)
            #     else:
            #         for n in range(self._model_setting.N):
            #             self.local_params.p_j[1][0, n] = np.random.beta(Xt_to_t1[0][:, n].sum(0) + self._hyper_params.p_j_a0,
            #                                           self._hyper_params.Theta_r_k.sum(0) + self._hyper_params.p_j_b0)
            #     self.local_params.p_j[1] = np.minimum(np.maximum(self.local_params.p_j[1], realmin), 1 - realmin)  # make sure p_j is no so large or so small
            #     self.local_params.c_j[1][:, :] = (1 - self.local_params.p_j[1]) / self.local_params.p_j[1]
            #
            #     for t in [i for i in range(self._model_setting.T + 1) if i > 1]:  # only T>=2 works  ==> for t = 3:T+1
            #         if t == self._model_setting.T:
            #             for n in range(self._model_setting.N):
            #                 self.local_params.c_j[t][0, n] = np.random.gamma(self._hyper_params.Theta_r_k.sum(0) + self._hyper_params.c_j_e0, 1)\
            #                                                  / (self.local_params.Theta[t - 1][:, n].sum(0) + self._hyper_params.c_j_f0)
            #         else:
            #             for n in range(self._model_setting.N):
            #                 self.local_params.c_j[t][0, n] = np.random.gamma(self.local_params.Theta[t][:, n].sum(0) + self._hyper_params.c_j_e0, 1)\
            #                                                  / (self.local_params.Theta[t - 1][:, n].sum(0) + self._hyper_params.c_j_f0)
            #     p_j_tmp = self._sampler._calculate_pj(self.local_params.c_j, self._model_setting.T)
            #     self.local_params.p_j[2:] = p_j_tmp[2:]

            # update U
            M_kj = np.transpose(np.sum(M_ijk, axis=0))  # K*N
            M_kjt = []

            tmp_start = 0
            tmp_end = 0
            for t in range(self._model_setting.T):

                tmp_end += self._model_setting.K[t]
                start_index = int(np.sum(self._model_setting.K[: t]))
                end_index = int(np.sum(self._model_setting.K[: t+1]))

                M_kjt.append(M_kj[start_index:end_index, :])  # K*N
                theta_inter_kjt = Theta_inter_k[:, start_index:end_index]

                M_k = np.sum(M_kjt[t], axis=1) / 2  # K*1
                theta_inter_k = np.sum(theta_inter_kjt, axis=0) / 2  # K*1
                self.global_params.U[t][:, 0] = self._sampler.gamma(M_k + 1) / (1 + theta_inter_k)

                U_all[tmp_start: tmp_end, iter] = self.global_params.U[t][:, 0]

                tmp_start = tmp_end

            # update Theta
            for t in range(self._model_setting.T - 1, -1, -1):  # for t = T:-1 :1

                if t == self._model_setting.T - 1:
                    shape = np.repeat(self._hyper_params.Theta_r_k, self._model_setting.N, axis=1)
                else:
                    shape = np.dot(self.global_params.Phi[t + 1], self.local_params.Theta[t + 1])

                for j in range(self._model_setting.N):
                    sigma_t = self.local_params.Sigma[t].copy()  # Note this copy here
                    theta_t = self.local_params.Theta[t].copy()  # Note this copy here
                    theta_t[:, j] = 0
                    theta_sigma_t = theta_t * sigma_t  # K*N

                    self.local_params.Theta[t][:, j] = self._sampler.gamma(Xt_to_t1[t][:, j] + M_kjt[t][:, j] + shape[:, j], 1)
                    self.local_params.Theta[t][:, j:j + 1] = self.local_params.Theta[t][:, j:j + 1] / (1 + self.local_params.c_j[t][:, j:j + 1] +
                                                                   sigma_t[:, j:j + 1] * self.global_params.U[t] * np.sum(theta_sigma_t, axis=1, keepdims=True))

                    self.local_params.Sigma[t][:, j:j + 1] = self._sampler.gamma(np.sum(M_kjt[t][:, j:j + 1], keepdims=True) + 1, 1)
                    self.local_params.Sigma[t][:, j:j + 1] = self.local_params.Sigma[t][:, j:j + 1] / (np.sum(self.local_params.Theta[t][:, j:j + 1] * self.global_params.U[t] * np.sum(theta_sigma_t, axis=1, keepdims=True)) + 1)

            end_time = time.time()

            stages = 'Training' if is_train else 'Testing'
            print(f'{stages} Stage: ', f'epoch {iter:3d} takes {end_time - start_time:.2f} seconds')

            re = np.dot(self.global_params.Phi[0], self.local_params.Theta[0])
            LH = np.sum(data * log_max(re) - re) #- log_max(gamma(data+1)))
            print("    Data Likelihood " + str(LH / self._model_setting.N))
            LH_list.append(LH / self._model_setting.N)

            re_graph = 0
            for t in range(self._model_setting.T):
                re_graph += np.matmul(np.transpose(self.local_params.Sigma[t] * self.local_params.Theta[t] * self.global_params.U[t]), self.local_params.Sigma[t] * self.local_params.Theta[t])
            re_graph[np.arange(self._model_setting.N), np.arange(self._model_setting.N)] = 0
            LH_graph = np.sum(data_A * log_max(re_graph) - re_graph) #- log_max(gamma(data_A+1)))
            print("    Graph Likelihood " + str(LH_graph / self._model_setting.N))
            LH_graph_list.append(LH_graph / self._model_setting.N)

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
            model_path : [str] the directory path to load the model.

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
            model_path : [str] the directory path to save the model, default './save_models/PGBN.npy'
        '''
        # create the directory path
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        # save the model
        model = {}
        for params in ['global_params', 'local_params', '_model_setting', '_hyper_params']:
            # Phi;  Theta c_j p_j;  K T V N Interation;  Theta_r_k (p_j_a0*4) Phi_eta eta
            if params in dir(self):
                model[params] = getattr(self, params)

        np.save(model_path + '/' + self._model_name + '.npy', model)
        print('model have been saved by ' + model_path + '/' + self._model_name + '.npy')


    def _calculate_pj(self, c_j: list, T: int):
        '''
        calculate p_j from layer 1 to T+1 according to c_j
        Inputs:
            c_j  : [list] T+1 1*N vector, the variables in the scale parameter in the Theta
            T    : [int] network depth
        Outputs:
            p_j  : [list] T+1 1*N vector, the variables in the scale parameter in the Theta

        '''
        p_j = []
        N = c_j[1].size
        p_j.append((1 - np.exp(-1)) * np.ones([1, N]))  # p_j_1
        p_j.append(1 / (1 + c_j[1]))                    # p_j_2

        for t in [i for i in range(T + 1) if i > 1]:    # p_j_3_T+1; only T>=2 works
            tmp = -np.log(np.maximum(1 - p_j[t - 1], realmin))
            p_j.append(tmp / (tmp + c_j[t]))

        return p_j

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

