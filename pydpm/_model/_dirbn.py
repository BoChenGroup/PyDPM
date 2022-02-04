"""
===========================================
Poisson Factor Analysis DirBN(Dirichlet belief networks) Demo
Dirichlet belief networks for topic structure learning
He Zhao, Lan Du, Wray Buntine, Mingyuan Zhou
Publihsed in Conference and Workshop on Neural Information Processing Systems 2018

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

from scipy import sparse


class DirBN(Basic_Model):
    def __init__(self, K: list, device='gpu'):
        """
        The basic model for DirBN
        Inputs:
            K      : [list] number of topics of 2 layers in DirBN;
            device : [str] 'cpu' or 'gpu';

        Attributes:
            @public:
                global_params : [Params] the global parameters of the probabilistic model
                local_params  : [Params] the local parameters of the probabilistic model

            @private:
                _model_setting : [Params] the model settings of the probabilistic model
                _hyper_params  : [Params] the hyper parameters of the probabilistic model

        """
        super(DirBN, self).__init__()
        setattr(self, '_model_name', 'DirBN')

        self._model_setting.K = K
        self._model_setting.device = device

        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''

        self._sampler = Basic_Sampler(self._model_setting.device)


    def initial(self, data: np.ndarray):
        '''
        Inintial the parameters of DirBN with the input documents
        Inputs:
            data : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary length of V

        Attributes:
            @public:
                global_params.Phi  : [np.ndarray] V*K matrix, K topics with a vocabulary length of V
                local_params.Theta : [np.ndarray] N*K matrix, the topic propotions of N documents

            @private:
                _model_setting.V        : [int] scalar, the length of the vocabulary
                _hyper_params.Phi_eta   : [float] scalar, the parameter in the prior of Phi

        '''
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'
        self._model_setting.V = data.shape[0]

        self.global_params.Phi = np.zeros((self._model_setting.K[0], self._model_setting.V)).astype(int)  # frequency, not distribution

        self._hyper_params.Phi_eta = 0.05

    def train(self, iter_all: int, data: np.ndarray, is_train: bool = True):
        '''
        Inputs:
            iter_all   : [int] scalar, the iterations of sampling
            data       : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary of length V
            is_train   : [bool] True or False, whether to update the global params in the probabilistic model

        Attributes:
            @public:
                local_params.Theta : [np.ndarray] N_train*K matrix, the topic propotions of N_train documents

            @private:
                _model_setting.N         : [int] scalar, the number of the documents in the corpus
                _model_setting.Iteration : [int] scalar, the iterations of gibbs sampling
                _hyper_params.DirBN_para : [dict] scalar, the hyper params of DirBN
                _hyper_params.theta_para : [dict] scalar, the hyper params theta of model

        Outputs:
                local_params  : [Params] the local parameters of the probabilistic model

        '''
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'
        self._model_setting.N = data.shape[1]
        self._model_setting.Iteration = iter_all

        # initial local parameters
        self.local_params.Theta = np.zeros((self._model_setting.K[0], self._model_setting.N)).astype(int)

        # initial _hyper_params
        self._hyper_params.DirBN_para = self._init_DirBN(self._model_setting.K, self._model_setting.V, self._hyper_params.Phi_eta)
        self._hyper_params.theta_para = self._init_theta(self._model_setting.K[0], self._model_setting.N)

        self.data = dict()
        self.data['train_ws'] = []
        self.data['train_ds'] = []
        for n in range(self._model_setting.N):
            for v in range(self._model_setting.V):
                self.data['train_ws'].extend([v] * int(data[v, n]))
                self.data['train_ds'].extend([n] * int(data[v, n]))
        self.data['train_ws'] = np.array(self.data['train_ws'])
        self.data['train_ds'] = np.array(self.data['train_ds'])

        self.zs = np.random.randint(self._model_setting.K[0], size=int(np.sum(data)))  # theme of words
        # self.zs_ds, self.local_params.Theta, theme frequency of docs
        if is_train:
            self.global_params.Phi = np.zeros((self._model_setting.K[0], self._model_setting.V)).astype(int)  # phi: distribution of words
        for i in range(len(self.zs)):
            self.local_params.Theta[self.zs[i], self.data['train_ds'][i]] += 1
            if is_train:  # when train
                self.global_params.Phi[self.zs[i], self.data['train_ws'][i]] += 1  # word frequency, not distribution

        self.n_dot_k = np.sum(self.local_params.Theta, 1)  # K, frequency of themes

        for iter in range(self._model_setting.Iteration):
            start_time = time.time()

            # sample topic assignments by the collapsed Gibbs sampling
            self.local_params.Theta, temp, self.n_dot_k, self.zs = self._collapsed_gibbs_topic_assignment_mex(self.local_params.Theta, copy.deepcopy(self.global_params.Phi), self.n_dot_k, self.zs, self.data['train_ws'],
                                                                             self.data['train_ds'],
                                                                             np.tile(self._hyper_params.theta_para['r_k'], (1, self._model_setting.N)),
                                                                             self._hyper_params.DirBN_para[0]['psi'],
                                                                             np.sum(self._hyper_params.DirBN_para[0]['psi'], 1))
            if is_train:
                self.global_params.Phi = temp

            self._sample_DirBN(self.global_params.Phi)  # get self._hyper_params.DirBN_para
            self._sample_theta(self.local_params.Theta)  # get self._hyper_params.theta_para

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
            model_path : [str] the directory path to save the model, default './save_models/DirBN.npy'
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
            model_path : [str] the directory path to load the model;

        '''
        assert os.path.exists(model_path), 'Path Error: can not find the path to load the model'
        model = np.load(model_path, allow_pickle=True).item()

        for params in ['global_params', 'local_params', '_model_setting', '_hyper_params']:
            if params in model:
                setattr(self, params, model[params])

    def _init_DirBN(self, ks, V, eta):
        '''
        initial DirBN params
        '''
        T = len(ks)
        DirBN_para = []
        for t in range(T):
            DirBN_para.append({'None': None})
        '''
        [0]
                 psi: [100×13370 double]
        n_topic_word: [100×13370 double]
                 phi: [100×13370 double]
        [1]
                  psi: [100×13370 double]
                  phi: [100×13370 double]
                 beta: [100×100 double]
          beta_gammak: [100×1 double]
               beta_c: 1.0298
          beta_gamma0: 16.8023
              beta_c0: 0.3564
         n_topic_word: [100×13370 double]
        n_topic_topic: [100×100 double]'''
        for t in range(T):
            DirBN_para[t]['psi'] = eta * np.ones((ks[t], V))
            if t > 0:
                if t < T - 1:
                    DirBN_para[t]['phi'] = self._sampler.gamma(DirBN_para[t]['psi'])
                else:
                    DirBN_para[t]['phi'] = self._sampler.gamma(DirBN_para[t]['psi'] * np.ones((ks[t], V)))
                DirBN_para[t]['phi'] = DirBN_para[t]['phi'] / DirBN_para[t]['phi'].sum(axis=1, keepdims=1)
                DirBN_para[t]['beta'] = 0.5 * np.ones((ks[t], ks[t - 1]))
                DirBN_para[t]['beta_gammak'] = 0.1 * np.ones((ks[t], 1))
                DirBN_para[t]['beta_c'] = 0.1
                DirBN_para[t]['beta_gamma0'] = 1.0
                DirBN_para[t]['beta_c0'] = 1.0
        return DirBN_para

    def _init_theta(self, K = None, N = None):
        theta_para = {'gamma0': 1, 'c0': 1, 'r_k': 1 / K*np.ones((K, 1)), 'p_j': (1 - np.exp(- 1)) * np.ones((1, N))}
        return theta_para

    def _collapsed_gibbs_topic_assignment_mex(self, ZSDS, ZSWS, n_dot_k, ZS, WS, DS, shape, eta, eta_sum):
        Ksize, Nsize = ZSDS.shape
        WordNum = WS.shape[0]
        prob_cumsum = np.zeros(Ksize)

        # collapsed_gibbs_topic_assignment
        for i in range(WordNum):
            v = WS[i]  # index of word
            j = DS[i]  # index of doc
            k = ZS[i]  # index of theme
            if ZS[i] > -1:
                ZSDS[k, j] -= 1
                ZSWS[k, v] -= 1
                n_dot_k[k] -= 1
            cum_sum = 0
            for k in range(Ksize):  # for each themes
                cum_sum += (eta[k, v] + ZSWS[k, v]) / (eta_sum[k] + n_dot_k[k]) * (ZSDS[k, j] + shape[k, j])
                prob_cumsum[k] = cum_sum
            probrnd = np.random.rand() * cum_sum
            k = self._binary_search(probrnd, prob_cumsum, Ksize)
            ZS[i] = k
            ZSDS[k, j] += 1
            ZSWS[k, v] += 1
            n_dot_k[k] += 1

        return ZSDS, ZSWS, n_dot_k, ZS

    def _binary_search(self, probrnd, prob_cumsum, Ksize):
        if probrnd <= prob_cumsum[0]:
            return 0
        else:
            kstart = 1
            kend = Ksize - 1
            while 1:
                if kstart >= kend:
                    return kend
                else:
                    k = kstart + int((kend - kstart) / 2)
                    if (prob_cumsum[k-1] > probrnd) & (prob_cumsum[k] > probrnd):
                        kend = k - 1
                    elif (prob_cumsum[k-1] < probrnd) & (prob_cumsum[k] < probrnd):
                        kstart = k + 1
                    else:
                        return k

        return k

    def _sample_DirBN(self, n_topic_word1 = None):
        T = len(self._hyper_params.DirBN_para)
        T_current = T
        self._hyper_params.DirBN_para[0]['n_topic_word'] = n_topic_word1
        if T > 1:
            # propagate the latent counts from the bottom up
            for t in range(T_current - 1):
                self._sample_DirBN_counts(t)
            # update the latent variables from the top down
            for t in range(T_current-1, -1, -1):
                # update psi
                if t < T-1:
                    self._hyper_params.DirBN_para[t]['psi'] = np.dot(self._hyper_params.DirBN_para[t+1]['beta'].T, self._hyper_params.DirBN_para[t+1]['phi'])
                else:
                    psi = self._sample_DirBN_eta(self._hyper_params.DirBN_para[T-1]['psi'][0, 0], self._hyper_params.DirBN_para[T-1]['n_topic_word'])
                    self._hyper_params.DirBN_para[T-1]['psi'] = np.ones_like(self._hyper_params.DirBN_para[T-1]['n_topic_word']) * psi
                # update beta
                if t > 0:
                    self._sample_DirBN_beta(t)
                # update phi
                phi = self._sampler.gamma(np.abs(self._hyper_params.DirBN_para[t]['psi'] + self._hyper_params.DirBN_para[t]['n_topic_word'])) + 2.2204e-16
                phi = phi / phi.sum(axis=1, keepdims=1)
                self._hyper_params.DirBN_para[t]['phi'] = phi
        else:
            psi = self._sample_DirBN_eta(self._hyper_params.DirBN_para[T-1]['psi'][0, 0], self._hyper_params.DirBN_para[T-1]['n_topic_word'])
            self._hyper_params.DirBN_para[T-1]['psi'] = np.ones_like(n_topic_word1) * psi
            phi = self._sampler.gamma(self._hyper_params.DirBN_para[T-1]['psi'] + self._hyper_params.DirBN_para[T-1]['n_topic_word']) + 2.2204e-16
            phi = phi / (np.sum(phi, 1).reshape(-1, 1))
            self._hyper_params.DirBN_para[T-1]['phi'] = phi

    def _sample_DirBN_eta(self, eta=None, n=None):
        mu_0 = 0.1
        nu_0 = 10.0
        K, V = n.shape
        # if (np.sum(V*eta<0)+np.sum(np.sum(n, 1)<0))>0:
        #     raise('L341, negative value')
        #     log_q = -np.log(np.random.beta(np.maximum(V*eta, realmin), np.maximum(np.sum(n, 1), realmin)))
        # else:
        #     log_q = -np.log(np.random.beta(V*eta, np.sum(n, 1)))
        log_q = -np.log(self._sampler.beta(np.maximum(V * eta, realmin), np.maximum(np.sum(n, 1), realmin)))
        t = np.zeros((K, V))
        t[n > 0] = 1
        for k in range(K):
            for v in range(V):
                for j in range(1, int(n[k, v])):
                    t[k, v] = t[k, v] + (np.random.rand() < eta / (eta + j))

        eta = self._sampler.gamma(mu_0 + np.sum(t)) / (nu_0 + V * np.sum(log_q))

        return eta

    def _sample_DirBN_beta(self, t=None):
        a0 = 0.01
        b0 = 0.01
        e0 = 1
        f0 = 1
        beta_gamma0 = self._hyper_params.DirBN_para[t]['beta_gamma0']
        beta_gammak = self._hyper_params.DirBN_para[t]['beta_gammak']
        beta_c0 = self._hyper_params.DirBN_para[t]['beta_c0']
        beta_c = self._hyper_params.DirBN_para[t]['beta_c']

        if (np.sum(np.sum(self._hyper_params.DirBN_para[t - 1]['psi'], 1) < 0) + np.sum(np.sum(self._hyper_params.DirBN_para[t-1]['n_topic_word'], 1) < 0)) > 0:
            Warning('negative value')
        w_log_inv_q = -np.log(self._sampler.beta(np.maximum(np.sum(self._hyper_params.DirBN_para[t - 1]['psi'], 1), realmin),
                                              np.maximum(np.sum(self._hyper_params.DirBN_para[t-1]['n_topic_word'], 1), realmin)))
        w_t_k2_k1 = self._hyper_params.DirBN_para[t]['n_topic_topic']
        K2, K1 = w_t_k2_k1.shape
        w_tt = np.zeros([K2, K1])
        w_tt[w_t_k2_k1 > 0] = 1

        for k2 in range(K2):
            for k1 in range(K1):
                for j in range(1, int(w_t_k2_k1[k2, k1])):
                    w_tt[k2, k1] = w_tt[k2, k1] + (np.random.rand() < beta_gammak[k2] / (beta_gammak[k2] + j))

        w_tt_k2_dot = np.sum(w_tt, 1)
        active_k1 = (~(np.isnan(w_log_inv_q))) & (~(np.isinf(w_log_inv_q))) & (w_log_inv_q != 0)
        a_K1 = np.sum(active_k1)
        temp = np.log(1 + w_log_inv_q / beta_c)
        temp = np.sum(temp[active_k1])
        beta_gammak = self._sampler.gamma(beta_gamma0 / K2 + w_tt_k2_dot) / (beta_c0 + temp)
        w_tt_k2_dot_t = np.zeros(K2)
        w_tt_k2_dot_t[w_tt_k2_dot > 0] = 1

        for k2 in range(K2):
            for j in range(1, int(w_tt_k2_dot[k2])):
                w_tt_k2_dot_t[k2] = w_tt_k2_dot_t[k2] + (np.random.rand() < (beta_gamma0 / K2) / (beta_gamma0 / K2 + j))

        beta_gamma0 = self._sampler.gamma(a0 + np.sum(w_tt_k2_dot_t)) / (b0 + np.log(1 + temp / beta_c0))
        beta_c0 = self._sampler.gamma(e0 + beta_gamma0) / (f0 + np.sum(beta_gammak))
        beta_c = self._sampler.gamma(1.0 + a_K1 * np.sum(beta_gammak)) / (1.0 + np.sum(self._hyper_params.DirBN_para[t]['beta']))
        DirBN_beta = self._sampler.gamma(w_t_k2_k1 + beta_gammak.reshape(-1, 1)) / (beta_c + np.tile(w_log_inv_q.reshape(-1 ,1), (1, K2)).T)

        self._hyper_params.DirBN_para[t]['beta_gammak'] = beta_gammak
        self._hyper_params.DirBN_para[t]['beta'][:, active_k1] = DirBN_beta[:, active_k1]
        self._hyper_params.DirBN_para[t]['beta_c'] = beta_c
        self._hyper_params.DirBN_para[t]['beta_gamma0'] = beta_gamma0
        self._hyper_params.DirBN_para[t]['beta_c0'] = beta_c0

    def _sample_DirBN_counts(self, t=None):
        n_topic_word = self._hyper_params.DirBN_para[t]['n_topic_word']
        phi = self._hyper_params.DirBN_para[t + 1]['phi']
        DirBN_beta = self._hyper_params.DirBN_para[t + 1]['beta']
        DirBN_psi = self._hyper_params.DirBN_para[t]['psi']
        K1, V = n_topic_word.shape
        K2 = phi.shape[0]
        w_t_k2_k1 = np.zeros((K2, K1))
        w_t_k2_v = np.zeros((K2, V))

        for k1 in range(K1):
            for v in range(V):
                for j in range(int(n_topic_word[k1, v])):
                    if j == 0:
                        if_t = 1
                    else:
                        if_t = np.random.rand() < DirBN_psi[k1, v] / (DirBN_psi[k1, v] + j+1)
                    if if_t > 0:
                        p = phi[:, v] * DirBN_beta[:, k1]
                        sum_cum = np.cumsum(p)
                        k2 = np.argwhere(sum_cum > (np.random.rand()*sum_cum[-1])).flatten()[0]
                        w_t_k2_k1[k2, k1] = w_t_k2_k1[k2, k1] + 1
                        w_t_k2_v[k2, v] = w_t_k2_v[k2, v] + 1
        self._hyper_params.DirBN_para[t+1]['n_topic_word'] = w_t_k2_v
        self._hyper_params.DirBN_para[t+1]['n_topic_topic'] = w_t_k2_k1

    def _sample_Theta(self, theta_count=None):
        '''theta_count:  [numpy.ndarray]'''
        b0 = 0.01
        a0 = 0.01
        t = self._crt_sum_mex_matrix_v1(theta_count.T, self._hyper_params.theta_para['r_k'].flatten())

        self._sample_rk(t, self._hyper_params.theta_para['r_k'], self._hyper_params.theta_para['p_j'], self._hyper_params.theta_para['gamma0'], self._hyper_params.theta_para['c0'])
        self._hyper_params.theta_para['p_j'] = self._sampler.beta(np.sum(theta_count, 0) + a0, np.sum(self._hyper_params.theta_para['r_k'], 0) + b0).reshape(1, -1)
        self._hyper_params.theta_para['theta'] = self._sampler.gamma(self._hyper_params.theta_para['r_k'] + theta_count) * self._hyper_params.theta_para['p_j']

    def _sample_rk(self, XTplusOne_sum=None, r_k=None, p_jTplusOne=None, gamma0=None, c0=None, IsNoSample=None, e0=None,
                   f0=None, a0=None, b0=None):
        p_jTplusOne = p_jTplusOne.flatten()
        r_k = r_k.flatten()
        IsNoSample = False
        e0 = 1
        f0 = 1
        a0 = 0.01
        b0 = 0.01
        if (len(XTplusOne_sum.shape) > 1):
            if (XTplusOne_sum.shape[1] > 1):
                # XTplusOne_sum = np.full(np.sum(XTplusOne_sum, 1))
                raise('unexcepted error')

        KT = len(r_k)
        if not IsNoSample:
            c0 = self._sampler.gamma(e0 + gamma0) / (f0 + np.sum(r_k))
            sumlogpi = np.sum(np.log(np.maximum(1-p_jTplusOne, realmin)))
            p_prime = - sumlogpi / (c0 - sumlogpi)
            # L_k = full(sum(XTplusOne,2));
            # XTplusOne_sum
            gamma0 = self._sampler.gamma(a0 + self._crt_sum_mex_v1(XTplusOne_sum, (gamma0 / KT))) / (b0 - np.log(np.maximum(1 - p_prime, realmin)))
            r_k = self._sampler.gamma(gamma0 / KT + XTplusOne_sum) / (c0 - sumlogpi)
        else:
            raise('unexcepted error')
            # c0 = (e0 + gamma0) / (f0 + np.sum(r_k))
            # sumlogpi = np.sum(np.log(np.max(1 - p_jTplusOne, realmin)))
            # p_prime = - sumlogpi / (c0 - sumlogpi)
            # # L_k = full(sum(XTplusOne,2));
            # # XTplusOne_sum;
            # #!!!!!!!!!!!!
            # temp = np.dot(gamma0 / KT, np.sum(np.psi(XTplusOne_sum + gamma0 / KT) - np.psi(gamma0 / KT)))
            # # Sample_rk.m:35
            # gamma0 = (a0 + temp) / (b0 - np.log(np.max(1 - p_prime, realmin)))
            # self._hyper_params.Theta_r_k = (gamma0 / KT + XTplusOne_sum) / (c0 - sumlogpi)
        self._hyper_params.theta_para['r_k'] = r_k.reshape(-1, 1)
        self._hyper_params.theta_para['gamma0'] = gamma0
        self._hyper_params.theta_para['c0'] = c0

    def _crt_sum_mex_matrix_v1(self, X, r):
        '''
        X: sparse(theta_count.T) N*K，
        r: self._hyper_params.theta_para['r_k'].T 1*K 0.1
        '''
        k, n = np.shape(X)
        lsum = np.zeros(n).astype(int)
        maxx = 0
        Xsparse = sparse.csc_matrix(X)
        pr = Xsparse.data
        ir = Xsparse.indices
        jc = Xsparse.indptr

        for j in range(n):
            starting_row_index = jc[j]
            stopping_row_index = jc[j+1]
            if starting_row_index == stopping_row_index:
                continue
            else:
                for current_row_index in range(starting_row_index, stopping_row_index):
                    maxx = int(max(maxx, pr[current_row_index]))
                prob = np.zeros(maxx)
                for i in range(maxx):
                    prob[i] = r[j] / (r[j] + i)

                lsum[j] = 0
                for current_row_index in range(starting_row_index, stopping_row_index):
                    for i in range(int(pr[current_row_index])):
                        if np.random.rand() <= prob[i]:
                            lsum[j] += 1

        return lsum

    def _crt_sum_mex_v1(self, x, r):
        if len(x.shape) == 2:
            Lenx = x.shape[0] * x.shape[1]
        else:
            Lenx = x.shape[0]
        maxx = 0
        for i in range(Lenx):
            if maxx < x[i]:
                maxx = x[i]
        prob = np.zeros(maxx)
        for i in range(maxx):
            prob[i] = r/(r+i)
        ij = 0
        Lsum = 0
        for i in range(Lenx):
            for j in range(x[i]):
                if np.random.rand() < prob[j]:
                    Lsum += 1
        return Lsum



