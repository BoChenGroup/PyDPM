"""
===========================================
WEDTM Demo
Inter and Intra Topic Structure Learning with Word Embeddings
He Zhao, Lan Du, Wray Buntine, Mingyaun Zhou
Published in International Council for Machinery Lubrication 2018

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>; Wei Zhao <13279389260@163.com>
# License: BSD-3-Clause


import os
import copy
import time
import math
import numpy as np

from ._basic_model import Basic_Model
from .._sampler import Basic_Sampler
from .._utils import *

from scipy import sparse

class WEDTM(Basic_Model):
    def __init__(self, K: [list], device='gpu'):
        """
        The basic model for WEDTM
        Inputs:
            K      : [list] number of topics of each layer;
            device : [str] 'cpu' or 'gpu';

        Attributes:
            @public:
                global_params : [Params] the global parameters of the probabilistic model
                local_params  : [Params] the local parameters of the probabilistic model

            @private:
                _model_setting : [Params] the model settings of the probabilistic model
                _hyper_params  : [Params] the hyper parameters of the probabilistic model

        """
        super(WEDTM, self).__init__()
        setattr(self, '_model_name', 'WEDTM')

        self._model_setting.K = K
        self._model_setting.T = len(K)
        self._model_setting.device = device
        
        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''

        self._sampler = Basic_Sampler(self._model_setting.device)


    def initial(self, data):
        '''
        Inintial the parameters of WEDTM with the input documents
        Inputs:
            data : [np.ndarray] or [scipy.sparse.csc.csc_matrix] V*N matrix, N bag-of-words vectors with a vocabulary length of V

        Attributes:
            @public:
                global_params.Phi  : [np.ndarray] V*K matrix, K topics with a vocabulary length of V
                local_params.Theta : [np.ndarray] N*K matrix, the topic propotions of N documents

            @private:
                _model_setting.V        : [int] scalar, the length of the vocabulary

        '''
        self._model_setting.V = data.shape[0]
        self.global_params.Phi = np.zeros((self._model_setting.K[0], self._model_setting.V)).astype(int)


    def train(self, embeddings: np.ndarray, S: int, iter_all: int, data: np.ndarray, is_train: bool = True):
        '''
        Inputs:
            embeddings     : [np.ndarray] V*D, word embedding of training words
            S              : [int] sub topics
            iter_all       : [np.ndarray] scalar, the iterations of gibbs sampling
            data           : [np.ndarray] V*N_train matrix, N_train bag-of-words vectors with a vocabulary length of V
            is_train       : [bool] True or False, whether to update the global params in the probabilistic model

        Attributes:
            @public:

                local_params.Theta : [np.ndarray] N_train*K matrix, the topic propotions of N_train documents

            @private:
                _model_setting.N         : [int] scalar, the number of the documents in the corpus
                _model_setting.Iteration : [int] scalar, the iterations of sampling

        Outputs:
                local_params  : [Params] the local parameters of the probabilistic model

        '''
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'

        self._model_setting.Iteration = [iter_all] * self._model_setting.T
        self._model_setting.N = data.shape[1]

        # initial local paramters
        self.local_params.Theta = np.zeros((self._model_setting.K[0], self._model_setting.N)).astype(int)

        # WS the trained words' word index
        # DS the trained words' doc index
        # ZS the trained words' random theme
        words_num = np.sum(data)
        WS = np.zeros(words_num).astype(int)
        DS = np.zeros(words_num).astype(int)
        wi, di = np.where(data)
        cc = data[wi, di]
        pos = 0
        for i in range(len(cc)):
            WS[pos:pos+cc[i]] = wi[i]
            DS[pos:pos+cc[i]] = di[i]
            pos = pos+cc[i]

        a0 = 0.01
        b0 = 0.01
        e0 = 1
        f0 = 1
        beta0 = 0.05

        # Add the default word embedding
        embeddings = np.insert(embeddings, embeddings.shape[1], values=np.ones(self._model_setting.V), axis=1)
        self.Theta = [[]] * self._model_setting.T
        c_j = [[]] * (self._model_setting.T + 1)
        for t in range(self._model_setting.T + 1):
            c_j[t] = np.ones((1, self._model_setting.N))

        self.Phi = [{}] * self._model_setting.T
        Xt_to_t1 = [[]] * self._model_setting.T
        WSZS = [[]] * self._model_setting.T
        paraGlobal = [{}] * self._model_setting.T
        # Initialise beta for t = 1
        beta1, self.beta_para = self._init_beta(self._model_setting.K[0], self._model_setting.V, S, embeddings, beta0)

        for Tcurrent in range(self._model_setting.T):
            if Tcurrent == 0:  # layer 1, initial params.
                ZS = np.random.randint(self._model_setting.K[Tcurrent], size=(len(DS)))  # theme of each words
                self.local_params.Theta = np.zeros((self._model_setting.K[Tcurrent], self._model_setting.N)).astype(int)  # Theta (K,N) distribution of theme
                for i in range(len(ZS)):
                    self.local_params.Theta[ZS[i], DS[i]] += 1
                if is_train:
                    self.global_params.Phi = np.zeros((self._model_setting.K[Tcurrent], self._model_setting.V)).astype(int)  # ZSWS Phi (K,V) distribution of words
                    for i in range(len(ZS)):
                        self.global_params.Phi[ZS[i], WS[i]] += 1
                WSZS[Tcurrent] = self.global_params.Phi.T
                Xt_to_t1[Tcurrent] = self.local_params.Theta
                n_dot_k = np.sum(self.local_params.Theta, 1)  # count number of each theme in doc
                p_j = self._calculate_pj(c_j, Tcurrent)
                r_k = 1 / self._model_setting.K[Tcurrent] * np.ones(self._model_setting.K[Tcurrent])
                gamma0 = 1
                c0 = 1
            else:
                self._model_setting.K[Tcurrent] = self._model_setting.K[Tcurrent - 1]
                if self._model_setting.K[Tcurrent] <= 4:
                    break
                self.Phi[Tcurrent] = np.random.rand(self._model_setting.K[Tcurrent - 1], self._model_setting.K[Tcurrent])
                self.Phi[Tcurrent] = self.Phi[Tcurrent] / np.maximum(realmin, np.sum(self.Phi[Tcurrent], 0))
                self.Theta[Tcurrent] = np.ones((self._model_setting.K[Tcurrent], self._model_setting.N)) / self._model_setting.K[Tcurrent]
                p_j = self._calculate_pj(c_j, Tcurrent)
                r_k = 1 / self._model_setting.K[Tcurrent] * np.ones(self._model_setting.K[Tcurrent])
                gamma0 = self._model_setting.K[Tcurrent] / self._model_setting.K[1]
                c0 = 1

            for iter in range(1, self._model_setting.Iteration[Tcurrent]):
                start_time = time.time()

                for t in range(Tcurrent + 1):
                    if t == 0:
                        dex111 = list(range(len(ZS)))
                        np.random.shuffle(dex111)
                        ZS = ZS[dex111]
                        DS = DS[dex111]
                        WS = WS[dex111]
                        if Tcurrent == 0:
                            shape = np.dot(r_k.reshape(-1, 1), np.ones((1, self._model_setting.N)))
                        else:
                            shape = np.dot(self.Phi[1], self.Theta[1])
                        beta1_sum = np.sum(beta1, 1)
                        # Modified from GNBP_mex_collapsed_deep.c in the GBN code,
                        # to support a full matrix of beta1
                        [self.local_params.Theta, temp, n_dot_k, ZS] = self._collapsed_gibbs_topic_assignment_mex(
                            self.local_params.Theta, self.global_params.Phi, n_dot_k, ZS, WS, DS, shape, beta1, beta1_sum)
                        if is_train:
                            self.global_params.Phi = temp
                        WSZS[t] = self.global_params.Phi.T
                        Xt_to_t1[t] = self.local_params.Theta
                        # Sample the variables related to sub-topics
                        beta1 = self.sample_beta(WSZS[t].T, embeddings, beta1)
                    else:
                        [Xt_to_t1[t], WSZS[t]] = self._sampler.multi_aug(Xt_to_t1[t-1], self.Phi[t], self.Theta[t])

                    if t > 0:
                        self.Phi[t] = self._sample_Phi(WSZS[t], beta0)
                        if np.count_nonzero(np.isnan(self.Phi[t])):
                            Warning('Phi Nan')
                            self.Phi[t][np.isnan(self.Phi[t])] = 0
                Xt = self._crt_sum_mex_matrix_v1(sparse.csc_matrix(Xt_to_t1[Tcurrent].T), r_k.reshape(1, -1).T).T
                r_k, gamma0, c0 = self._sample_rk(Xt, r_k, p_j[Tcurrent+1], gamma0, c0)

                if iter > 10:
                    if Tcurrent > 0:
                        p_j[1] = self._sampler.beta(np.sum(Xt_to_t1[0], 0)+a0, np.sum(self.Theta[1], 0)+b0)
                    else:
                        p_j[1] = self._sampler.beta(np.sum(Xt_to_t1[0], 0)+a0, np.sum(r_k)+b0)
                    p_j[1] = np.minimum(np.maximum(p_j[1], np.spacing(1)), 1-np.spacing(1))
                    c_j[1] = (1 - p_j[1]) / p_j[1]
                    for t in range(2, Tcurrent+2):
                        if t == Tcurrent+1:
                            c_j[t] = self._sampler.gamma(np.sum(r_k)*np.ones((1, self._model_setting.N))+e0) / (np.sum(self.Theta[t-1], 0)+f0)
                        else:
                            c_j[t] = self._sampler.gamma(np.sum(self.Theta[t], 0)+e0) / (np.sum(self.Theta[t-1], 0)+f0)
                    p_j_temp = self._calculate_pj(c_j, Tcurrent)
                    p_j[2:] = p_j_temp[2:]

                for t in range(Tcurrent, -1, -1):
                    if t == Tcurrent:
                        shape = r_k.reshape(-1, 1)
                    else:
                        shape = np.dot(self.Phi[t+1], self.Theta[t+1])
                    if t > 0:
                        self.Theta[t] = self._sampler.gamma(shape+Xt_to_t1[t]) * (1/(c_j[t+1] - np.log(np.maximum(1 - p_j[t], realmin))))
                        # (100, 12337/987)  (1, 12337)
                        if np.count_nonzero(np.isnan(self.Theta[t])):
                            Warning('Theta Nan')
                            self.Theta[t][np.isnan(self.Theta[t])] = 0

                end_time = time.time()
                stages = 'Training' if is_train else 'Testing'
                print(f'{stages} Stage: ',
                      f'Layer {Tcurrent:3d}, epoch {iter:3d} takes {end_time - start_time:.2f} seconds, topics {np.count_nonzero(Xt):3d}')

            for t in range(Tcurrent + 1):
                if t == 0:
                    self.Phi[t] = self._sample_Phi(WSZS[t], beta1.T, True)
                else:
                    self.Phi[t] = self._sample_Phi(WSZS[t], beta0, True)

            paraGlobal[Tcurrent]['Phi'] = self.Phi
            paraGlobal[Tcurrent]['r_k'] = r_k
            paraGlobal[Tcurrent]['gamma0'] = gamma0
            paraGlobal[Tcurrent]['c0'] = c0
            paraGlobal[Tcurrent]['K'] = self._model_setting.K[:Tcurrent]
            paraGlobal[Tcurrent]['beta0'] = beta0
            paraGlobal[Tcurrent]['beta_para'] = self.beta_para
            paraGlobal[Tcurrent]['p_j'] = p_j  # for theta
            paraGlobal[Tcurrent]['c_j'] = c_j
            paraGlobal[Tcurrent]['Xt_to_t1'] = Xt_to_t1
            paraGlobal[Tcurrent]['cjmedian'] = []
            for t in range(Tcurrent + 1):
                paraGlobal[Tcurrent]['cjmedian'].append(np.median(c_j[t]))

        return copy.deepcopy(self.local_params)


    def test(self, embeddings: np.ndarray, S: int, iter_all: list, data: np.ndarray):
        '''
        Inputs:
            embeddings     : [np.ndarray] V*D, word embedding of training words
            S              : [int] number of sub topics
            iter_all       : [np.ndarray] scalar, the iterations of gibbs sampling
            data           : [np.ndarray] V*N_train matrix, N_train bag-of-words vectors with a vocabulary length of V

        Outputs:
                local_params  : [Params] the local parameters of the probabilistic model

        '''
        local_params = self.train(embeddings, S, iter_all, data, is_train=False)

        return local_params


    def save(self, model_path: str = './save_models'):
        '''
        Save the model to the specified directory.
        Inputs:
            model_path : [str] the directory path to save the model, default './save_models/WEDTM.npy'
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


    def _init_beta(self, K, V, S, embeddings, beta):
        L = embeddings.shape[1]
        beta_para = [{}] * S
        for s in range(S):
            # variables for sub-topic s
            beta_para[s]['beta_s'] = beta/S * np.ones((K, V))
            beta_para[s]['alpha_k'] = 0.1 * np.ones((K, 1))
            beta_para[s]['W'] = 0.1 * np.ones((K, L))
            beta_para[s]['pi'] = np.dot(beta_para[s]['W'], embeddings.T)
            beta_para[s]['sigma'] = np.ones((K, L))
            beta_para[s]['c0'] = 1
            beta_para[s]['alpha0'] = 1

        beta1 = beta * np.ones((K, V))

        return beta1, beta_para


    def _calculate_pj(self, c_j, T):
        '''
        calculate p_j from layer 1 to T+1
        same as pfa
        '''
        p_j = [[]] * (T+2)
        N = len(c_j[1])
        p_j[0] = (1-np.exp(-1)) * np.ones((1, N))
        p_j[1] = 1/(1 + c_j[1])
        for t in range(2, T+2):
            temp = -np.log(np.maximum(1-p_j[t - 1], realmin))
            p_j[t] = temp / (temp + c_j[t])
            if np.count_nonzero(np.isnan(p_j[t])):
                Warning('pj Nan')
                p_j[t][np.isnan(p_j[t])] = np.spacing(1)
        return p_j


    def _collapsed_gibbs_topic_assignment_mex(self, ZSDS, ZSWS, n_dot_k, ZS, WS, DS, shape, eta, eta_sum):
        '''
        same as DirBN
        '''
        Ksize, Nsize = ZSDS.shape
        WordNum = WS.shape[0]
        prob_cumsum = np.zeros((Ksize, 1))

        for i in range(WordNum):
            v = WS[i]
            j = DS[i]
            k = ZS[i]
            if ZS[i] > -1:
                ZSDS[k, j] -= 1
                ZSWS[k, v] -= 1
                n_dot_k[k] -= 1
            cum_sum = 0
            for k in range(Ksize):
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
                    if (prob_cumsum[k - 1][0] > probrnd) & (prob_cumsum[k][0] > probrnd):
                        kend = k - 1
                    elif (prob_cumsum[k - 1][0] < probrnd) & (prob_cumsum[k][0] < probrnd):
                        kstart = k + 1
                    else:
                        return k
        return k


    def _sample_beta(self, n_topic_word, F, beta1):
        a0 = 0.01
        b0 = 0.01
        e0 = 1
        f0 = 1
        S = len(self.beta_para)
        L = F.shape[1]

        # The word count for each v and k in the first layer
        [K, V] = n_topic_word.shape
        n_sum = np.sum(n_topic_word, 1)

        ## Eq. (3)
        log_inv_q = -np.log(self._sampler.beta(np.sum(beta1, 1), np.maximum(n_sum, realmin)))
        log_log_inv_q = np.log(np.maximum(log_inv_q, realmin))

        # Active topics in the first layer
        active_k = (~np.isnan(log_inv_q)) & (~np.isinf(log_inv_q)) & (n_sum > 0) & (log_inv_q != 0)

        ## Eq. (4) and (6)
        h = np.zeros((K, V, S)).astype(int)
        for k in range(K):
            for v in range(V):
                for j in range(n_topic_word[k, v]):
                    if j == 0:
                        is_add_table = 1
                    else:
                        is_add_table = (np.random.rand() < beta1[k, v] / (beta1[k, v] + j + 1))
                    if is_add_table > 0:
                        p = np.zeros((S, 1))
                        for s in range(S):
                            p[s] = self.beta_para[s]['beta_s'][k, v]
                        sum_cum = np.cumsum(p)
                        temp = np.argwhere(sum_cum > np.random.rand() * sum_cum[-1])
                        if len(temp) > 0:
                            ss = temp[0]
                        else:
                            continue
                        h[k, v, ss] = h[k, v, ss] + 1
        beta1 = 0

        for s in range(S):
            ## For each sub-topic s
            alpha_k = self.beta_para[s]['alpha_k']
            pi_pg = self.beta_para[s]['pi']
            W = self.beta_para[s]['W']
            c0 = self.beta_para[s]['c0']
            alpha0 = self.beta_para[s]['alpha0']
            h_s = h[:, :, s]

            # Sample alpha_k for each sub-topic s with the hierarchical gamma
            h_st = np.zeros((K, V)).astype(int)
            # Eq. (11)
            h_st[h_s > 0] = 1
            for k in range(K):
                for v in range(V):
                    for j in range(h_s[k, v] - 1):
                        h_st[k, v] = h_st[k, v] + (np.random.rand() < alpha_k[k] / (alpha_k[k] + j + 1)).astype(int)

            # Eq. (10)
            h_st_dot = np.sum(h_st, 1)
            # Active topics in each sub-topic s
            local_active_k = h_st_dot > 0 & active_k
            l_a_K = sum(local_active_k)

            x = pi_pg + log_log_inv_q.reshape(-1, 1)
            dex = x < 0
            temp = np.zeros(x.shape)
            temp[dex] = np.log1p(np.exp(x[dex]))
            temp[~dex] = x[~dex]+np.log1p(np.exp(-x[~dex]))
            temp = np.sum(temp, 1)

            # Eq. (9)
            alpha_k = (self._sampler.gamma(alpha0 / l_a_K + h_st_dot) / (c0 + temp)).reshape(-1, 1)
            h_stt = np.zeros((K, 1))
            h_stt[h_st_dot > 0] = 1
            for k in range(K):
                for j in range(h_st_dot[k] - 1):
                    h_stt[k] = h_stt[k] + (np.random.rand() < (alpha0 / l_a_K) / (alpha0 / l_a_K + j + 1)).astype(int)

            temp2 = temp / (c0 + temp)
            # L17 in Figure 1 in the appendix
            alpha0 = self._sampler.gamma(a0 + np.sum(h_stt)) / (b0 - np.sum(np.log(1 - temp2[local_active_k])) / l_a_K)
            c0 = self._sampler.gamma(e0 + alpha0) / (f0 + np.sum(alpha_k[local_active_k]))

            ## Sample Polya-Gamma variables
            # Eq. (15)
            pi_pg_vec = pi_pg + log_log_inv_q.reshape(-1,1)
            pi_pg_vec = pi_pg_vec.reshape(K * V, 1)
            temp = h_s + alpha_k  # reshape(h_s + alpha_k, K*V,1)
            temp = temp.reshape(K * V, 1)
            omega_vec = self._polya_gam_rnd_gam(temp, pi_pg_vec, 2)
            omega_mat = omega_vec
            omega_mat = omega_mat.reshape(K, V)

            ## Sample sigma
            sigma_w = self._sampler.gamma(1e-2 + 0.5 * l_a_K) / (
                        1e-2 + np.sum(np.power(W[local_active_k, :], 2), 0) * 0.5)
            sigma_w = np.tile(sigma_w, (K, 1))

            ## Sample W
            # Eq. (14)
            for k in range(K):
                if local_active_k[k] > 0:
                    Hgam = F.T * omega_mat[k, :]
                    invSigmaW = np.diag(sigma_w[k, :]) + np.dot(Hgam, F)
                    MuW = np.dot(np.linalg.inv(invSigmaW), (
                        np.sum(F.T * (0.5 * h_s[k, :].reshape(1, -1) - 0.5 * alpha_k[k, :] - (log_log_inv_q[k]) * omega_mat[k, :]), 1)))
                    R = self._choll(invSigmaW)
                    W[k, :] = MuW + np.dot(np.linalg.inv(R), np.random.rand(L, 1)).flatten()
                else:
                    W[k, :] = 1e-10

            # Update pi, Eq. (8)
            pi_pg = np.dot(W, F.T)

            ## Sample beta for each sub-topic s
            # Eq. (7)
            beta_s = self._sampler.gamma(alpha_k + h_s) / (np.exp(-pi_pg) + log_inv_q.reshape(-1, 1))
            beta_s[local_active_k == 0, :] = 0.05 / S
            beta_s[(np.sum(np.isnan(beta_s), 1)) != 0, :] = 0.05 / S
            beta_s[(np.sum(np.isnan(beta_s) | np.isinf(beta_s), 1)) != 0, :] = 0.05 / S
            beta_s[(np.sum(beta_s, 1).astype(bool)), :] = 0.05 / S

            ## Update beta1
            beta1 = beta1 + beta_s

            ## Collect results
            # self.beta_para[s]['beta_s'] = beta_s
            self.beta_para[s]['pi'] = pi_pg
            self.beta_para[s]['W'] = W
            self.beta_para[s]['alpha_k'] = alpha_k
            self.beta_para[s]['sigma'] = sigma_w
            self.beta_para[s]['h_s'] = sparse.csc_matrix(h_s)
            self.beta_para[s]['c0'] = c0
            self.beta_para[s]['alpha0'] = alpha0

        return beta1


    def _polya_gam_rnd_gam(self, a, c, KK, IsBiased=None):
        '''
        Generating Polya-Gamma random varaibles using approximation method
        '''
        IsBiased = False
        x = 1 / 2 / math.pi ** 2 * np.sum(self._sampler.gamma(np.dot(a, np.ones((1, KK))), 1) /
                                          (np.power((np.array([i for i in range(KK)]) + 0.5), 2) + np.power(c, 2) / 4 / math.pi ** 2), 1)

        if ~IsBiased:
            temp = abs(c / 2)
            temp[temp <= 0] = realmin
            xmeanfull = (np.tanh(temp) / (temp) / 4)
            xmeantruncate = 1 / 2 / math.pi ** 2 * np.sum(
                1 / (np.power((np.array([i for i in range(KK)]) + 0.5), 2) + np.power(c, 2) / 4 / math.pi ** 2), 1)
            x = x * xmeanfull.flatten() / (xmeantruncate)

        return x

    def _choll(self, A):
        # same as dpfa
        P = A.copy()
        q = np.linalg.cholesky(P)
        q = q.T

        return q

    def _crt_sum_mex_matrix_v1(self, X, r):
        # same as DirBN sample_theta
        k, n = np.shape(X)
        if len(r) == 1:
            r = r[0]
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

    def _sample_rk(self,XTplusOne_sum=None, r_k=None, p_jTplusOne=None, gamma0=None, c0=None, IsNoSample=None, e0=None,
                  f0=None, a0=None, b0=None):
        '''
        get theta_para.r_k, theta_para.gamma0, theta_para.c0
        '''
        if len(p_jTplusOne) == 1:
            p_jTplusOne = p_jTplusOne[0]
        IsNoSample = False
        e0 = 1
        f0 = 1
        a0 = 0.01
        b0 = 0.01
        if (len(XTplusOne_sum.shape) > 1):
            if (XTplusOne_sum.shape[1] > 1):
                # XTplusOne_sum = np.full(np.sum(XTplusOne_sum, 1))
                print('unexcepted error')

        KT = len(r_k)
        if ~IsNoSample:
            c0 = self._sampler.gamma(e0 + gamma0) / (f0 + np.sum(r_k))
            temp = 1 - p_jTplusOne
            temp[temp <= 0] = 2.2251e-308
            sumlogpi = np.sum(np.log(temp))
            p_prime = - sumlogpi / (c0 - sumlogpi)
            # L_k = full(sum(XTplusOne,2));
            # XTplusOne_sum
            gamma0 = self._sampler.gamma(a0 + self.CRT_sum_mex_v1(XTplusOne_sum, (gamma0 / KT))) / (b0 - np.log(max(1 - p_prime, 2.2251e-308)))
            r_k = self._sampler.gamma(gamma0 / KT + XTplusOne_sum) / (c0 - sumlogpi)

        else:
            print('unexcepted error')

        return r_k, gamma0, c0

    def _crt_sum_mex_v1(self, x, r):
        # same to dpfa-CRT
        xx = np.unique(x)
        jj = np.array([np.argwhere(xx == t) for t in x.flatten()]).flatten()
        L = np.zeros(len(x))
        Lsum = 0
        if not x is None:
            for i in range(len(xx)):
                y = int(xx[i])
                if y > 0:
                    L[jj == i] = np.sum(np.random.rand(np.count_nonzero(jj == i), y) <= (r / (r + np.array([t for t in range(y)]))), 1)[0]
            Lsum = int(sum(L))

        return Lsum

    def _sample_Phi(self, WSZS, Eta, IsNoSample=False):
        if ~IsNoSample:
            Phi = self._sampler.gamma(Eta + WSZS)
            temp = np.sum(Phi, 0)
            tempdex = temp > 0
            Phi[:, tempdex] = Phi[:, tempdex] / temp[tempdex]
            Phi[:, ~tempdex] = 0
            if np.count_nonzero(np.isnan(Phi)):
                Warning('Phi Nan')
                tempdex = temp > 0
                Phi[:, ~tempdex] = 0
        else:
            Phi = Eta + WSZS
            temp = np.sum(Phi, 0)
            Phi = Phi / temp
            if np.count_nonzero(np.isnan(Phi)):
                Warning('Phi Nan')
                tempdex = temp > 0
                Phi[:, ~tempdex] = 0
        return Phi

    # collapsed_gibbs_topic_assignment_mex from dirbn.collapsed_gibbs_topic_assignment_mex
    # TrimTcurrent_WEDTM -to-do but not necessary. Prune the inactive factors of the current top hidden layer
    # code before 'iter>10' is similar to sample_theta from DirBN
    # sample_beta func is similar to sample_DirBN_beta, sample_DirBN_counts from DirBN, but been polished
    # some code of sample_beta is similar to dpfa - train 3..
    # PolyaGamRnd_Gam and PolyaGamRndTruncated(dpfa), similar but not same
    # choll is from dpfa



