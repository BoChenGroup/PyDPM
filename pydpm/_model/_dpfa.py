"""
===========================================
Deep Poisson Factor Analysis Demo
Scalable Deep Poisson Factor Analysis for Topic Modeling
Zhe Gan, Changyou Chen, Ricardo Henao, David Carlson, Lawrence Carin
Publised in International Conference on Machine Learning 2015
https://github.com/zhegan27/dpfa_icml2015 news_dpfa_sbn_gibbs.m  pfa_dsbn_gibbs.m
===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Wei Zhao <13279389260@163.com>
# License: BSD-3-Clause

import os
import copy
import time
import math
import numpy as np

from ._basic_model import Basic_Model
from .._sampler import Basic_Sampler
from .._utils import *


class DPFA(Basic_Model):
    def __init__(self, K: list, device='cpu'):
        """
        The basic model for DPFA
        Inputs:
            K      : [list] numbers of topics of 3 layers in DPFA(PFA+DSBN+Gibbs);
            device : [str] 'cpu' or 'gpu';

        Attributes:
            @public:
                global_params : [Params] the global parameters of the probabilistic model
                local_params  : [Params] the local parameters of the probabilistic model

            @private:
                _model_setting : [Params] the model settings of the probabilistic model
                _hyper_params  : [Params] the hyper parameters of the probabilistic model

        """
        super(DPFA, self).__init__()
        setattr(self, '_model_name', 'DPFA')
        
        self._model_setting.K1, self._model_setting.K2, self._model_setting.K3 = K
        self._model_setting.device = device

        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''

        self._sampler = Basic_Sampler(self._model_setting.device)


    def initial(self, data: np.ndarray):
        '''
        Inintial the parameters of DPFA with the input documents
        Inputs:
            data : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary length of V

        Attributes:
            @public:
                global_params.Phi  : [np.ndarray] V*K matrix, K topics with a vocabulary length of V
                local_params.Theta : [np.ndarray] N*K matrix, the topic propotions of N documents

            @private:
                _model_setting.V        : [int] scalar, the length of the vocabulary
        '''
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'
        self._model_setting.V = data.shape[0]

        self.global_params.Phi = np.random.rand(self._model_setting.V, self._model_setting.K1)
        self.global_params.Phi = self.global_params.Phi / np.sum(self.global_params.Phi, 0)


    def train(self, burnin: int, collection: int, data: np.ndarray, is_train: bool = True):
        '''
        Inputs:
            burnin     : [int] the iterations of burnin stage
            collection : [int] the iterations of collection stage
            data       : [np.ndarray] V*N_train matrix, N_train bag-of-words vectors with a vocabulary length of V
            is_train   : [bool] True or False, whether to update the global params in the probabilistic model

        Attributes:
            @public:
                local_params.Theta  : [np.ndarray] N_train*K matrix, the topic propotions of N train documents

            @private:
                _model_setting.N          : [int] scalar, the number of the documents in the corpus
                _model_setting.burnin     : [int] scalar, the iterations of burnin stage
                _model_setting.collection : [int] scalar, the iterations of collection stage

        Outputs:
                local_params  : [Params] the local parameters of the probabilistic model
        '''
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'

        self._model_setting.N = data.shape[1]
        self._model_setting.burnin = burnin
        self._model_setting.collection = collection

        # initial local parameters
        self.local_params.Theta = 1 / self._model_setting.K1 * np.ones((self._model_setting.K1, self._model_setting.N))

        H1train = np.ones((self._model_setting.K1, self._model_setting.N))

        W1 = 0.1 * np.random.randn(self._model_setting.K1, self._model_setting.K2)
        W2 = 0.1 * np.random.randn(self._model_setting.K2, self._model_setting.K3)
        c1 = 0.1 * np.random.randn(self._model_setting.K1, 1)
        c2 = 0.1 * np.random.randn(self._model_setting.K2, 1)
        Pi = 1 / self._model_setting.K3 * np.ones((self._model_setting.K3, 1))
        H3train = (np.tile(Pi, (1, self._model_setting.N)) > np.random.rand(self._model_setting.K3, self._model_setting.N)).astype(int)
        T = np.dot(W2, H3train)
        prob = 1 / (1 + np.exp(-T))
        H2train = (prob >= np.random.rand(self._model_setting.K2, self._model_setting.N)).astype(int)

        c0 = 1
        gamma0 = 1
        eta = 0.05
        r_k = 50 / self._model_setting.K1 * np.ones((self._model_setting.K1, 1))
        p_i_train = 0.5 * np.ones((1, self._model_setting.N))
        e0 = 1e-2
        f0 = 1e-2

        self.mid = dict()
        self.mid['K'] = []
        self.mid['PhiThetaTrain'] = 0
        self.mid['Count'] = 0

        self.result = dict()
        self.result['loglikeTrain'] = []
        self.result['K'] = []
        self.result['PhiThetaTrain'] = 0
        self.result['Count'] = 0
        self.result['Phi'] = np.zeros((self._model_setting.V, self._model_setting.K1))
        self.result['r_k'] = np.zeros((self._model_setting.K1, 1))
        self.result['W1'] = np.zeros((self._model_setting.K1, self._model_setting.K2))
        self.result['W2'] = np.zeros((self._model_setting.K2, self._model_setting.K3))
        self.result['c1'] = np.zeros((self._model_setting.K1, 1))
        self.result['c2'] = np.zeros((self._model_setting.K2, 1))
        self.result['Pi'] = np.zeros((self._model_setting.K3, 1))
        self.result['ThetaTrain'] = np.zeros((self._model_setting.K1, self._model_setting.N))
        self.result['H1train'] = np.zeros((self._model_setting.K1, self._model_setting.N))
        self.result['H2train'] = np.zeros((self._model_setting.K2, self._model_setting.N))
        self.result['H3train'] = np.zeros((self._model_setting.K3, self._model_setting.N))
        self.result['x_kntrain'] = np.zeros((self._model_setting.K1, self._model_setting.N))

        for iter in range(self._model_setting.burnin + self._model_setting.collection):
            start_time = time.time()

            # 1. Sample x_pnk
            x_kntrain, x_pk = self._sampler.multi_aug(data, self.global_params.Phi, self.local_params.Theta)

            # c1 c2 Pi
            c1 = c1.reshape(-1, 1)
            c2 = c2.reshape(-1, 1)
            Pi = Pi.reshape(-1, 1)

            if (iter < np.ceil(self._model_setting.burnin / 40).astype(int)):
                self.local_params.Theta = self._sampler.gamma(x_kntrain + np.dot(r_k, np.ones((1, self._model_setting.N))) * H1train,
                                             np.dot(np.ones((self._model_setting.K1, 1)), p_i_train))
                if is_train:
                    self.global_params.Phi = self._sampler.gamma(eta + x_pk, 1)
                    self.global_params.Phi = self.global_params.Phi / np.sum(self.global_params.Phi, 0)

            else:
                # 2. Sample H1
                lix = (x_kntrain == 0)
                cix, rix = np.where(x_kntrain.T == 0)
                T = np.dot(W1, H2train) + c1
                prob = 1 / (1 + np.exp(-T))
                p1 = prob[lix] * ((1 - p_i_train[0, cix].T) ** r_k[rix].flatten())
                p0 = 1 - prob[lix]
                H1train = np.ones((self._model_setting.K1, self._model_setting.N))
                H1train[lix] = ((p1 / (p1 + p0)) > np.random.rand(len(rix)))

                # 3. inference of sbn
                # (1). update gamma0
                Xmat = np.dot(W1, H2train) + c1
                Xvec = Xmat.reshape(self._model_setting.K1 * self._model_setting.N, 1)
                gamma0vec = self._polya_gam_rnd_truncated(np.ones((self._model_setting.K1 * self._model_setting.N, 1)), Xvec, 20)
                gamma0Train = gamma0vec.reshape(self._model_setting.K1, self._model_setting.N)

                # (2). update W1
                for j in range(self._model_setting.K1):
                    Hgam = H2train * gamma0Train[j, :]
                    invSigmaW = np.eye(self._model_setting.K2) + np.dot(Hgam, H2train.T)
                    MuW = np.dot(np.linalg.inv(invSigmaW), (np.sum(H2train * (H1train[j, :] - 0.5 - c1[j] * gamma0Train[j, :]), 1)))
                    R = self._choll(invSigmaW)
                    W1[j, :] = MuW + np.dot(np.linalg.inv(R), np.random.rand(self._model_setting.K2, 1)).flatten()

                # (3). update H2
                res = np.dot(W1, H2train)
                for k in range(self._model_setting.K2):
                    res = res - np.dot(W1[:, k].reshape(-1, 1), H2train[k, :].reshape(1, -1))
                    mat1 = res + c1
                    vec1 = sum((H1train - 0.5 - gamma0Train * mat1) * W1[:, k].reshape(-1, 1))
                    vec2 = sum(gamma0Train * np.power(W1[:, k].reshape(-1, 1), 2)) / 2
                    logz = vec1 - vec2 + np.dot(W2[k, :], H3train) + c2[k]
                    probz = 1 / (1 + np.exp(-logz))
                    H2train[k, :] = (probz > np.random.rand(1, self._model_setting.N))
                    res = res + np.dot(W1[:, k].reshape(-1, 1), H2train[k, :].reshape(1, -1))

                # (4). update c1
                sigmaC = 1 / (np.sum(gamma0Train, 1) + 1)
                muC = sigmaC * np.sum(H1train - 0.5 - gamma0Train * np.dot(W1, H2train), 1)
                c1 = np.random.normal(muC, np.sqrt(sigmaC))

                # (5). update gamma1
                Xmat = np.dot(W2, H3train) + c2
                Xvec = Xmat.reshape(self._model_setting.K2 * self._model_setting.N, 1)
                gamma1vec = self._polya_gam_rnd_truncated(np.ones((self._model_setting.K2 * self._model_setting.N, 1)), Xvec, 20)
                gamma1Train = gamma1vec.reshape(self._model_setting.K2, self._model_setting.N)

                # (6). update W2
                for k in range(self._model_setting.K2):
                    Hgam = H3train * gamma1Train[k, :]
                    invSigmaW = np.eye(self._model_setting.K3) + np.dot(Hgam, H3train.T)
                    MuW = np.dot(np.linalg.inv(invSigmaW), np.sum(H3train * (H2train[k, :] - 0.5 - c2[k] * gamma1Train[k, :]), 1))
                    R = self._choll(invSigmaW)
                    W2[k, :] = MuW + np.dot(np.linalg.inv(R), np.random.randn(self._model_setting.K3, 1)).flatten()

                # (7). update H3
                res = np.dot(W2, H3train)
                for k in range(self._model_setting.K3):
                    res = res - np.dot(W2[:, k].reshape(-1, 1), H3train[k, :].reshape(1, -1))
                    mat1 = res + c2
                    vec1 = sum((H2train - 0.5 - gamma1Train * mat1) * W2[:, k].reshape(-1, 1))
                    vec2 = sum(gamma1Train * np.power(W2[:, k].reshape(-1, 1), 2)) / 2
                    logz = vec1 - vec2
                    probz = np.exp(logz) * Pi[k] / (np.exp(logz) * Pi[k] + 1 - Pi[k])
                    H3train[k, :] = (probz > np.random.rand(1, self._model_setting.N))
                    res = res + np.dot(W2[:, k].reshape(-1, 1), H3train[k, :].reshape(1, -1))

                Znz = np.sum(H3train, 1)
                Pi = np.random.beta(c0 / self._model_setting.K3 + Znz, c0 - c0 / self._model_setting.K3 + self._model_setting.N - Znz)

                # (8). update c2
                sigmaC = 1 / (np.sum(gamma1Train, 1) + 1)
                muC = sigmaC * np.sum(H2train - 0.5 - gamma1Train * np.dot(W2, H3train), 1)
                c2 = np.random.normal(muC, np.sqrt(sigmaC))

                # 4. Sample r_k
                _, kk = np.where(x_kntrain.T)
                temp = x_kntrain.T.flatten()
                counts = temp[temp != 0]
                ll = np.zeros(len(counts))
                L_k = np.zeros((self._model_setting.K1, 1))
                for k in range(self._model_setting.K1):
                    L_k[k], ll[kk == k] = self._crt(counts[kk == k], r_k[k])
                sumbpi = np.sum(H1train * np.log(1-p_i_train+realmin*((1-p_i_train) < realmin)), 1)
                p_prime_k = -sumbpi / (c0 - sumbpi)
                gamma0 = self._sampler.gamma(e0 + self._crt(L_k.flatten(), np.array([gamma0]))[0], 1 / (f0 - sum(np.log(1 - p_prime_k + realmin * ((1 - p_prime_k) < realmin)))))
                r_k = self._sampler.gamma(gamma0 + L_k.flatten(), 1 / (-sumbpi + c0))

                # 5. Sample Theta
                self.local_params.Theta = self._sampler.gamma(x_kntrain + np.dot(r_k.reshape(-1, 1), np.ones((1, self._model_setting.N))) * H1train,
                                             np.dot(np.ones((self._model_setting.K1, 1)), p_i_train))

                # 6. Sample Phi
                if is_train:
                    self.global_params.Phi = self._sampler.gamma(eta + x_pk, 1)
                    self.global_params.Phi = self.global_params.Phi / np.sum(self.global_params.Phi, 0)

            # Now, collect self.results
            if iter < self._model_setting.burnin:
                X1train = np.dot(self.global_params.Phi, self.local_params.Theta)
                self.mid['PhiThetaTrain'] = self.mid['PhiThetaTrain'] + X1train
                self.mid['Count'] = self.mid['Count'] + 1
                temp = np.count_nonzero(np.sum(x_kntrain, 1))
                self.mid['K'].append(temp)

            else:
                X1train = np.dot(self.global_params.Phi, self.local_params.Theta)
                self.result['PhiThetaTrain'] = self.result['PhiThetaTrain'] + X1train
                self.result['Count'] = self.result['Count'] + 1
                temp = np.count_nonzero(np.sum(x_kntrain, 1))
                self.result['K'].append(temp)

                if is_train:
                    self.result['Phi'] = self.result['Phi'] + self.global_params.Phi / self._model_setting.collection

                self.result['r_k'] = self.result['r_k'] + r_k / self._model_setting.collection
                self.result['W1'] = self.result['W1'] + W1 / self._model_setting.collection
                self.result['W2'] = self.result['W2'] + W2 / self._model_setting.collection
                self.result['c1'] = self.result['c1'] + c1 / self._model_setting.collection
                self.result['c2'] = self.result['c2'] + c2 / self._model_setting.collection
                self.result['Pi'] = self.result['Pi'] + Pi / self._model_setting.collection

                self.result['ThetaTrain'] = self.result['ThetaTrain'] + self.local_params.Theta / self._model_setting.collection
                self.result['H1train'] = self.result['H1train'] + H1train / self._model_setting.collection
                self.result['H2train'] = self.result['H2train'] + H2train / self._model_setting.collection
                self.result['H3train'] = self.result['H3train'] + H3train / self._model_setting.collection
                self.result['x_kntrain'] = self.result['x_kntrain'] + x_kntrain / self._model_setting.collection

            end_time = time.time()
            stages = 'Training' if is_train else 'Testing'
            print(f'{stages} Stage: ',
                  f'epoch {iter:3d} takes {end_time - start_time:.2f} seconds')

        return copy.deepcopy(self.local_params)


    def test(self, burnin: int, collection: int, data: np.ndarray):
        '''
        Inputs:
            burnin     : [int] the iterations of burnin stage
            collection : [int] the iterations of collection stage
            data       : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary of length V

        Outputs:
            local_params  : [Params] the local parameters of the probabilistic model

        '''
        local_params = self.train(burnin, collection, data, is_train=False)

        return local_params


    def save(self, model_path: str = './save_models'):
        '''
        Save the model to the specified directory.
        Inputs:
            model_path : [str] the directory path to save the model, default './save_models/DPFA.npy'
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


    def _polya_gam_rnd_truncated(self, a, c, KK, IsBiased=None):
        '''
        Generating Polya-Gamma random varaibles using approximation method
        '''
        IsBiased = False
        x = 1 / 2 / math.pi**2 * np.sum(self._sampler.gamma(np.dot(a, np.ones((1, KK))), 1) /
                                         (np.power((np.array([i for i in range(KK)]) + 0.5), 2) + np.power(c, 2) / 4 / math.pi**2), 1)
        if ~IsBiased:
            temp = abs(c / 2)
            temp[temp<=0] = realmin
            xmeanfull = (np.tanh(temp) / (temp) / 4)
            xmeantruncate = 1 / 2 / math.pi**2 * np.sum(1 / (np.power((np.array([i for i in range(KK)]) + 0.5), 2) + np.power(c, 2)/4/math.pi**2), 1)
            x = x * xmeanfull.flatten() / (xmeantruncate)

        return x


    def _choll(self, A):
        P = A.copy()
        q = np.linalg.cholesky(P)
        q = q.T

        return q


    def _crt(self, x, r):
        xx = np.unique(x)
        jj = np.array([np.argwhere(xx == t) for t in x.flatten()]).flatten()
        L = np.zeros(len(x))
        Lsum = 0
        if not x is None:
            for i in range(len(xx)):
                y = int(xx[i])
                if y > 0:
                    L[jj == i] = np.sum(np.random.rand(np.count_nonzero(jj == i), y) <= (r / (r + [t for t in range(y)])), 1)
            Lsum = int(sum(L))

        return Lsum, L





