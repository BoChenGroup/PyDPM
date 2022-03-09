"""
===========================================
Bidirectional Convolutional Poisson Gamma Dynamical Systems Demo todo 两个模型，其中一个为双向的
Wenchao Chen, Chaojie Wang, Bo Cheny, Yicheng Liu, Hao Zhang
Published in Neural Information Processing Systems 2020

===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Wei Zhao <13279389260@163.com>; Jiawen Wu <wjw19960807@163.com>
# License: BSD-3-Clause


import os
import copy
import time
import numpy as np

from ._basic_model import Basic_Model, Params
from .._sampler import Basic_Sampler
from .._utils import *


class CPGDS(Basic_Model):
    def __init__(self, K: int, device='gpu'):
        """
        The basic model for CPGDS
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
        super(CPGDS, self).__init__()
        setattr(self, '_model_name', 'CPGDS')

        self._model_setting.K = K
        self._model_setting.L = 1  # todo
        self._model_setting.device = device

        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''

        self._sampler = Basic_Sampler(self._model_setting.device)

        # self.Crt_Matrix = Model_Sampler_CPU.Crt_Matrix
        # self.Multrnd_Matrix_CPU = Model_Sampler_CPU.Multrnd_Matrix
        # self.Sample_Pi = Model_Sampler_CPU.Sample_Pi
        # self.Multrnd_Matrix = self._sampler.multi_aug


    def initial(self, word2index, doc_split, doc_label, doc_len):
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
        # assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'
        # self._model_setting.V = data.shape[0]
        # self._model_setting.Stationary = 1

        self._model_setting._structure = Params()
        _structure = self._model_setting._structure

        # CPGBN
        _structure.K1_V1 = len(word2index)  # todo 未定义self._model_setting.V
        _structure.K1_S3 = _structure.K1_V1
        _structure.K1_S4 = 3
        _structure.K1_S1 = _structure.K1_V1 - _structure.K1_S3 + 1


        self._model_setting.SweepTime = 2
        self._model_setting.batch_size = 25
        # PGDS
        self._model_setting.Station = 0

        self._model_setting.taoOFR = 0
        self._model_setting.kappaOFR = 0.9
        self._model_setting.kappa0 = 0.7
        self._model_setting.tao0 = 20  # todo 此处几项归为 model setting 还是 hyper params? 有两个 tao0
        self._model_setting.epsi0 = 1

        # CPGBN
        self._hyper_params.a0 = 0.1  # p
        self._hyper_params.b0 = 0.1
        self._hyper_params.eta = 0.1  # Phi

        # PGDS
        self._hyper_params.tao0 = 1

        # initial global params
        # CPGBN
        self.global_parmas.D1_k1 = np.random.rand(self._model_setting.K, _structure.K1_S3, _structure.K1_S4)
        for k1 in range(self._model_setting.K):
            self.global_parmas.D1_k1[k1, :, :] = self.global_parmas.D1_k1[k1, :, :] / np.sum(self.global_parmas.D1_k1[k1, :, :])
        # PGDS
        self.global_parmas.Pi = np.eye(self._model_setting.K)
        self.global_parmas.V = 0.1 * np.ones([self._model_setting.K, 1])

        # in config.py not used
        # Setting.batch_size = 25
        # Setting.SweepTime = 2
        # Setting.K1 = 1000
        # SuperParams
        # self._hyper_params.gamma0 = 0.1  # r
        # self._hyper_params.c0 = 0.1
        # self._hyper_params.e0 = 0.1  # c
        # self._hyper_params.f0 = 0.1
        # self._hyper_params.eta0 = 1
        # self._hyper_params.epilson0 = 0.1


    def train(self, iter_all: int, word2index, doc_split, doc_label, doc_len, is_train: bool = True):
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

        self._model_setting.Iter = iter_all  # 20 default
        self._model_setting.Burnin = 0.6 * self._model_setting.Iter
        self._model_setting.Collection = self._model_setting.Iter - self._model_setting.Burnin

        self._model_setting.N = len(doc_split)
        self._model_setting.batch_num = np.floor(self._model_setting.N / self._model_setting.batch_size).astype('int')
        self._model_setting.iterall = self._model_setting.SweepTime * self._model_setting.batch_num
        self._model_setting.ForgetRate = np.power((self._model_setting.taoOFR + np.linspace(1, self._model_setting.iterall, self._model_setting.iterall)),
                                                  -self._model_setting.kappaOFR)

        epsit = np.power((self._model_setting.tao0 + np.linspace(1, self._model_setting.iterall, self._model_setting.iterall)),
                         -self._model_setting.kappa0)
        epsit = self._model_setting.epsi0 * epsit / epsit[0]

        # initial local parameters
        self.local_params.Theta_kt_all = [0] * self._model_setting.N
        self.local_params.Zeta_t_all = [0] * self._model_setting.N
        self.local_params.Delta_t_all = [0] * self._model_setting.N

        # training phase
        for sweepi in range(self._model_setting.SweepTime):
            for MBt in range(self._model_setting.batch_num):
                start_time = time.time()

                MBObserved = (sweepi * self._model_setting.batch_num + MBt).astype('int')
                doc_batch = doc_split[MBt * self._model_setting.batch_size: (MBt + 1) * self._model_setting.batch_size]

                # ======================= Preprocess =======================#
                Batch_Sparse = Params()
                Batch_Sparse.rows = []
                Batch_Sparse.cols = []
                Batch_Sparse.values = []
                Batch_Sparse.word2sen = []
                Batch_Sparse.sen2doc = []
                Batch_Sparse.sen_len = []
                Batch_Sparse.doc_len = []

                for Doc_index, Doc in enumerate(doc_batch):
                    for Sen_index, Sen in enumerate(Doc):
                        Batch_Sparse.rows.extend(Sen)
                        Batch_Sparse.cols.extend([i for i in range(len(Sen))])
                        Batch_Sparse.values.extend([25 for i in range(len(Sen))])
                        Batch_Sparse.word2sen.extend([len(Batch_Sparse.sen_len) for i in range(len(Sen))])  # the sentence index for word
                        Batch_Sparse.sen2doc.append(Doc_index)  # the document index for sentence
                        Batch_Sparse.sen_len.append(len(Sen))  # the word number for each sentence
                    Batch_Sparse.doc_len.append(len(Doc))  # the sentence number for each doc

                Batch_Sparse.max_doc_len = np.max(np.array(Batch_Sparse.doc_len))  # the max sentence number for each document

                # ======================= Setting CPGBN=======================#
                # initial local parameters in convolutional layers
                _structure = self._model_setting._structure
                _structure.K1_V2 = np.max(np.array(Batch_Sparse.sen_len))  # the max word number for each sentence
                _structure.K1_S1 = _structure.K1_V1 - _structure.K1_S3 + 1
                _structure.K1_S2 = _structure.K1_V2 - _structure.K1_S4 + 1
                _structure.N_Sen = np.max(np.array(Batch_Sparse.word2sen)) + 1  # the number of total sentences

                # ======================= Initial Local Params =======================#
                # CPGBN
                self.local_params.W1_nk1 = np.random.rand(_structure.N_Sen, self._model_setting.K, _structure.K1_S1, _structure.K1_S2)  # N*K*K1_S1*K1_S2
                # PGDS
                self.local_params.Theta_knt = np.ones([self._model_setting.K, self._model_setting.batch_size, Batch_Sparse.max_doc_len])  # K*Batch_size*T
                self.local_params.Zeta_nt = np.ones([self._model_setting.batch_size, Batch_Sparse.max_doc_len + 1])  # Batch_size*(T+1)
                self.local_params.Delta_nt = np.ones([self._model_setting.batch_size, Batch_Sparse.max_doc_len])  # Batch_size*T
                self.local_params.c2_nt = np.ones([self._model_setting.batch_size, Batch_Sparse.max_doc_len])  # Batch_size*T

                #===========================Collecting variables==================#
                EWSZS_D = 0
                EWSZS_Pi = 0

                # ======================= GPU Initial  =======================#
                X_rows = np.array(Batch_Sparse.rows, dtype=np.int32)  # rows
                X_cols = np.array(Batch_Sparse.cols, dtype=np.int32)  # cols
                X_values = np.array(Batch_Sparse.values, dtype=np.int32)
                X_sen_index = np.array(Batch_Sparse.word2sen, dtype=np.int32)  # pages

                word_total = len(X_rows)  # the number of word
                word_aug_stack = np.zeros((self._model_setting.K * _structure.K1_S4 * word_total), dtype=np.float32)
                MultRate_stack = np.zeros((self._model_setting.K * _structure.K1_S4 * word_total), dtype=np.float32)
                Batch_Para = np.array([self._model_setting.K, _structure.K1_S1, _structure.K1_S2, _structure.K1_S3, _structure.K1_S4, word_total], dtype=np.int32)

                block_x = 128  # todo 修改了gpu调用方式
                grid_x = 128
                grid_y = word_total / (block_x * grid_x) + 1

                time_Conv = 0
                time_Aug = 0
                time_Gam = 0

                for iter in range(self._model_setting.Iter):

                    # ====================== Augmentation ======================#
                    self.global_parmas.D1_k1_Aug = np.zeros_like(self.global_parmas.D1_k1)
                    self.global_parmas.W1_nk1_Aug = np.zeros_like(self.global_parmas.W1_nk1)

                    W1_nk1 = np.array(self.global_parmas.W1_nk1, dtype=np.float32, order='C')
                    D1_k1 = np.array(self.global_parmas.D1_k1, dtype=np.float32, order='C')
                    W1_nk1_Aug = np.zeros(W1_nk1.shape, dtype=np.float32, order='C')
                    D1_k1_Aug = np.zeros(D1_k1.shape, dtype=np.float32, order='C')

                    time_1 = time.time()
                    # data augmentation
                    # self._sampler.Crt_Conv_Multi_Sampler()  # todo 匹配这个函数的参数
                    W1_nk1_Aug, D1_k1_Aug = self._sampler.conv_multi_aug(X_rows, X_cols, X_sen_index, X_values, self.global_params.D1_k1, self.local_params.W1_nk1)
                    # fuc = mod.get_function("Multi_Sampler")  # Crt_Conv_Multi_Sampler()
                    # fuc(drv.In(Batch_Para), drv.In(word_aug_stack), drv.In(MultRate_stack), drv.In(X_rows), drv.In(X_cols), drv.In(X_sen_index),
                    #     drv.In(X_values), drv.In(W1_nk1), drv.In(D1_k1), drv.InOut(W1_nk1_Aug), drv.InOut(D1_k1_Aug),
                    #     grid=(int(grid_x), int(grid_y), 1), block=(int(block_x), 1, 1))
                    time_2 = time.time()
                    time_Conv += time_2 - time_1

                    self.global_parmas.W1_nk1_Aug = W1_nk1_Aug  # N*K*K1_S1*K1_S2; Note: Don't add round here, case the scores are too small here!!!
                    self.global_parmas.D1_k1_Aug = D1_k1_Aug  # K*K1_S3*K1_S4
                    self.global_parmas.W1_nk1_Aug_Pool = np.sum(np.sum(self.global_parmas.W1_nk1_Aug, axis=3, keepdims=True), axis=2, keepdims=True)  # N*K
                    self.global_parmas.W1_nk1_Aug_Rate = self.global_parmas.W1_nk1_Aug / (self.global_parmas.W1_nk1_Aug_Pool + realmin)  # N*K*K1_S1*K1_S2

                    # ====================== Augmentation ======================#

                    A_knt = np.zeros([self._model_setting.K, self._model_setting.batch_size, Batch_Sparse.max_doc_len])  # K*Batch_size*T

                    for n in range(self._model_setting.batch_size):

                        A_sen_index = np.array(np.where(np.array(Batch_Sparse.sen2doc) == n))
                        A_kt = np.transpose(self.global_parmas.W1_nk1_Aug_Pool[A_sen_index[0, :], :, 0, 0])  # K*T
                        A_knt[:, n, -Batch_Sparse.doc_len[n]:] = A_kt  # K*Batch_size*T

                    Z_kdotnt = np.zeros([self._model_setting.K, self._model_setting.batch_size, Batch_Sparse.max_doc_len + 1])  # K*Batch_size*(T+1) for Augmented Matrix
                    Z_dotknt = np.zeros([self._model_setting.K, self._model_setting.batch_size, Batch_Sparse.max_doc_len + 1])  # K*Batch_size*(T+1) for Theta
                    Z_kkdot = np.zeros([self._model_setting.K, self._model_setting.K])  # K*K for Pi

                    time_1 = time.time()
                    for t in range(Batch_Sparse.max_doc_len - 1, 0, -1):  # T-1 : 1　Augment T-1 times

                        # the augmented input for the Theta t
                        Q_kn = A_knt[:, :, t] + Z_dotknt[:, :, t + 1]  # K*N
                        # the augmented input from layer t to layer t-1
                        Z_kdotnt[:, :, t] = self._sampler.crt(Q_kn.astype('double'), self._hyper_params.tao0 * np.dot(self.global_parmas.Pi, self.local_params.Theta_knt[:, :, t - 1]))

                        # the augmented input from the Theta t-1
                        # augmented with CPU
                        # [Z_dotknt[:, :, t], Z_kkt] = PGBN_sampler.Multrnd_Matrix(np.array(Z_kdotnt[:, :, t], dtype=np.double, order='C'), self.global_parmas.Pi,
                        #                                                          np.array(self.local_params.Theta_knt[:, :, t-1], dtype=np.double, order='C'))
                        # augmented with GPU
                        [Z_dotknt[:, :, t], Z_kkt] = self._sampler.crt_multi_aug(np.array(Z_kdotnt[:, :, t], dtype=np.double, order='C'), self.global_parmas.Pi,
                                                                                 np.array(self.local_params.Theta_knt[:, :, t - 1], dtype=np.double, order='C'))
                        Z_kkdot = Z_kkdot + Z_kkt

                    time_2 = time.time()
                    time_Aug += time_2 - time_1

                    # Calculate Zeta
                    if self._model_setting.Station == 0:
                        for t in range(Batch_Sparse.max_doc_len - 1, -1, -1):
                            self.local_params.Zeta_nt[:, t] = np.log(1 + self.local_params.Zeta_nt[:, t + 1] + self.local_params.Delta_nt[:, t] / self._hyper_params.tao0)

                    time_1 = time.time()
                    # Update Theta
                    for t in range(Batch_Sparse.max_doc_len):

                        if t == 0:
                            shape_kn = self._hyper_params.tao0 * self.global_parmas.V
                        else:
                            shape_kn = self._hyper_params.tao0 * np.dot(self.global_parmas.Pi, self.local_params.Theta_knt[:, :, t - 1])

                        self.local_params.c2_nt[:, t] = np.random.gamma(np.sum(shape_kn, axis=0) + self._hyper_params.a0) / \
                                                        (np.sum(self.local_params.Theta_knt[:, :, t], axis=0) + self._hyper_params.b0)

                        shape_theta = A_knt[:, :, t] + Z_dotknt[:, :, t + 1] + shape_kn
                        scale_theta = self.local_params.Delta_nt[:, t:t + 1] + self.local_params.c2_nt[:, t:t + 1] + self._hyper_params.tao0 * self.local_params.Zeta_nt[:, t + 1:t + 2]
                        self.local_params.Theta_knt[:, :, t] = np.random.gamma(shape_theta) / np.transpose(scale_theta)

                    time_2 = time.time()
                    time_Gam += time_2 - time_1

                    # Update W
                    for n in range(self._model_setting.batch_size):
                        Theta_kn = self.local_params[:, n, -Batch_Sparse.doc_len[n]:]
                        A_sen_index = np.array(np.where(np.array(Batch_Sparse.sen2doc) == n))
                        self.global_parmas.W1_nk1[A_sen_index[0, :], :, :, :] = self.global_parmas.W1_nk1_Aug_Rate[A_sen_index[0, :], :, :, :] * \
                                                                                np.reshape(np.transpose(Theta_kn), [Batch_Sparse.doc_len[n], self._model_setting.K, 1, 1])

                    if iter > (self._model_setting.Burnin - 1):
                        EWSZS_D = EWSZS_D + self.global_parmas.D1_k1_Aug
                        EWSZS_Pi = EWSZS_Pi + Z_kkdot

                print('Sampler_Conv: {:<8.4f}'.format(time_Conv))
                print('Sampler_Aug: {:<8.4f}'.format(time_Aug))
                print('Sampler_Theta: {:<8.4f}'.format(time_Gam))

                # ================update global parameters===================#
                Phi = np.transpose(np.reshape(self.global_parmas.D1_k1, [self._model_setting.K, _structure.K1_S3 * _structure.K1_S4]))  # todo Phi ?
                EWSZS_D = np.transpose(np.reshape(EWSZS_D, [self._model_setting.K, _structure.K1_S3 * _structure.K1_S4]))
                EWSZS_D = self._model_setting.batch_num * EWSZS_D / self._model_setting.Collection
                EWSZS_Pi = self._model_setting.batch_num * EWSZS_Pi / self._model_setting.Collection

                if (MBObserved == 0):
                    NDot_D = EWSZS_D.sum(0)
                    NDot_Pi = EWSZS_Pi.sum(0)
                else:
                    NDot_D = (1 - self._hyper_params.ForgetRate[MBObserved]) * NDot_D + self._hyper_params.ForgetRate[MBObserved] * EWSZS_D.sum(0)
                    NDot_Pi = (1 - self._hyper_params.ForgetRate[MBObserved]) * NDot_Pi + self._hyper_params.ForgetRate[MBObserved] * EWSZS_Pi.sum(0)

                # Update D
                tmp = EWSZS_D + self._hyper_params.eta
                tmp = (1 / (NDot_D + realmin)) * (tmp - tmp.sum(0) * Phi)
                tmp1 = (2 / (NDot_D + realmin)) * Phi
                tmp = Phi + epsit[MBObserved] * tmp + np.sqrt(epsit[MBObserved] * tmp1) * np.random.randn(Phi.shape[0], Phi.shape[1])
                Phi = self.ProjSimplexSpecial(tmp, Phi, 0)
                self.global_parmas.D1_k1 = np.reshape(np.transpose(Phi), [self._model_setting.K, _structure.K1_S3, _structure.K1_S4])

                # Update Pi
                Pi_prior = np.eye(self._model_setting.K)
                # Pi_prior = np.dot(self.global_parmas.V, np.transpose(self.global_parmas.V))
                # Pi_prior[np.arange(K), np.arange(K)] = 0
                # Pi_prior = Pi_prior + np.diag(np.reshape(Params.Xi * self.global_parmas.V, [K, 1]))  # todo 未找到 Xi

                tmp = EWSZS_Pi + Pi_prior
                tmp = (1 / (NDot_Pi + realmin)) * (tmp - tmp.sum(0) * self.global_parmas.Pi)
                tmp1 = (2 / (NDot_Pi + realmin)) * self.global_parmas.Pi
                tmp = self.global_parmas.Pi + epsit[MBObserved] * tmp + np.sqrt(epsit[MBObserved] * tmp1) * np.random.randn(
                    self.global_parmas.Pi.shape[0], self.global_parmas.Pi.shape[1])
                self.global_parmas.Pi = self.ProjSimplexSpecial(tmp, self.global_parmas.Pi, 0)

                end_time = time.time()
                print("{} takes {} seconds".format(MBt, end_time - start_time))

                # if np.mod(MBObserved, 100) == 0:  # todo 保存模型
                #     Params_save = {}
                #     Params_save['D'] = self.global_parmas.D1_k1
                #     Params_save['Pi'] = self.global_parmas.Pi
                #
                #     cPickle.dump(Params_save, open(model_path, 'wb'))

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


    def ProjSimplexSpecial(self, Phi_tmp, Phi_old, epsilon):
        Phinew = Phi_tmp - (Phi_tmp.sum(0) - 1) * Phi_old
        if np.where(Phinew[:, :] <= 0)[0].size > 0:
            Phinew = np.maximum(epsilon, Phinew)
            Phinew = Phinew / np.maximum(realmin, Phinew.sum(0))
        return Phinew


