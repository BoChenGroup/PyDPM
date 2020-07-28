"""
===========================================
Convolutional Poisson Factor Analysis
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

from pydpm.utils.Metric import *
import numpy as np
import time

class CPFA(object):

    def __init__(self, kernel=100, device='gpu'):

        self.K = kernel

        if device == 'gpu':
            self.device = 'gpu'
            from pydpm.utils import Model_Sampler_GPU
            from pydpm import distribution
            self.Multrnd_Matrix = Model_Sampler_GPU.Multrnd_Matrix_GPU
            self.Crt_Matrix = Model_Sampler_GPU.Crt_Matrix_GPU
            self.Crt_Multirnd_Matrix = Model_Sampler_GPU.Crt_Multirnd_Matrix_GPU
            self.gamma = distribution.gamma
            self.dirichlect = distribution.dirichlet
            self.Conv_Multi_Matrix = Model_Sampler_GPU.conv_multi_sample
        else:
            raise Exception('device type error')

    def initial(self, data, dtype='dense'):

        # data: N*V*L
        if dtype == 'dense':
            self.batch_file_index, self.batch_rows, self.batch_cols = np.where(data)
            self.batch_value = data[self.batch_file_index, self.batch_rows, self.batch_cols]
            N, V, L = data.shape
        elif dtype == 'sparse':
            self.batch_file_index, self.batch_rows, self.batch_cols, self.batch_value = data[0]
            N, V, L = data[1]
        else:
            raise Exception('data type error')

        self.N, self.V, self.L = N,V,L

        self.Setting = {}
        self.Setting['N_train'] = self.N  # 大于等于10的2194个样本
        self.Setting['K1'] = self.K
        self.Setting['K1_V1'] = self.V
        self.Setting['K1_V2'] = self.L + 2  # 这里因为padding　所以加上２　
        self.Setting['K1_S3'] = self.V
        self.Setting['K1_S4'] = 3
        self.Setting['K1_S1'] = self.Setting['K1_V1'] + 1 - self.Setting['K1_S3']
        self.Setting['K1_S2'] = self.Setting['K1_V2'] + 1 - self.Setting['K1_S4']  # 这里因为padding 所以和原来np.max(batch_len)一样
        self.Setting['Iter'] = 200
        self.Setting['Burinin'] = 0.75 * self.Setting['Iter']
        self.Setting['Collection'] = self.Setting['Iter'] - self.Setting['Burinin']

        # SuperParamsSetting
        self.SuperParams = {}
        self.SuperParams['gamma0'] = 0.1  # r
        self.SuperParams['c0'] = 0.1
        self.SuperParams['a0'] = 0.1  # p
        self.SuperParams['b0'] = 0.1
        self.SuperParams['e0'] = 0.1  # c
        self.SuperParams['f0'] = 0.1
        self.SuperParams['eta'] = 0.05  # Phi

        self.Params = {}
        self.Params['D1_k1'] = np.random.rand(self.Setting['K1'], self.Setting['K1_S3'], self.Setting['K1_S4'])
        for k1 in range(self.Setting['K1']):
            self.Params['D1_k1'][k1, :, :] = self.Params['D1_k1'][k1, :, :] / np.sum(self.Params['D1_k1'][k1, :, :])
        self.Params['W1_nk1'] = np.random.rand(self.Setting['N_train'], self.Setting['K1'], self.Setting['K1_S1'], self.Setting['K1_S2'])
        self.Params['c2_n'] = 1 * np.ones([self.Setting['N_train']])
        self.Params['p2_n'] = 1 / (1 + self.Params['c2_n'])

        self.Params['Gamma'] = 0.1 * np.ones([self.Setting['K1'], self.Setting['K1_S1'], self.Setting['K1_S2']])


    def train(self, iter_all=100):

        for t in range(iter_all):

            start_time = time.time()

            # ==========================增广==========================＃
            self.Params['D1_k1_Aug'] = np.zeros_like(self.Params['D1_k1'])  # 增广矩阵用于更新s34维度上增广
            self.Params['W1_nk1_Aug'] = np.zeros_like(self.Params['W1_nk1'])  # 增广矩阵用于更新s12维度上增广

            W1_nk1_Aug, D1_k1_Aug = self.Conv_Multi_Matrix(self.batch_file_index, self.batch_rows, self.batch_cols, self.batch_value, self.Params['W1_nk1'], self.Params['D1_k1'], self.Setting)

            # 第一层增广的结果
            self.Params['W1_nk1_Aug'] = np.array(np.round(W1_nk1_Aug), dtype='float32')  # N*K1*S1*S2
            self.Params['D1_k1_Aug'] = np.array(np.round(D1_k1_Aug), dtype='float32')  # K1*S3*S4

            # ==========================采样==========================＃

            for k1 in range(self.Setting['K1']):
                # update 1th D
                X_k1_34 = self.Params['D1_k1_Aug'][k1, :, :]  # 按三四维增广的矩阵
                D1_k1_s = (X_k1_34 + self.SuperParams['eta']) / np.sum(X_k1_34 + self.SuperParams['eta'])
                self.Params['D1_k1'][k1, :, :] = D1_k1_s

            self.Params['p2_n_aug'] = np.sum(np.sum(np.sum(self.Params['W1_nk1_Aug'], axis=3), axis=2), axis=1)
            self.Params['p2_n'] = np.random.beta(self.SuperParams['a0'] + self.Params['p2_n_aug'],
                                            np.sum(self.Params['Gamma']) + self.SuperParams['b0'])
            self.Params['c2_n'] = (1 - self.Params['p2_n']) / self.Params['p2_n']

            for k1 in range(self.Setting['K1']):
                # Multi
                X_k1_n12 = np.reshape(self.Params['W1_nk1_Aug'][:, k1, :, :],
                                      [self.Setting['N_train'], self.Setting['K1_S1'] * self.Setting['K1_S2']])
                X_k1_12n = np.transpose(X_k1_n12)  # s12*N

                # update 1th W
                W_k1_sn = np.random.gamma(np.transpose(self.Params['Gamma'][k1, :, :]) + X_k1_12n) / (
                            1 + self.Params['c2_n'])  # 假设r_k都相同
                self.Params['W1_nk1'][:, k1, :, :] = np.reshape(np.transpose(W_k1_sn), [self.Setting['N_train'], self.Setting['K1_S1'], self.Setting['K1_S2']])

            end_time = time.time()

            # Likelyhood
            print("epoch " + str(t) + " takes " + str(end_time - start_time) + " seconds")
