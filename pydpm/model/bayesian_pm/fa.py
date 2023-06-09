"""
===========================================
Factor Analysis
===========================================
"""

# Author: Xinyang Liu <lxy771258012@163.com>;
# License: BSD-3-Claus

import os
import copy
import time
import numpy as np

from ..basic_model import Basic_Model, Params
from ...sampler import Basic_Sampler
from ...utils import *



class FA(Basic_Model):
    def __init__(self, M: int, device='gpu'):
        """
        The basic model for FA
        Inputs:
            M      : [int] dimension of factor;
            device : [str] 'cpu' or 'gpu';

        Attributes:
            @public:
                global_params : [Params] the global parameters of the probabilistic model
                local_params  : [Params] the local parameters of the probabilistic model

            @private:
                _model_setting : [Params] the model settings of the probabilistic model

        """
        super(FA, self).__init__()
        setattr(self, '_model_name', 'FA')

        self._model_setting.M = M
        self._model_setting.device = device

        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''

        self._sampler = Basic_Sampler(self._model_setting.device)


    def initial(self, data: np.ndarray):
        '''
        Inintial the parameters of LDA with the input documents
        Inputs:
            data : [np.ndarray] V*N matrix, N samples, V features per sample

        Attributes:
            @public:
                global_params.Phi  : [np.ndarray] V*K matrix, K topics with a vocabulary length of V

            @private:
                self._model_setting.D : [int] scalar, the dimension of the features
                self._model_setting.N : [int] scalar, number of samples
                self._hyper_params.a : [float] scalar, hyper-parameters
                self._hyper_params.b : [float] scalar, hyper-parameters
                self._hyper_params.c : [float] scalar, hyper-parameters
                self._hyper_params.d : [float] scalar, hyper-parameters
                self._hyper_params.e : [float] scalar, hyper-parameters
                self._hyper_params.f : [float] scalar, hyper-parameters
        '''
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'
        [self._model_setting.D, self._model_setting.N] = data.shape
        self._hyper_params.a, self._hyper_params.b, self._hyper_params.c, \
        self._hyper_params.d, self._hyper_params.e, self._hyper_params.f, = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    def sample_w(self, x, output, N, M, D):
        for i in range(M):
            t = copy.deepcopy(output.w)
            f = copy.deepcopy(output.z)
            t = np.delete(t, i, axis=1)
            f = np.delete(f, i, axis=0)
            tmp = np.sum(np.matmul(t, f), axis=1)
            arr = x - np.matmul(t, f)
            cs = np.sum(np.tile(output.z[i, :], (D, 1)) * arr, axis=1).T
            ds = cs * output.cr
            s = np.sum(np.power(output.z[i, :], 2))
            q = 1.0 / (output.cr * s + output.cw[:, i].T)
            u = ds * q
            t1 = np.power(q, 0.5).T
            t2 = self._sampler.standard_normal((D, 1))
            t3 = t1 * t2
            output.w[0: D, i] = (u.T + t3)[:, 0]

        return output

    def sample_z(self, x, output, N, M, D):
        for i in range(M):
            q = 1.0 / (np.matmul(np.matmul(output.w[:, i].T, output.Ar), output.w[:, i]) + output.cz[:, i])
            t = copy.deepcopy(output.w)
            f = copy.deepcopy(output.z)
            t = np.delete(t, i, axis=1)
            f = np.delete(f, i, axis=0)
            arr = x - np.matmul(t, f)
            s = np.sum(arr, axis=1)
            u = np.matmul(np.matmul(q * arr.T, output.Ar), output.w[:, i])
            output.z[i, 0:N] = (u.T + np.power(q, 0.5) * self._sampler.standard_normal((1, N)))[0, :]

        return output
    def sample_cr(self, x, output, hyper_p, N, M, D):
        aa = 0.5 * N + hyper_p.e
        bb = 0.5 * np.sum(np.power(x - np.matmul(output.w, output.z), 2), axis=1).T + hyper_p.f
        shape = aa * np.ones([1, D])
        scale = 1.0 / bb * np.ones([1, D])
        output.cr = self._sampler.gamma(shape, scale)
        output.Ar = np.diag(output.cr[0])

        return output

    def sample_cw(self, x, output, hyper_p, N, M, D):
        aa = 0.5 + hyper_p.a
        bb = 0.5 * np.power(output.w, 2) + hyper_p.b
        shape = aa * np.ones([D, M])
        scale = 1.0 / bb * np.ones([D, M])
        output.cw = self._sampler.gamma(shape, scale)

        return output

    def sample_cz(self, x, output, hyper_p, N, M, D):
        aa = 0.5 * N + hyper_p.c
        bb = 0.5 * (np.sum(np.power(output.z, 2), axis=1)).T + hyper_p.d
        shape = aa * np.ones([1, M])
        scale = 1.0 / bb * np.ones([1, M])
        output.cz = self._sampler.gamma(shape, scale)

        return output

    def hasattrs(self, obj):
        '''
            Check weather the local parameters need to be initialized
            Return:
                True: Initialize
                False: Not initialize
        '''
        attr_strs = ['w', 'z', 'cw', 'cz', 'cr', 'Ar']
        for attr_str in attr_strs:
            if not hasattr(obj, attr_str):
                return False
        return True

    def train(self, data: np.ndarray, num_epochs: int, is_train: bool = True, is_initial_local: bool=True):
        '''
        Inputs:
            data       : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary of length V
            is_train   : [bool] True or False, whether to update the global params in the probabilistic model
            num_epochs : [int] scalar, the epochs of gibbs sampling
            is_initial_local : [bool] weather initialize the local parameters

        Attributes:
            @public:
                local_params.w  : [np.ndarray] D*M matrix
                local_params.z  : [np.ndarray] M*N matrix
                local_params.cw : [np.ndarray] D*M matrix
                local_params.cz : [np.ndarray] 1*M vector
                local_params.cr : [np.ndarray] 1*D vector
                local_params.Ar : [np.ndarray] D*D diagonal matrix

            @private:
                _model_setting.Iteration : [int] scalar, the iterations of gibbs sampling

        Outputs:
                local_params  : [Params] the local parameters of the probabilistic model

        '''
        assert type(data) is np.ndarray, 'Data type error: the input data should be a 2-D np.ndarray'

        # [self._model_setting.D, self._model_setting.N] = data.shape
        self._model_setting.Iteration = num_epochs

        # initial local parameters
        # X = data - np.tile(np.mean(data, axis=1, keepdims=True), self._model_setting.N)
        if is_initial_local or not hasattr(self.local_params):
            self.local_params.w = np.ones([self._model_setting.D, self._model_setting.M])
            self.local_params.z = np.ones([self._model_setting.M, self._model_setting.N])


            cw_shape = self._hyper_params.a * np.ones([self._model_setting.D, self._model_setting.M])
            cw_scale = self._hyper_params.b * np.ones([self._model_setting.D, self._model_setting.M])
            self.local_params.cw = self._sampler.gamma(cw_shape, cw_scale)


            cz_shape = self._hyper_params.c * np.ones([1, self._model_setting.M])
            cz_scale = self._hyper_params.d * np.ones([1, self._model_setting.M])
            self.local_params.cz = self._sampler.gamma(cz_shape, cz_scale)


            cr_shape = self._hyper_params.e * np.ones([1, self._model_setting.D])
            cr_scale = self._hyper_params.f * np.ones([1, self._model_setting.D])
            self.local_params.cr = self._sampler.gamma(cr_shape, cr_scale)

            self.local_params.Ar = np.diag(self.local_params.cr[0])

        fw = np.zeros([self._model_setting.D, self._model_setting.M])
        fz = np.zeros([self._model_setting.M, self._model_setting.N])
        fcw = np.zeros([self._model_setting.D, self._model_setting.M])
        fcz = np.zeros([1, self._model_setting.M])
        fcr = np.zeros([1, self._model_setting.D])
        fAr = np.diag(fcr[0])
        count = 0

        # gibbs sampling
        iter_updata = int(0.7 * self._model_setting.Iteration)
        for iter in range(self._model_setting.Iteration):
            if (iter > iter_updata) and (np.mod(iter, 5) == 1):
                fw = fw + self.local_params.w
                fz = fz + self.local_params.z
                fcw = fcw + self.local_params.cw
                fcz = fcz + self.local_params.cz
                fcr = fcr + self.local_params.cr
                fAr = fAr + self.local_params.Ar
                count = count + 1
            self.local_params = self.sample_w(data, self.local_params, self._model_setting.N, self._model_setting.M,
                                           self._model_setting.D)
            self.local_params = self.sample_z(data, self.local_params, self._model_setting.N, self._model_setting.M,
                                           self._model_setting.D)
            self.local_params = self.sample_cz(data, self.local_params, self._hyper_params, self._model_setting.N, self._model_setting.M,
                                           self._model_setting.D)
            self.local_params = self.sample_cr(data, self.local_params, self._hyper_params, self._model_setting.N, self._model_setting.M,
                                           self._model_setting.D)
            self.local_params = self.sample_cw(data, self.local_params, self._hyper_params, self._model_setting.N, self._model_setting.M,
                                           self._model_setting.D)

            fw_t = fw / (count + 1)
            fz_t = fz / (count + 1)
            res = data - np.matmul(fw_t, fz_t)
            resq = np.mean(np.sum(np.power(res, 2)))
            resx = np.mean(np.sum(np.power(data, 2)))
            error = resq / resx
            print("Epoch {}|{} error = {:.6f}".format(iter, self._model_setting.Iteration, error))

        # fw = fw / count
        # fz = fz / count
        # fcz = fcz / count
        # fcr = fcr / count
        # fAr = fAr / count
        # fcw = fcw / count

        return copy.deepcopy(self.local_params)


    def test(self, data: np.ndarray, num_epochs: int, is_train = False, is_initial_local: bool=True):
        '''
        Inputs:
            data       : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary of length V
            is_train   : [bool] True or False, whether to update the global params in the probabilistic model
            num_epochs : [int] scalar, the epochs of gibbs sampling
            is_initial_local : [bool] weather initialize the local parameters

        Outputs:
            local_params  : [Params] the local parameters of the probabilistic model

        '''
        local_params = self.train(data, num_epochs, is_train=False, is_initial_local=is_initial_local)

        return local_params


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



