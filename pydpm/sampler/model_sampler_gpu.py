"""
===========================================
Model Sampler implemented on GPU
===========================================

"""

import os
import numpy as np
import ctypes
import subprocess

from .pre_process import para_preprocess


# def find_in_path(name, path):
#     "Find a file in a search path"
#     # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
#     for dir in path.split(os.pathsep):
#         binpath = os.path.join(dir, name)
#         if os.path.exists(binpath):
#             return os.path.abspath(binpath)
#     return None
#
#
# def get_nvcc_path():
#     # get nvcc path
#     if 'CUDAHOME' in os.environ:
#         home = os.environ['CUDAHOME']
#         nvcc = os.path.join(home, 'bin', 'nvcc')
#     else:
#         # otherwise, search the PATH for NVCC
#         default_path = os.path.join(os.sep, 'usr', 'local', 'cuda', 'bin')
#         nvcc = find_in_path('nvcc', os.environ['PATH'] + os.pathsep + default_path)
#         # if nvcc is None:
#         #     raise EnvironmentError('The nvcc binary could not be '
#         #                            'located in your $PATH. Either add it to your path, or set $CUDAHOME')
#     return nvcc


class model_sampler_gpu(object):

    def __init__(self, system_type='Windows', seed=0):
        """
        The basic class for sampling distribution on cpu
        """
        super(model_sampler_gpu, self).__init__()

        self.system_type = system_type
        self.seed = seed

        # ------------------------------------------------ basic sampler ------------------------------------------
        if system_type == 'Windows':
            '''
            To compile CUDA C/C++ under Windows system, Visual Studio and CUDA should have been installed.
            This module has been tested under Visual Studio 2019(with MSVC v142 - VS 2019 C++ x64/x86 tools) and CUDA Toolkit 11.5.
            '''
            compact_path = os.path.dirname(__file__) + "\_compact\model_sampler.dll"
            if not os.path.exists(compact_path):

                # subprocess.call == os.system
                install_flag = subprocess.call('nvcc -o ' +'"'+ compact_path +'"'+  ' --shared ' +'"'+ compact_path[:-4] + '_win.cu' +'"', shell=True) # ""保证路径中的空格
                if install_flag != 0: # not install success

                    search_flag = os.system('where nvcc')
                    if search_flag != 0: # not search success
                        Warning('Could not locate the path of nvcc, please make sure nvcc can be located by the system command "where nvcc"')
                    else: # search success
                        path_results = subprocess.check_output('where nvcc', shell=True)
                        path_results = str(path_results, encoding='utf-8')
                        nvcc_path = path_results.split('\n')[0]
                        nvcc_path = nvcc_path[: nvcc_path.find('nvcc.exe')] + 'nvcc.exe'
                        install_flag = subprocess.call('"'+ nvcc_path +'"'+ ' -o ' +'"'+ compact_path +'"'+ ' --shared ' +'"'+ compact_path[:-4] + '_win.cu' +'"', shell=True)

                if install_flag == 0:
                    print("The model sampler has been installed successfully!")

            dll = ctypes.cdll.LoadLibrary(compact_path)

        elif system_type == 'Linux':

            compact_path = os.path.dirname(__file__) + "/_compact/model_sampler.so"
            # if True:
            if not os.path.exists(compact_path):
                install_flag = subprocess.call('nvcc -Xcompiler -fPIC -shared -o ' +'"'+ compact_path +'"'+ ' ' +'"'+ compact_path[:-3] + '_linux.cu' +'"', shell=True)
                if install_flag != 0:  # not install success
                    search_flag = os.system('which nvcc')

                    if search_flag != 0:  # not search success
                        Warning('Could not locate the path of nvcc, please make sure nvcc can be located by the system command "which nvcc"')
                    else:  # search success
                        path_results = subprocess.check_output('which nvcc', shell=True)
                        path_results = str(path_results, encoding='utf8')
                        nvcc_path = path_results.split('\n')[0]
                        nvcc_path = nvcc_path[: nvcc_path.find('nvcc')] + 'nvcc'
                        install_flag = subprocess.call('"'+ nvcc_path +'"'+ ' -Xcompiler -fPIC -shared -o ' +'"'+ compact_path +'"'+ ' ' +'"'+ compact_path[:-3] + '_linux.cu' +'"', shell=True)

                if install_flag == 0:
                    print("The model sampler has been installed successfully!")

            dll = ctypes.cdll.LoadLibrary(compact_path)

        # ------------------------------------------------substorage ------------------------------------------
        self._init_status = dll._init_status
        self._init_status.argtypes = [ctypes.c_size_t]
        self._init_status.restype = ctypes.c_void_p
        self.rand_status = self._init_status(self.seed)

        # ----------------------------------------------cuda sampler ------------------------------------------
        self._multi_aug = dll._multi_aug
        self._multi_aug.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_void_p]

        self._crt_multi_aug = dll._crt_multi_aug
        self._crt_multi_aug.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_void_p]

        self._conv_multi_aug = dll._conv_multi_aug
        self._conv_multi_aug.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_void_p]


    def multi_aug(self, X_t, Phi_t, Theta_t, dtype='dense'):
        """
        sampler for the gamma distribution
        Inputs:
            X_t     : [np.ndarray];
            Phi_t   : [np.ndarray];
            Theta_t : [np.ndarray];
            dtype   : 'dense' or 'sparse';
        Outputs:
            output : [np.ndarray] or [pycuda.gpuarray] the resulting matrix on the device 'cpu' or 'gpu'
        """
        assert dtype in ['dense', 'sparse'], print(f'the dtype of X_t should be dense or sparse')

        if dtype == 'dense':

            X_t = np.array(X_t, dtype=np.float32, order='C')
            Phi_t = np.array(Phi_t, dtype=np.float32, order='C')
            Theta_t = np.array(Theta_t, dtype=np.float32, order='C')

            if len(X_t.shape) == 0:
                X_t = X_t[:, np.newaxis, np.newaxis]
            elif len(X_t.shape) == 1:
                X_t = X_t[:, np.newaxis]

            if len(Theta_t.shape) == 0:
                Theta_t = Theta_t[:, np.newaxis, np.newaxis]
            elif len(Theta_t.shape) == 1:
                Theta_t = Theta_t[:, np.newaxis]

            [X_t_rows, X_t_cols] = np.where(X_t > 0.5)
            X_t_values = X_t[(X_t_rows, X_t_cols)]
            [V, K] = Phi_t.shape
            J = Theta_t.shape[1]
            Num_Elems = len(X_t_values)
            Params = np.array([V, K, J, Num_Elems], dtype=np.int32, order='C')
        else:
            [X_t_rows, X_t_cols, X_t_values] = X_t[0], X_t[1], X_t[2]
            [V, K] = Phi_t.shape
            J = Theta_t.shape[1]
            Num_Elems = len(X_t_values)
            Params = np.array([V, K, J, Num_Elems], dtype=np.int32, order='C')

        # input variables
        X_t_values = np.array(X_t_values, dtype=np.float32, order='C')
        X_t_rows = np.array(X_t_rows, dtype=np.float32, order='C')
        X_t_cols = np.array(X_t_cols, dtype=np.float32, order='C')
        Phi_t = np.array(Phi_t, dtype=np.float32, order='C')
        Theta_t = np.array(Theta_t, dtype=np.float32, order='C')

        # output variables
        XVK = np.zeros([V, K], dtype=np.float32, order='C')
        XKJ = np.zeros([K, J], dtype=np.float32, order='C')

        if Num_Elems > 0:
            # pointer
            Params_p = ctypes.cast(Params.ctypes.data, ctypes.POINTER(ctypes.c_int))
            X_t_values_p = ctypes.cast(X_t_values.ctypes.data, ctypes.POINTER(ctypes.c_float))
            X_t_rows_p = ctypes.cast(X_t_rows.ctypes.data, ctypes.POINTER(ctypes.c_float))
            X_t_cols_p = ctypes.cast(X_t_cols.ctypes.data, ctypes.POINTER(ctypes.c_float))
            Phi_t_p = ctypes.cast(Phi_t.ctypes.data, ctypes.POINTER(ctypes.c_float))
            Theta_t_p = ctypes.cast(Theta_t.ctypes.data, ctypes.POINTER(ctypes.c_float))
            XVK_p = ctypes.cast(XVK.ctypes.data, ctypes.POINTER(ctypes.c_float))
            XKJ_p = ctypes.cast(XKJ.ctypes.data, ctypes.POINTER(ctypes.c_float))

            self._multi_aug(Params_p, X_t_values_p, X_t_rows_p, X_t_cols_p, Phi_t_p, Theta_t_p, XVK_p, XKJ_p, self.rand_status)

        return XKJ, XVK

    def crt_multi_aug(self, X_t, Phi_t, Theta_t, dtype='dense'):
        """
        sampler for the gamma distribution
        Inputs:
            X_t     : [np.ndarray];
            Phi_t   : [np.ndarray];
            Theta_t : [np.ndarray];
            dtype   : 'dense' or 'sparse';
        Outputs:
            output : [np.ndarray] or [pycuda.gpuarray] the resulting matrix on the device 'cpu' or 'gpu'
        """
        assert dtype in ['dense', 'sparse'], print(f'the dtype of X_t should be dense or sparse')

        if dtype == 'dense':

            X_t = np.array(X_t, dtype=np.float32, order='C')
            Phi_t = np.array(Phi_t, dtype=np.float32, order='C')
            Theta_t = np.array(Theta_t, dtype=np.float32, order='C')

            if len(X_t.shape) == 0:
                X_t = X_t[:, np.newaxis, np.newaxis]
            elif len(X_t.shape) == 1:
                X_t = X_t[:, np.newaxis]

            if len(Theta_t.shape) == 0:
                Theta_t = Theta_t[:, np.newaxis, np.newaxis]
            elif len(Theta_t.shape) == 1:
                Theta_t = Theta_t[:, np.newaxis]

            [X_t_rows, X_t_cols] = np.where(X_t > 0.5)
            X_t_values = X_t[(X_t_rows, X_t_cols)]
            [V, K] = Phi_t.shape
            J = Theta_t.shape[1]
            Num_Elems = len(X_t_values)
            Params = np.array([V, K, J, Num_Elems], dtype=np.int32, order='C')
        else:
            [X_t_rows, X_t_cols, X_t_values] = X_t[0], X_t[1], X_t[2]
            [V, K] = Phi_t.shape
            J = Theta_t.shape[1]
            Num_Elems = len(X_t_values)
            Params = np.array([V, K, J, Num_Elems], dtype=np.int32, order='C')

        # input variables
        X_t_values = np.array(X_t_values, dtype=np.float32, order='C')
        X_t_rows = np.array(X_t_rows, dtype=np.float32, order='C')
        X_t_cols = np.array(X_t_cols, dtype=np.float32, order='C')
        Phi_t = np.array(Phi_t, dtype=np.float32, order='C')
        Theta_t = np.array(Theta_t, dtype=np.float32, order='C')

        # output variables
        XVK = np.zeros([V, K], dtype=np.float32, order='C')
        XKJ = np.zeros([K, J], dtype=np.float32, order='C')

        if Num_Elems > 0:
            # pointer
            Params_p = ctypes.cast(Params.ctypes.data, ctypes.POINTER(ctypes.c_int))
            X_t_values_p = ctypes.cast(X_t_values.ctypes.data, ctypes.POINTER(ctypes.c_float))
            X_t_rows_p = ctypes.cast(X_t_rows.ctypes.data, ctypes.POINTER(ctypes.c_float))
            X_t_cols_p = ctypes.cast(X_t_cols.ctypes.data, ctypes.POINTER(ctypes.c_float))
            Phi_t_p = ctypes.cast(Phi_t.ctypes.data, ctypes.POINTER(ctypes.c_float))
            Theta_t_p = ctypes.cast(Theta_t.ctypes.data, ctypes.POINTER(ctypes.c_float))
            XVK_p = ctypes.cast(XVK.ctypes.data, ctypes.POINTER(ctypes.c_float))
            XKJ_p = ctypes.cast(XKJ.ctypes.data, ctypes.POINTER(ctypes.c_float))

            self._crt_multi_aug(Params_p, X_t_values_p, X_t_rows_p, X_t_cols_p, Phi_t_p, Theta_t_p, XVK_p, XKJ_p, self.rand_status)

        return XKJ, XVK

    def conv_multi_aug(self, X_rows, X_cols, X_pages, X_values, D1_k1, W1_nk1):
        """
        sampler for the gamma distribution
        Inputs:
            X_t     : [np.ndarray];
            Phi_t   : [np.ndarray];
            Theta_t : [np.ndarray];
            dtype   : 'dense' or 'sparse';
        Outputs:
            output : [np.ndarray] or [pycuda.gpuarray] the resulting matrix on the device 'cpu' or 'gpu'
        """
        [K1, K0, K1_S3, K1_S4] = D1_k1.shape
        [Num_Doc, K1, K1_S1, K1_S2] = W1_nk1.shape

        # 增广矩阵用于更新s12维度上增广
        X_rows = np.array(X_rows, dtype=np.float32, order='C')
        X_cols = np.array(X_cols+1, dtype=np.float32, order='C')
        X_pages = np.array(X_pages, dtype=np.float32, order='C')
        X_values = np.array(X_values, dtype=np.float32, order='C')
        Num_Elems = X_values.size # the number of word elements
        Params = np.array([K0, K1, K1_S1, K1_S2, K1_S3, K1_S4, Num_Elems, Num_Doc], dtype=np.int32, order='C')

        D1_k1 = np.array(np.swapaxes(D1_k1, 0, 1), dtype=np.float32, order='C')
        W1_nk1 = np.array(W1_nk1, dtype=np.float32, order='C')
        D1_k1_Aug = np.zeros(D1_k1.shape, dtype=np.float32, order='C')
        W1_nk1_Aug = np.zeros(W1_nk1.shape, dtype=np.float32, order='C')

        if Num_Elems > 0:
            # pointer
            Params_p = ctypes.cast(Params.ctypes.data, ctypes.POINTER(ctypes.c_int))
            X_rows_p = ctypes.cast(X_rows.ctypes.data, ctypes.POINTER(ctypes.c_float))
            X_cols_p = ctypes.cast(X_cols.ctypes.data, ctypes.POINTER(ctypes.c_float))
            X_pages_p = ctypes.cast(X_pages.ctypes.data, ctypes.POINTER(ctypes.c_float))
            X_values_p = ctypes.cast(X_values.ctypes.data, ctypes.POINTER(ctypes.c_float))
            D1_k1_p = ctypes.cast(D1_k1.ctypes.data, ctypes.POINTER(ctypes.c_float))
            W1_nk1_p = ctypes.cast(W1_nk1.ctypes.data, ctypes.POINTER(ctypes.c_float))
            D1_k1_Aug_p = ctypes.cast(D1_k1_Aug.ctypes.data, ctypes.POINTER(ctypes.c_float))
            W1_nk1_Aug_p = ctypes.cast(W1_nk1_Aug.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._conv_multi_aug(Params_p, X_rows_p, X_cols_p, X_pages_p, X_values_p, D1_k1_p, W1_nk1_p, D1_k1_Aug_p, W1_nk1_Aug_p, self.rand_status)
        # print(np.sum(W1_nk1), np.sum(D1_k1), np.sum(W1_nk1_Aug), np.sum(D1_k1_Aug))
        return W1_nk1_Aug, np.swapaxes(D1_k1_Aug, 0, 1)