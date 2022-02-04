import os
import numpy as np
import ctypes

from ._pre_process import para_preprocess


def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def get_nvcc_path():
    # get nvcc path
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = os.path.join(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        default_path = os.path.join(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path('nvcc', os.environ['PATH'] + os.pathsep + default_path)
        # if nvcc is None:
        #     raise EnvironmentError('The nvcc binary could not be '
        #                            'located in your $PATH. Either add it to your path, or set $CUDAHOME')
    return nvcc


class distribution_sampler_gpu(object):

    def __init__(self, system_type='Windows', seed=0):
        """
        The basic class for sampling distribution on cpu
        """
        super(distribution_sampler_gpu, self).__init__()

        self.system_type = system_type
        self.seed = seed

        # ------------------------------------------------ basic sampler ------------------------------------------
        if system_type == 'Windows':
            '''
            To compile CUDA C/C++ under Windows system, Visual Studio and CUDA should have been installed.
            This module has been tested under Visual Studio 2019(with MSVC v142 - VS 2019 C++ x64/x86 tools) and CUDA Toolkit 11.5.
            '''
            compact_path = os.path.dirname(__file__) + "\_compact\sampler_kernel.dll"
            if not os.path.exists(compact_path):
                nvcc_path = get_nvcc_path()
                try:
                    os.system(nvcc_path+' -o '+compact_path+' --shared '+compact_path[:-4]+'_win.cu')
                except:
                    os.system(r'nvcc -o '+'"'+compact_path+'"'+' --shared '+'"'+compact_path[:-4]+'_win.cu'+'"')
            dll = ctypes.cdll.LoadLibrary(compact_path)
        
        elif system_type == 'Linux':
            compact_path = os.path.dirname(__file__) + "/_compact/sampler_kernel.so"
            if not os.path.exists(compact_path):
                nvcc_path = get_nvcc_path()
                os.system(nvcc_path+' -Xcompiler -fPIC -shared -o '+compact_path+' '+compact_path[:-3]+'_linux.cu')
            dll = ctypes.cdll.LoadLibrary(compact_path)

        # ------------------------------------------------substorage ------------------------------------------
        self._init_status = dll._init_status
        self._init_status.argtypes = [ctypes.c_size_t]
        self._init_status.restype = ctypes.c_void_p
        self.rand_status = self._init_status(self.seed)

        # ----------------------------------------------cuda sampler ------------------------------------------
        self._sample_gamma = dll._sample_gamma
        self._sample_gamma.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_standard_gamma = dll._sample_standard_gamma
        self._sample_standard_gamma.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_beta = dll._sample_beta
        self._sample_beta.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_normal = dll._sample_normal
        self._sample_normal.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_standard_normal = dll._sample_standard_normal
        self._sample_standard_normal.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p]

        self._sample_uniform = dll._sample_uniform
        self._sample_uniform.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_standard_uniform = dll._sample_standard_uniform
        self._sample_standard_uniform.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p]

        # self._sample_binomial = dll._sample_binomial
        # self._sample_binomial.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_negative_binomial = dll._sample_negative_binomial
        self._sample_negative_binomial.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_multinomial = dll._sample_multinomial
        self._sample_multinomial.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_poisson = dll._sample_poisson
        self._sample_poisson.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_crt = dll._sample_crt
        self._sample_crt.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_cauchy = dll._sample_cauchy
        self._sample_cauchy.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_standard_cauchy = dll._sample_standard_cauchy
        self._sample_standard_cauchy.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p]

        self._sample_chisquare = dll._sample_chisquare
        self._sample_chisquare.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_noncentral_chisquare = dll._sample_noncentral_chisquare
        self._sample_noncentral_chisquare.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_exponential = dll._sample_exponential
        self._sample_exponential.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_f = dll._sample_f
        self._sample_f.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_noncentral_f = dll._sample_noncentral_f
        self._sample_noncentral_f.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_geometric = dll._sample_geometric
        self._sample_geometric.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_gumbel = dll._sample_gumbel
        self._sample_gumbel.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_hypergeometric = dll._sample_hypergeometric
        self._sample_hypergeometric.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_laplace = dll._sample_laplace
        self._sample_laplace.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_logistic = dll._sample_logistic
        self._sample_logistic.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_power = dll._sample_power
        self._sample_power.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_zipf = dll._sample_zipf
        self._sample_zipf.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_pareto = dll._sample_pareto
        self._sample_pareto.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_rayleigh = dll._sample_rayleigh
        self._sample_rayleigh.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_t = dll._sample_t
        self._sample_t.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_triangular = dll._sample_triangular
        self._sample_triangular.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_weibull = dll._sample_weibull
        self._sample_weibull.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


    def gamma(self, shape, scale=1.0, times=1):
        """
        sampler for the gamma distribution
        Inputs:
            shape  : [float] or [np.ndarray] shape parameter;
            scale  : [float] or [np.ndarray] scale parameter;
            times  : [int] the times required to sample;
            device : [str] 'cpu' or 'gpu';
        Outputs:
            output : [np.ndarray] or [pycuda.gpuarray] the resulting matrix on the device 'cpu' or 'gpu'
        """
        # in: non-negative
        matrix_scale, nElems, shape, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, shape, scale)
        shape_p = ctypes.cast(shape.ctypes.data, ctypes.POINTER(ctypes.c_float))
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_gamma(shape_p, scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def standard_gamma(self, shape, times=1):
        # shape: non-negative
        matrix_scale, nElems, shape, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, shape)
        shape_p = ctypes.cast(shape.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_standard_gamma(shape_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def dirichlet(self, shape, times=1):  # cacul by gamma. _sampler and cupy all take this way
        matrix_scale, nElems, shape, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, shape)
        shape_p = ctypes.cast(shape.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_standard_gamma(shape_p, output_p, matrix_scale, times, self.rand_status)
        if times == 1:
            output = output / np.sum(output, axis=-1, keepdims=True)
        else:
            output = output / np.sum(output, axis=-2, keepdims=True)
        return output[0] if scalar_flag else output

    def beta(self, a, b, times=1):
        # in: positive
        matrix_scale, nElems, a, b, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, a, b)
        a_p = ctypes.cast(a.ctypes.data, ctypes.POINTER(ctypes.c_float))
        b_p = ctypes.cast(b.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_beta(a_p, b_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def normal(self, loc=0.0, scale=1.0, times=1):
        # scale: non-negative
        matrix_scale, nElems, loc, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, loc, scale)
        loc_p = ctypes.cast(loc.ctypes.data, ctypes.POINTER(ctypes.c_float))
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_normal(loc_p, scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def standard_normal(self, size=1):
        assert type(size) == int or type(size) == tuple, "param size(Output shape) should be int or tuple of ints"
        nElems = size if type(size) == int else np.prod(size)
        output = np.empty(size, dtype=np.float32, order='C')
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_standard_normal(output_p, nElems, self.rand_status)
        return output[0] if size == 1 else output

    def uniform(self, low=0.0, high=1.0, times=1):
        # low < high
        matrix_scale, nElems, low, high, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, low, high)
        low_p = ctypes.cast(low.ctypes.data, ctypes.POINTER(ctypes.c_float))
        high_p = ctypes.cast(high.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_uniform(low_p, high_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def standard_uniform(self, size=1):
        assert type(size) == int or type(size) == tuple, "param size(Output shape) should be int or tuple of ints"
        nElems = size if type(size) == int else np.prod(size)
        output = np.empty(size, dtype=np.float32, order='C')
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_standard_uniform(output_p, nElems, self.rand_status)
        return output[0] if size == 1 else output

    def binomial(self, count=1, prob=0.5, times=1):  # 可以进行很深的优化cupy, 但依然转调multinomial
        """
        sampler for the gamma distribution
        Inputs:
            count  : [int] or [np.ndarray] count parameter;
            prob   : [float] or [np.ndarray] prob parameter, between 0 and 1;
            times  : [int] the times required to sample;
            device : [str] 'cpu' or 'gpu';
        Outputs:
            output : [np.ndarray] or [pycuda.gpuarray] the resulting matrix on the device 'cpu' or 'gpu'
        """
        count = np.array(count, dtype=np.int32, order='C')
        prob = np.array(prob, dtype=np.float32, order='C')

        assert count.size == prob.size, 'Param count and prob should have the same length.'
        assert len(count.shape) <= 2, 'Shape Error: the dimension of the input parameter a in the sampling distirbution shoud not be larger than 2'
        assert len(prob.shape) <= 2, 'Shape Error: the dimension of the input parameter b in the sampling distirbution shoud not be larger than 2'

        if count.size == prob.size:
            prob = np.expand_dims(prob, axis=-1)

        if prob.shape[-1] == 1:
            del_prob = 1 - prob
            prob = np.concatenate([prob, del_prob], axis=-1)
        return self.multinomial(count, prob, times)

    def negative_binomial(self, r, p, times=1):
        # r: *int, larger than 1
        # p: between 0 and 1
        # output: int
        matrix_scale, nElems, r, p, output, output_scale, scalar_flag = para_preprocess(times, [np.int32, np.float32], np.int32, r, p)
        r_p = ctypes.cast(r.ctypes.data, ctypes.POINTER(ctypes.c_int))
        p_p = ctypes.cast(p.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_int))

        self._sample_negative_binomial(r_p, p_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def multinomial(self, count=[1, 1], prob=[[0.5, 0.5], [0.2, 0.8]], times=1):
        """
        sampler for the multi distribution
        Inputs:
            count  : [int] or [np.ndarray] count parameter;
            prob   : [float] or [np.ndarray] prob parameter;
            times  : [int] the times required to sample;
            device : [str] 'cpu' or 'gpu';
        Outputs:
            output : [np.ndarray] or [pycuda.gpuarray] the resulting matrix on the device 'cpu' or 'gpu'
        """
        # in: non-negative
        # sum prob in a axis should be equal to 1
        count = np.array(count, dtype=np.int32, order='C')
        prob = np.array(prob, dtype=np.float32, order='C')

        assert count.size == 1 or count.size == prob.shape[0], 'Shape Error: '
        assert len(count.shape) <= 2, 'Shape Error: the dimension of the input parameter a in the sampling distirbution shoud not be larger than 2'
        assert len(prob.shape) <= 2, 'Shape Error: the dimension of the input parameter b in the sampling distirbution shoud not be larger than 2'

        output_scale = prob.shape if times == 1 or (times == 1 and count.size == 1) else prob.shape + (times,)
        output = np.zeros(output_scale, dtype=np.int32, order='C')
        matrix_scale_1 = count.size
        matrix_scale_2 = prob.size

        count_p = ctypes.cast(count.ctypes.data, ctypes.POINTER(ctypes.c_int))
        prob_p = ctypes.cast(prob.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_int))

        self._sample_multinomial(count_p, prob_p, output_p, matrix_scale_1, matrix_scale_2, times, self.rand_status)
        return output

    def poisson(self, lam=1.0, times=1):
        # lam: non-negative
        # output: int
        matrix_scale, nElems, lam, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.int32, lam)
        lam_p = ctypes.cast(lam.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_int))

        self._sample_poisson(lam_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def crt(self, customers, prob, times=1):  # dirichlet相近
        # Chinese restaurant process
        # https://qianyang-hfut.blog.csdn.net/article/details/52371443
        # customers 第几个顾客 prob开新比例 out桌子数量
        # customers: int
        # prob:
        # out: int
        matrix_scale, nElems, customers, prob, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.int32, customers, prob)
        customers_p = ctypes.cast(customers.ctypes.data, ctypes.POINTER(ctypes.c_float))
        prob_p = ctypes.cast(prob.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_int))

        self._sample_crt(customers_p, prob_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def cauchy(self, loc=0.0, scale=1.0, times=1):
        matrix_scale, nElems, loc, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, loc, scale)
        loc_p = ctypes.cast(loc.ctypes.data, ctypes.POINTER(ctypes.c_float))
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_cauchy(loc_p, scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def standard_cauchy(self, size=1):
        assert type(size) == int or type(size) == tuple, "param size(Output shape) should be int or tuple of ints"
        nElems = size if type(size) == int else np.prod(size)
        output = np.empty(size, dtype=np.float32, order='C')
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_standard_cauchy(output_p, nElems, self.rand_status)
        return output[0] if size == 1 else output

    def chisquare(self, degrees, times=1):
        # degrees int
        matrix_scale, nElems, degrees, output, output_scale, scalar_flag = para_preprocess(times, np.int32, np.float32, degrees)
        degrees_p = ctypes.cast(degrees.ctypes.data, ctypes.POINTER(ctypes.c_int))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_chisquare(degrees_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def noncentral_chisquare(self, df, nonc, times=1):
        # df: int
        # nonc: Non-centrality, must be non-negative.
        matrix_scale, nElems, df, nonc, output, output_scale, scalar_flag = para_preprocess(times, [np.int32, np.float32], np.float32, df, nonc)
        df_p = ctypes.cast(df.ctypes.data, ctypes.POINTER(ctypes.c_int))
        nonc_p = ctypes.cast(nonc.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_noncentral_chisquare(df_p, nonc_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def exponential(self, Lambda=1.0, times=1):
        matrix_scale, nElems, Lambda, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, Lambda)
        Lambda_p = ctypes.cast(Lambda.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_exponential(Lambda_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def standard_exponential(self, size=1):
        assert type(size) == int or type(size) == tuple, "param size(Output shape) should be int or tuple of ints"
        matrix_scale = size if type(size) == int else np.prod(size)
        Lambda = np.ones(size, dtype=np.float32, order='C')
        Lambda_p = ctypes.cast(Lambda.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output = np.empty(size, dtype=np.float32, order='C')
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_exponential(Lambda_p, output_p, matrix_scale, 1, self.rand_status)
        return output[0] if size == 1 else output

    def f(self, n1, n2, times=1):
        # n1, n2: int
        matrix_scale, nElems, n1, n2, output, output_scale, scalar_flag = para_preprocess(times, np.int32, np.float32, n1, n2)
        n1_p = ctypes.cast(n1.ctypes.data, ctypes.POINTER(ctypes.c_int))
        n2_p = ctypes.cast(n2.ctypes.data, ctypes.POINTER(ctypes.c_int))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_f(n1_p, n2_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def noncentral_f(self, dfnum, dfden, nonc, times=1):
        # dfnum, Numerator degrees of freedom
        # dfden, Denominator degrees of freedom, must be > 0.
        matrix_scale, nElems, dfnum, nonc, output, output_scale, scalar_flag = para_preprocess(times, [np.int32, np.float32], np.float32, dfnum, nonc)
        if type(dfden) in [float, int]:
            dfden = np.array([dfden] * matrix_scale, dtype=np.int32, order='C')
        else:
            dfden = np.array(dfden, dtype=np.int32, order='C')
            assert dfden.shape == dfnum.shape, 'param dfden should be scalar or have the same shape as dfnum or nonc.'

        dfnum_p = ctypes.cast(dfnum.ctypes.data, ctypes.POINTER(ctypes.c_int))
        dfden_p = ctypes.cast(dfden.ctypes.data, ctypes.POINTER(ctypes.c_int))
        nonc_p = ctypes.cast(nonc.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_noncentral_f(dfnum_p, dfden_p, nonc_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def geometric(self, p, times=1):
        # p: (0, 1)
        # output: int
        matrix_scale, nElems, p, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.int32, p)
        p_p = ctypes.cast(p.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_int))

        self._sample_geometric(p_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def gumbel(self, loc=0.0, scale=1.0, times=1):
        # scale: non-negative
        matrix_scale, nElems, loc, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, loc, scale)
        loc_p = ctypes.cast(loc.ctypes.data, ctypes.POINTER(ctypes.c_float))
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_gumbel(loc_p, scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def hypergeometric(self, ngood, nbad, nsample, times=1):
        # all input int
        # out int
        assert '等长?....'  # should be ngood.size >= nbad.size
        matrix_scale, nElems, ngood, nsample, output, output_scale, scalar_flag = para_preprocess(times, np.int32, np.int32, ngood, nsample)
        if type(nbad) in [float, int]:
            nbad = np.array([nbad] * matrix_scale, dtype=np.int32, order='C')
        else:
            nbad = np.array(nbad, dtype=np.int32, order='C')
            assert nbad.shape == ngood.shape, 'param nbad should be scalar or have the same shape as ngood or nsample.'
        ngood_p = ctypes.cast(ngood.ctypes.data, ctypes.POINTER(ctypes.c_int))
        nbad_p = ctypes.cast(nbad.ctypes.data, ctypes.POINTER(ctypes.c_int))
        nsample_p = ctypes.cast(nsample.ctypes.data, ctypes.POINTER(ctypes.c_int))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_int))

        self._sample_hypergeometric(ngood_p, nbad_p, nsample_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def laplace(self, loc=0.0, scale=1.0, times=1):
        # scale: non-negative
        matrix_scale, nElems, loc, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, loc, scale)
        loc_p = ctypes.cast(loc.ctypes.data, ctypes.POINTER(ctypes.c_float))
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_laplace(loc_p, scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def logistic(self, loc=0.0, scale=1.0, times=1):
        # scale: non-negative
        matrix_scale, nElems, loc, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, loc, scale)
        loc_p = ctypes.cast(loc.ctypes.data, ctypes.POINTER(ctypes.c_float))
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_logistic(loc_p, scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def power(self, a, times=1):
        # a: non-negative
        matrix_scale, nElems, a, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, a)
        a_p = ctypes.cast(a.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_power(a_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def zipf(self, a, times=1):
        # a: > 1
        # output: int
        matrix_scale, nElems, a, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.int32, a)
        a_p = ctypes.cast(a.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_int))

        self._sample_zipf(a_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def pareto(self, k, xm=1, times=1):
        # k > 1 (sampler, why?) 幂级数
        matrix_scale, nElems, k, xm, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, k, xm)
        k_p = ctypes.cast(k.ctypes.data, ctypes.POINTER(ctypes.c_float))
        xm_p = ctypes.cast(xm.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_pareto(k_p, xm_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def rayleigh(self, scale=1.0, times=1):
        # scale: non-negative
        matrix_scale, nElems, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, scale)
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_rayleigh(scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def t(self, df, times=1):
        # df: positive
        matrix_scale, nElems, df, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, df)
        df_p = ctypes.cast(df.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_t(df_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def triangular(self, left, mode, right, times=1):
        # 等长
        # left <= mode <= right
        # left < right
        assert '等长?....'
        matrix_scale, nElems, left, mode, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, left, mode)
        if type(right) in [float, int]:
            right = np.array([right]*matrix_scale, dtype=np.float32, order='C')
        else:
            right = np.array(right, dtype=np.float32, order='C')
            assert right.shape == left.shape, 'param right should be scalar or have the same shape as left or mode.'
        left_p = ctypes.cast(left.ctypes.data, ctypes.POINTER(ctypes.c_float))
        mode_p = ctypes.cast(mode.ctypes.data, ctypes.POINTER(ctypes.c_float))
        right_p = ctypes.cast(right.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_triangular(left_p, mode_p, right_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output

    def weibull(self, shape, scale, times=1):
        # a: non-negative (np. said)
        matrix_scale, nElems, shape, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, shape, scale)
        shape_p = ctypes.cast(shape.ctypes.data, ctypes.POINTER(ctypes.c_float))
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_weibull(shape_p, scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output



