import os
import numpy as np
import ctypes

from ._pre_process import para_preprocess


class distribution_sampler_gpu(object):

    def __init__(self, system_type='Windows', seed=0):
        """
        The basic class for sampling distribution on cpu
        """
        super(distribution_sampler_gpu, self).__init__()

        self.system_type = system_type
        self.seed = seed

        if system_type == 'Windows':
            # ------------------------------------------------ basic sampler ------------------------------------------
            compact_path = os.path.dirname(__file__) + ".\_compact\sampler_kernel.dll"
            dll = ctypes.cdll.LoadLibrary(compact_path)

        elif system_type == 'Linux':
            # ------------------------------------------------ basic sampler ------------------------------------------
            compact_path = os.path.dirname(__file__) + ".\_compact\sampler_kernel.so"
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
        self._sample_noncentral_chisquare.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_exponential = dll._sample_exponential
        self._sample_exponential.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_f = dll._sample_f
        self._sample_f.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_geometric = dll._sample_geometric
        self._sample_geometric.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_gumbel = dll._sample_gumbel
        self._sample_gumbel.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        self._sample_hypergeometric = dll._sample_hypergeometric
        self._sample_hypergeometric.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

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

    def dirichlet(self, shape, times=1):
        matrix_scale, nElems, shape, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, shape)
        shape_p = ctypes.cast(shape.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_standard_gamma(shape_p, output_p, matrix_scale, times, self.rand_status)
        if times == 1:
            output = output / np.sum(output, axis=-1, keepdims=True)
        else:
            output = output / np.sum(output, axis=-2, keepdims=True)
        return output[0] if scalar_flag else output

    # cacul by gamma. _sampler and cupy all take this way, but there are some differences between them. Need 2
    def beta(self, a, b, times=1):
        # in: positive
        matrix_scale, nElems, a, b, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, a, b)
        a_p = ctypes.cast(a.ctypes.data, ctypes.POINTER(ctypes.c_float))
        b_p = ctypes.cast(b.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_beta(a_p, b_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
    def normal(self, loc=0.0, scale=1.0, times=1):
        # scale: non-negative
        matrix_scale, nElems, loc, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, loc, scale)
        loc_p = ctypes.cast(loc.ctypes.data, ctypes.POINTER(ctypes.c_float))
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_normal(loc_p, scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
    def standard_normal(self, size=1):  # times换一种方式:size，int or tuple of ints, optional
        assert type(size) == int or type(size) == tuple, "param size(Output shape) should be int or tuple of ints"
        nElems = size if type(size) == int else np.prod(size)
        output = np.empty(size, dtype=np.float32, order='C')
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_standard_normal(output_p, nElems, self.rand_status)
        return output[0] if size == 1 else output
    #
    def uniform(self, low=0.0, high=1.0, times=1):
        # low < high
        matrix_scale, nElems, low, high, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, low, high)
        low_p = ctypes.cast(low.ctypes.data, ctypes.POINTER(ctypes.c_float))
        high_p = ctypes.cast(high.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_uniform(low_p, high_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
    def standard_uniform(self, size=1):
        assert type(size) == int or type(size) == tuple, "param size(Output shape) should be int or tuple of ints"
        nElems = size if type(size) == int else np.prod(size)
        output = np.empty(size, dtype=np.float32, order='C')
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_standard_uniform(output_p, nElems, self.rand_status)
        return output[0] if size == 1 else output
    #
    def binomial(self, count=1, prob=0.5, times=1):  # 可以进行很深的优化cupy, 但依然转调multinomial
        """
        sampler for the gamma distribution
        Inputs:
            count  : [int] or [np.ndarray] count parameter;
            prob   : [float] or [np.ndarray] prob parameter;
            times  : [int] the times required to sample;
            device : [str] 'cpu' or 'gpu';
        Outputs:
            output : [np.ndarray] or [pycuda.gpuarray] the resulting matrix on the device 'cpu' or 'gpu'
        """
        count = np.array(count, dtype=np.int32, order='C')
        prob = np.array(prob, dtype=np.float32, order='C')

        assert len(count.shape) <= 2, 'Shape Error: the dimension of the input parameter a in the sampling distirbution shoud not be larger than 2'
        assert len(prob.shape) <= 2, 'Shape Error: the dimension of the input parameter b in the sampling distirbution shoud not be larger than 2'

        if count.size == prob.size:
            prob = np.expand_dims(prob, axis=-1)

        if prob.shape[-1] == 1:
            del_prob = 1 - prob
            prob = np.concatenate([prob, del_prob], axis=-1)
        return self.multinomial(count, prob, times)
    # 转调multinomial需要进行修改
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

    def multinomial(self, count=[1, 1], prob=[[0.5, 0.5], [0.2, 0.3]], times=1):
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
        count = np.array(count, dtype=np.int32, order='C')
        prob = np.array(prob, dtype=np.float32, order='C')

        assert len(count.shape) <= 2, 'Shape Error: the dimension of the input parameter a in the sampling distirbution shoud not be larger than 2'
        assert len(prob.shape) <= 2, 'Shape Error: the dimension of the input parameter b in the sampling distirbution shoud not be larger than 2'

        output_scale = prob.shape + (times, )
        output = np.zeros(output_scale, dtype=np.int32, order='C')
        matrix_scale_1 = count.size
        matrix_scale_2 = prob.size

        count_p = ctypes.cast(count.ctypes.data, ctypes.POINTER(ctypes.c_int))
        prob_p = ctypes.cast(prob.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_int))

        self._sample_multinomial(count_p, prob_p, output_p, matrix_scale_1, matrix_scale_2, times, self.rand_status)
        return output
    # input param count must be int scalar, just as numpy
    def poisson(self, lam=1.0, times=1):
        # lam: non-negative
        # output: int
        matrix_scale, nElems, lam, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.int32, lam)
        lam_p = ctypes.cast(lam.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_int))

        self._sample_poisson(lam_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
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
    #
    def cauchy(self, loc=0.0, scale=1.0, times=1):
        matrix_scale, nElems, loc, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, loc, scale)
        loc_p = ctypes.cast(loc.ctypes.data, ctypes.POINTER(ctypes.c_float))
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_cauchy(loc_p, scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
    def standard_cauchy(self, size=1):
        assert type(size) == int or type(size) == tuple, "param size(Output shape) should be int or tuple of ints"
        nElems = size if type(size) == int else np.prod(size)
        output = np.empty(size, dtype=np.float32, order='C')
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_standard_cauchy(output_p, nElems, self.rand_status)
        return output[0] if size == 1 else output
    #
    def chisquare(self, degrees, times=1):
        # degrees int
        matrix_scale, nElems, degrees, output, output_scale, scalar_flag = para_preprocess(times, np.int32, np.float32, degrees)
        degrees_p = ctypes.cast(degrees.ctypes.data, ctypes.POINTER(ctypes.c_int))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_chisquare(degrees_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
    def noncentral_chisquare(self, degrees, loc, scale, times=1):
        # degrees int
        # scale: non-negative
        assert '等长?....'
        matrix_scale, nElems, degrees, loc, output, output_scale, scalar_flag = para_preprocess(times, [np.int32, np.float32], np.float32, degrees, loc)
        if type(scale) in [float, int]:
            scale = np.array([scale]*matrix_scale, dtype=np.float32, order='C')
        else:
            scale = np.array(scale, dtype=np.float32, order='C')
            assert scale.shape == loc.shape, 'param scale should be scalar or have the same shape as degrees or loc.'
        degrees_p = ctypes.cast(degrees.ctypes.data, ctypes.POINTER(ctypes.c_int))
        loc_p = ctypes.cast(loc.ctypes.data, ctypes.POINTER(ctypes.c_float))
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_noncentral_chisquare(degrees_p, loc_p, scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    # without comparsion. the sampler is differ from np and stats.
    def exponential(self, Lambda=1.0, times=1):
        matrix_scale, nElems, Lambda, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, Lambda)
        Lambda_p = ctypes.cast(Lambda.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_exponential(Lambda_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
    def standard_exponential(self, size=1):
        assert type(size) == int or type(size) == tuple, "param size(Output shape) should be int or tuple of ints"
        matrix_scale = size if type(size) == int else np.prod(size)
        Lambda = np.ones(size, dtype=np.float32, order='C')
        Lambda_p = ctypes.cast(Lambda.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output = np.empty(size, dtype=np.float32, order='C')
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_exponential(Lambda_p, output_p, matrix_scale, 1, self.rand_status)
        return output[0] if size == 1 else output
    #
    def f(self, n1, n2, times=1):
        # n1, n2: int
        matrix_scale, nElems, n1, n2, output, output_scale, scalar_flag = para_preprocess(times, np.int32, np.float32, n1, n2)
        n1_p = ctypes.cast(n1.ctypes.data, ctypes.POINTER(ctypes.c_int))
        n2_p = ctypes.cast(n2.ctypes.data, ctypes.POINTER(ctypes.c_int))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_f(n1_p, n2_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
    def noncentral_f(self, n1, n2, loc, scale, times=1):
        # n1, n2: int
        # scale: non-negative
        assert '等长'
        matrix_scale, nElems, n1, n2, output, output_scale, scalar_flag = para_preprocess(times, np.int32, np.float32, n1, n2)
        # param scale need 2 be modefied
        loc = np.array(loc, dtype=np.float32, order='C')
        loc = self.ndarray2c_ptr(loc)
        scale = np.array(scale, dtype=np.float32, order='C')
        scale = self.ndarray2c_ptr(scale)
        self._sample_noncentral_normal(n1, n2, loc, scale, output, matrix_scale, times, self.rand_status)
        output = self.output_to_cpu(output, nElems, np.float32)[:nElems].reshape(output_scale)
        output = output[0] if scalar_flag else output

        return output
    # later
    def geometric(self, p, times=1):
        # p: (0, 1)
        # output: int
        matrix_scale, nElems, p, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.int32, p)
        p_p = ctypes.cast(p.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_int))

        self._sample_geometric(p_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
    def gumbel(self, loc=0.0, scale=1.0, times=1):
        # scale: non-negative
        matrix_scale, nElems, loc, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, loc, scale)
        loc_p = ctypes.cast(loc.ctypes.data, ctypes.POINTER(ctypes.c_float))
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_gumbel(loc_p, scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
    def hypergeometric(self, ngood, nbad, nsample, times=1):
        # all input int
        # out int
        assert '等长'  # should be ngood.size >= nbad.size
        prob = np.array(ngood) / (np.array(ngood) + np.array(nbad))
        matrix_scale, nElems, prob, nsample, output, output_scale, scalar_flag = para_preprocess(times, [np.float32, np.int32], np.int32, prob, nsample)

        prob_p = ctypes.cast(prob.ctypes.data, ctypes.POINTER(ctypes.c_float))
        nsample_p = ctypes.cast(nsample.ctypes.data, ctypes.POINTER(ctypes.c_int))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_int))

        self._sample_hypergeometric(prob_p, nsample_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    # the sample of _sampler is wrong. It should be without replacement.
    def laplace(self, loc=0.0, scale=1.0, times=1):
        # scale: non-negative
        matrix_scale, nElems, loc, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, loc, scale)
        loc_p = ctypes.cast(loc.ctypes.data, ctypes.POINTER(ctypes.c_float))
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_laplace(loc_p, scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
    def logistic(self, loc=0.0, scale=1.0, times=1):
        # scale: non-negative
        matrix_scale, nElems, loc, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, loc, scale)
        loc_p = ctypes.cast(loc.ctypes.data, ctypes.POINTER(ctypes.c_float))
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_logistic(loc_p, scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
    def power(self, a, times=1):
        # a: non-negative
        matrix_scale, nElems, a, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, a)
        a_p = ctypes.cast(a.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_power(a_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
    def zipf(self, a, times=1):
        # a: > 1
        # output: int
        matrix_scale, nElems, a, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.int32, a)
        a_p = ctypes.cast(a.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_int))

        self._sample_zipf(a_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
    def pareto(self, k, xm=1, times=1):
        # k > 1 (sampler, why?) 幂级数
        matrix_scale, nElems, k, xm, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, k, xm)
        k_p = ctypes.cast(k.ctypes.data, ctypes.POINTER(ctypes.c_float))
        xm_p = ctypes.cast(xm.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_pareto(k_p, xm_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    # without comparison. unknown definition. (_sampler) differ from numpy
    def rayleigh(self, scale=1.0, times=1):
        # scale: non-negative
        matrix_scale, nElems, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, scale)
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_rayleigh(scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
    def t(self, df, times=1):
        # df: positive
        matrix_scale, nElems, df, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, df)
        df_p = ctypes.cast(df.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_t(df_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #
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
    #
    def weibull(self, shape, scale, times=1):
        # a: non-negative (np. said)
        matrix_scale, nElems, shape, scale, output, output_scale, scalar_flag = para_preprocess(times, np.float32, np.float32, shape, scale)
        shape_p = ctypes.cast(shape.ctypes.data, ctypes.POINTER(ctypes.c_float))
        scale_p = ctypes.cast(scale.ctypes.data, ctypes.POINTER(ctypes.c_float))
        output_p = ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float))

        self._sample_weibull(shape_p, scale_p, output_p, matrix_scale, times, self.rand_status)
        return output[0] if scalar_flag else output
    #