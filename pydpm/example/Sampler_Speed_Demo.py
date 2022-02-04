import inspect
import numpy as np


def getSamplerFun(sampler, distributions):
    """
    get distributions' function by class sampler.
    Inputs:
        sampler       : sampler class, pydpm._sampler.Basic_Sampler;
        distributions : [str] list of distributions;
    Outputs:
        funcs         : {str: fun}, dict of distributions' functions;
    """
    # get funcs
    funcs = {}
    for dist in distributions:
        funcs[dist] = getattr(sampler, dist)

    return funcs


def getSamplerParams(funcs, distributions, shape):
    """
    get distributions' params by sampler' funcs.
    Inputs:
        funcs         : {str: fun}, dict of distributions' functions;
        distributions : [str] list of distributions;
        shape         : int or tuple, params' shape.
    Outputs:
        params        : {str: [params]}, list of params of each distribution.
    """
    # get params
    int_1 = np.ones(shape, dtype=np.int32, order='C')
    float_05 = np.ones(shape, dtype=np.float32, order='C') / 2
    float_50 = np.ones(shape, dtype=np.float32, order='C') * 5
    int_8 = np.ones(shape, dtype=np.int32, order='C') * 8

    params = {}
    params_special = {'f': [int_8, int_1],
                      'hypergeometric': [int_8, np.ones(shape, dtype=np.int32, order='C')*5, int_8],
                      'multinomial': [5, [0.5, 0.3, 0.2], 100000],
                      'pareto': [float_50, float_05],
                      'triangular': [float_05, float_50, np.ones(shape, dtype=np.float32, order='C')],
                      'uniform': [float_50, float_50],
                      'zipf': [float_50], }
    # 特殊，gamma存在shape的分支，geometric存在采样时间的问题
    init_param = {'size': shape}  # , 'times': 1}
    for p in ['a', 'b', 'prob', 'loc', 'scale', 'shape', 'Lambda', 'p', 'nonc', 'lam']:
        init_param[p] = float_05
    for p in ['count', 'degrees', 'customers', 'n1', 'n2', 'r', 'df', 'dfnum', 'dfden']:
        init_param[p] = int_8
    for dist in distributions:  # 通过param自动填充参数
        param = []
        if dist in params_special.keys():
            params[dist] = params_special[dist]
        else:
            print("%12s" % dist, inspect.signature(funcs[dist]).parameters)
            # print(' '*12, inspect.signature(getattr(np.random, dist)).parameters)
            for p in inspect.signature(funcs[dist]).parameters:
                if p != 'times':
                    param.append(init_param[p])
            params[dist] = param

    return params


def getTorchFunandParams(distributions, shape):
    '''
    get distributions' functions and params from torch.distributions.

    Inputs:
        distributions : [str] list of distributions;
        shape         : int or tuple, params' shape.
    Outputs:
        funcs         : {str: fun}, dict of distributions' functions;
        params        : {str: [params]}, list of params of each distribution.
    '''
    import torch
    funcs = {'beta': torch.distributions.beta.Beta,
                   'exponential': torch.distributions.Exponential,
                   'gamma': torch.distributions.gamma.Gamma,
                   'laplace': torch.distributions.laplace.Laplace,
                   'normal': torch.distributions.Normal,
                   'poisson': torch.distributions.Poisson, }
    # beta(a, b)
    # exp(Lambda)
    # gamma(shape, scale)
    # laplace(loc, scale)
    # normal(loc, scale)
    # poisson(lam)
    float_05 = torch.ones(shape) / 2
    params = {'beta': [float_05, float_05],
              'exponential': [float_05],
              'gamma': [float_05, float_05],
              'laplace': [float_05, float_05],
              'normal': [float_05, float_05],
              'poisson': [float_05], }

    return funcs, params


def getTensorflowFunandParams(distributions, shape):
    '''
    get distributions' functions and params from tensorflow.

    Inputs:
        distributions : [str] list of distributions;
        shape         : int or tuple, params' shape.
    Outputs:
        funcs         : {str: fun}, dict of distributions' functions;
        params        : {str: [params]}, list of params of each distribution.
    '''
    import tensorflow as tf
    import tensorflow_probability as tfp

    funcs = {'beta': tfp.distributions.Beta,
                   'exponential': tfp.distributions.Exponential,
                   'gamma': tfp.distributions.gamma.Gamma,
                   'laplace': tfp.distributions.laplace.Laplace,
                   'normal': tfp.distributions.Normal,
                   'poisson': tfp.distributions.Poisson, }
    float_05 = tf.ones(shape) / 2
    params = {'beta': [float_05, float_05],
              'exponential': [float_05],
              'gamma': [float_05, float_05],
              'laplace': [float_05, float_05],
              'normal': [float_05, float_05],
              'poisson': [float_05], }

    return funcs, params


def repeatSamplerFuncs(funcs, params, repeats, type='pydpm'):
    '''
    sample each distributions funcs repeats times and calculate the average times cost.

    Inputs:
        funcs     : [samplerFuncs]
        params    : [[param] of each dist]
        repeats   : int, repeats times;
        typt      : str, cu, numpy, torch or tensorflow
    Outputs:
        avgt      : [float], average times of each sampler funcs.
    '''
    assert type in ['pydpm', 'numpy', 'torch', 'tensorflow']
    avgt = []
    for dist in distributions:
        start_time = time.time()
        for _ in range(repeats):
            if (type == 'pydpm' or type == 'numpy'):
                funcs[dist](*params[dist])
            else:
                funcs[dist](*params[dist]).sample()
        avgt.append((time.time() - start_time) / repeats)
    return avgt


def plotComparision(distributions, title, types, *args):
    num_types = len(types)
    assert (num_types == len(args)), 'types and avgts do not correspond'
    bar_width = 1 / (num_types + 1)
    index = np.arange(len(distributions))
    colors = ['dodgerblue', 'gray', 'peru']
    for i in range(num_types):
        plt.bar(index + i * bar_width, height=args[i], width=bar_width, color=colors[i], label=types[i])
    plt.legend()
    plt.xticks(index + bar_width/2, distributions, rotation=90)
    plt.ylabel('time(s)')
    # plt.title(title)
    plt.tight_layout()
    plt.show()


# -----------------test the accuracy --------------------
if __name__ == "__main__":
    from pydpm._sampler import Basic_Sampler
    import matplotlib.pyplot as plt
    import time

    sampler = Basic_Sampler('gpu')

    # all distributions: a part of 'distributions = sampler.__dict__.keys()'
    '''distributions = ['beta', 'binomial', 'cauchy', 'chisquare', 'crt', 'dirichlet', 'exponential', 'f', 'gamma', 'geometric', 'gumbel',
    'hypergeometric', 'laplace', 'logistic', 'multinomial', 'negative_binomial', 'noncentral_chisquare', 'noncentral_f',
    'normal', 'pareto', 'poisson', 'power', 'rayleigh', 'standard_cauchy', 'standard_exponential',
    'standard_gamma', 'standard_normal', 'standard_uniform', 't', 'triangular', 'uniform', 'weibull', 'zipf']'''

    shape = (999, 128)  # shape of distributions' params
    repeats = 100

    ## Compare with numpy
    distributions = ['beta', 'chisquare', 'exponential', 'f', 'gamma', 'geometric', 'gumbel', 'laplace', 'logistic',
                     'multinomial', 'negative_binomial', 'normal', 'poisson', 'power', 'zipf']  # 'weibull', 'dirichlet', 'pareto'
    funcs = getSamplerFun(sampler, distributions)
    funcs_np = getSamplerFun(np.random, distributions)
    params = getSamplerParams(funcs, distributions, shape)

    avgt_cu = repeatSamplerFuncs(funcs, params, repeats, 'pydpm')
    avgt_np = repeatSamplerFuncs(funcs_np, params, repeats, 'numpy')
    plotComparision(distributions, 'compare sampler speed with numpy', ['pydpm', 'numpy'], avgt_cu, avgt_np)

    ## compare with PyTorch and Tensorflow
    distributions = ['beta', 'exponential', 'gamma', 'laplace', 'normal', 'poisson']
    funcs_cu = getSamplerFun(sampler, distributions)
    params_cu = getSamplerParams(funcs_cu, distributions, shape)
    funcs_torch, params_torch = getTorchFunandParams(distributions, shape)
    funcs_tf, params_tf = getTensorflowFunandParams(distributions, shape)

    avgt_cu = repeatSamplerFuncs(funcs_cu, params_cu, repeats, 'pydpm')
    avgt_torch = repeatSamplerFuncs(funcs_torch, params_torch, repeats, 'torch')
    avgt_tf = repeatSamplerFuncs(funcs_tf, params_tf, repeats, 'tensorflow')
    # avgt_cu = [0.0008358, 0.0002895999, 0.00047686, 0.00042855, 0.00040248, 0.0002757]
    # avgt_torch = [0.024683303833, 0.00413030385, 0.009334292, 0.0006732845, 0.000860307, 0.0043436401]
    # avgt_tf = [0.008836011, 0.00104925394, 0.00438073, 0.00101441, 0.0008372998, 0.00203398]
    plotComparision(distributions, 'compare sampler speed with torch and tensorflow', ['pydpm', 'torch', 'tensorflow'], avgt_cu, avgt_torch, avgt_tf)


