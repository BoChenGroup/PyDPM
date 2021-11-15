import os
import numpy as np
import ctypes

class distribution_sampler_cpu(object):

    def __init__(self):
        """
        The basic class for sampling distribution on cpu
        """
        super(distribution_sampler_cpu, self).__init__()

        # sampler for basic distributions
        setattr(self, 'standard_normal', np.random.standard_normal)
        setattr(self, 'normal', np.random.normal)
        setattr(self, 'standard_gamma', np.random.standard_gamma)
        setattr(self, 'gamma', np.random.gamma)
        setattr(self, 'standard_cauchy', np.random.standard_cauchy)
        # setattr(self, 'cauchy', np.random.cauchy) # numpy doesnot has this distribution
        setattr(self, 'chisquare', np.random.chisquare)
        setattr(self, 'beta', np.random.beta)
        # setattr(self, 'crt', np.random.crt) # numpy doesnot has this distribution
        setattr(self, 'dirichlet', np.random.dirichlet)
        setattr(self, 'poisson', np.random.poisson)
        setattr(self, 'weibull', np.random.weibull)
        setattr(self, 'negative_binomial', np.random.negative_binomial)
        setattr(self, 'lognormal', np.random.lognormal)
        setattr(self, 'binomial', np.random.binomial)
        setattr(self, 'multinomial', np.random.multinomial)
        setattr(self, 'laplace', np.random.laplace)
        setattr(self, 'logistic', np.random.logistic)
        setattr(self, 'exponential', np.random.exponential)
        setattr(self, 'standard_exponential', np.random.standard_exponential)
        setattr(self, 'noncentral_chisquare', np.random.noncentral_chisquare)
        setattr(self, 'zipf', np.random.zipf)
        setattr(self, 'triangular', np.random.triangular)
        setattr(self, 'noncentral_f', np.random.noncentral_f)
        setattr(self, '_f', np.random.f)
        # setattr(self, 't', np.random.t) # numpy doesnot has this distribution
        setattr(self, 'geometric', np.random.geometric)
        setattr(self, 'hypergeometric', np.random.hypergeometric)
        setattr(self, 'gumbel', np.random.gumbel)
        setattr(self, 'pareto', np.random.pareto)
        setattr(self, 'power', np.random.power)
        setattr(self, 'rayleigh', np.random.rayleigh)
