import os
import numpy as np
import ctypes

class distribution_sampler_cpu(object):

    def __init__(self):
        """
        The basic class for sampling distribution on cpu
        """
        super(distribution_sampler_cpu, self).__init__()
