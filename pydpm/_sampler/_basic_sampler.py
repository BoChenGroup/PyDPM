
# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>; Wei Zhao <13279389260@163.com>
# License: BSD-3-Claus

import platform
import ctypes
import os
import numpy as np

class Basic_Sampler(object):
    def __init__(self, device='cpu', seed=0, *args, **kwargs):
        """
        The basic sampler model for training all probabilistic models in this package
        Attributes:
            @public:


            @private:
                _model_setting : [Params] the model settings of the probabilistic model
                _hyper_params  : [Params] the hyper parameters of the probabilistic model

        """
        super(Basic_Sampler, self).__init__()

        assert device in ['cpu', 'gpu'], 'Device Error, device should be "cpu" or "gpu" '
        self.device = device
        self.seed = seed

        system_type = platform.system()
        assert system_type in ['Windows', 'Linux'], 'System Error, system should be "Windows" or "Linux" '
        self.system_type = system_type

        if self.device == 'cpu':
            self._cpu_sampler_initial()

        elif self.device == 'gpu':
            self._gpu_sampler_initial()

    def _cpu_sampler_initial(self):

        from ._distribution_sampler_cpu import distribution_sampler_cpu
        sampler = distribution_sampler_cpu()
        for distribution_name in dir(sampler):
            if distribution_name[0] != '_':
                setattr(self, distribution_name, getattr(sampler, distribution_name))
            else:
                continue


        from ._model_sampler_cpu import model_sampler_cpu
        sampler = model_sampler_cpu(self.system_type)
        for distribution_name in dir(sampler):
            if distribution_name[0] != '_':
                setattr(self, distribution_name, getattr(sampler, distribution_name))
            else:
                continue



    def _gpu_sampler_initial(self):

        from ._distribution_sampler_gpu import distribution_sampler_gpu
        sampler = distribution_sampler_gpu(self.system_type)
        for distribution_name in dir(sampler):
            if distribution_name[0] != '_':
                setattr(self, distribution_name, getattr(sampler, distribution_name))
            else:
                continue

        from ._model_sampler_gpu import model_sampler_gpu
        sampler = model_sampler_gpu(self.system_type)
        for distribution_name in dir(sampler):
            if distribution_name[0] != '_':
                setattr(self, distribution_name, getattr(sampler, distribution_name))
            else:
                continue



