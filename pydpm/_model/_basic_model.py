
# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>; Wei Zhao <13279389260@163.com>
# License: BSD-3-Claus

class Params(object):
    def __init__(self):
        """
        The basic class for storing the parameters in the probabilistic model
        """
        super(Params, self).__init__()


class Basic_Model(object):
    def __init__(self, *args, **kwargs):
        """
        The basic model for all probabilistic models in this package
        Attributes:
            @public:
                global_params : [Params] the global parameters of the probabilistic model
                local_params  : [Params] the local parameters of the probabilistic model

            @private:
                _model_setting : [Params] the model settings of the probabilistic model
                _hyper_params  : [Params] the hyper parameters of the probabilistic model

        """
        super(Basic_Model, self).__init__()

        setattr(self, 'global_params', Params())
        setattr(self, 'local_params', Params())

        setattr(self, '_model_setting', Params())
        setattr(self, '_hyper_params', Params())

