"""
===========================================
Compatible with torch and tensorflow
===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu <wjw19960807@163.com>
# Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>

import warnings

try:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.Variable(1)

except:
    try:
        from torch.cuda import FloatTensor
        x = FloatTensor(1)
    except:
        warnings.warn("not find torch or tensorflow packages,DSG may be error after running a torch or tensorflow code")