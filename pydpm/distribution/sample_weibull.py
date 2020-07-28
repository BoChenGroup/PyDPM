"""
===========================================
Weibull Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

check by chaojie

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.weibull(0.5, 0.5, times=100000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))


"""



import numpy as np
from pycuda.compiler import SourceModule
from .pre_process import para_preprocess
import pycuda.autoinit
import pydpm.distribution.compat


Sampler = SourceModule("""
#include <stdio.h>

__device__ int cudarand(long long seed)
{
    if (seed == 0)
    {
        seed = 1;
    }
    long long temp=(48271 * seed + 0) % 2147483647;
    return temp;
}

__device__ float log_max(float x)
{
    return log(max(x, float(2.2e-10)));
}

__global__ void rand_Weibull(float *randomseed, float *target, int * matrix_scale,float *shape, float *scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<matrix_scale[0])
    {
        int current_index = idx/matrix_scale[1];
        float wei_scale = scale[current_index];
        float wei_shape = shape[current_index];
        float uniform = randomseed[idx];
        target[idx]=float(wei_scale * pow(double(-log_max(1 - uniform)), double(1.0 / wei_shape)));
    }
}

""")

def weibull(shape, scale, times=1,device='cpu'):

    shape, scale, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.float32,np.float32, shape, scale)
    func = Sampler.get_function('rand_Weibull')
    func(randomseed, output, matrix_scale, shape, scale,
         grid=(partition[0], 1, 1), block=(partition[1], 1, 1))
    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output
