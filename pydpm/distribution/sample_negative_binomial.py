"""
===========================================
Negative_binomial Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.negative_binomial(100, 0.5, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

start_time = time.time()
b = np.random.negative_binomial(100, 0.5, 1000000)
end_time = time.time()
print('numpy takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(b), np.std(b)))

plt.figure()
plt.hist(a, bins=50, density=True)
plt.title('DSG.beta')
plt.show()

plt.figure()
plt.hist(b, bins=50, density=True)
plt.title('numpy.random.beta')
plt.show()



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

__global__ void rand_Negative_binomial(float *randomseed, int *target, int * matrix_scale,float *r, float *p)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<matrix_scale[0])
    {
        int seed = cudarand(randomseed[idx] * 2147483647.0);
        seed=cudarand(seed);
        int current_index = idx/matrix_scale[1];
        int suc = 0.0;
        float fail = 0.0;
        int total_r = r[current_index];
        float prob = p[current_index];
        while(total_r>fail)
        {
            seed = cudarand(seed);
            float temp = seed/2147483647.0;
            if (temp<prob)
            {
                suc++;
            }
            else
            {
                fail++;
            }
        }
        target[idx]=suc;
    }
}

""")

def negative_binomial(r, p, times=1,device='cpu'):

    if np.sum(r<1)>0:
        raise Exception('parameter r should be larger than 1!')

    if np.sum(p<=0 and p>1):
        raise Exception('parameter p should be less than 1!')

    r, p, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.float32, np.int32, r, p)

    func = Sampler.get_function('rand_Negative_binomial')
    func(randomseed, output, matrix_scale, r, p,
         grid=(partition[0], 1, 1), block=(partition[1], 1, 1))

    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output