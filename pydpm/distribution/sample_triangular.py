"""
===========================================
Triangular Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

check by chaojie

slow than numpy !!

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.triangular(0.5, 1, 1, times=100000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

start_time = time.time()
b = np.random.triangular(0.5,1, 1,100000)
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
from pydpm.distribution.pre_process import para_preprocess
from pycuda import gpuarray
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

__global__ void rand_Triangular(float* randomseed, float* target, int* matrix_scale, float* left, float* mid , float * right)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<matrix_scale[0])
    {
        int seed = cudarand(randomseed[idx] * 2147483647.0);
        int current_index=idx/matrix_scale[1];
        float l=left[current_index];
        float r=right[current_index];
        float m=mid[current_index];
        while (1)
        {
            seed=cudarand(seed);
            float randx = (seed/2147483647.0)*(r-l)+l;
            seed=cudarand(seed);
            float randy = (seed/2147483647.0)*(2/(r-l));
            float reject = randx<=m ? 2*(randx-l)/((r-l)*(m-l)) : 2*(r-randx)/((r-l)*(r-m));
            if (randy<=reject)
            {
                target[idx]=randx;
                return;
            }
        }
    }
}
""")

def triangular(left=0, mid=1, right=1 , times=1,device='cpu'):

    left, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.float32, np.float32, left)

    mid = gpuarray.to_gpu(np.array(mid, dtype=np.float32, order='C'))
    right = gpuarray.to_gpu(np.array(right, dtype=np.float32, order='C'))

    func = Sampler.get_function('rand_Triangular')
    func(randomseed, output, matrix_scale, left, mid, right, grid=(partition[0], 1, 1), block=(partition[1], 1, 1))

    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output