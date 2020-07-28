"""
===========================================
Dirichlet Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

check by chaojie

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.dirichlet([0.2, 0.3, 0.1, 0.2, 0.2], 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(np.sum(a,axis=-2)), np.std(np.sum(a,axis=-2))))


start_time = time.time()
b = np.random.dirichlet([0.2, 0.3, 0.1, 0.2, 0.2], 1000000)
end_time = time.time()
print('numpy takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(np.sum(b,axis=-1)), np.std(np.sum(b,axis=-1))))

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

__device__ float single_normal(float rand1, float rand2)
{
    float U, V;
    float z;

    U = rand1 / 2147483647.0;
    V = rand2 / 2147483647.0;
    z = sqrt(-2.0 * log(U))* sin(2.0 * 3.141592654 * V);
    return z;
}

__global__ void rand_Gamma(float *randomseed, float *target, int *matrix_scale,float *shape, float *scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<matrix_scale[0])
    {
        float d, x, v, u, p;
        float cc;
        int current_position=idx/matrix_scale[1];
        float beta = scale[current_position];
        float alpha = shape[current_position];
        if (alpha<1) {
            alpha += 1.0;
            p = shape[current_position];
        }
        else p = 1;
        int seed = cudarand(randomseed[idx] * 2147483647.0);
        float dd = alpha - (1.0 / 3.0);
        cc = (1.0 / 3.0) / sqrt(dd);
        for (;;)
        {
            do
            {
                float r1 = seed;
                float r2 = cudarand(seed);
                seed = cudarand(r2);
                x = single_normal(r1, r2);
                v = 1.0 + cc * x;
            } while (v <= 0);
            v = v * v*v;
            u = seed / 2147483647.0;
            seed = cudarand(seed);
            if (u < 1 - 0.0331 *x*x*x*x)
                break;
            if (log(u) < 0.5 * x * x + dd * (1 - v + log(v)))
                break;
        }
        d = 1 / beta * dd * v;
        if (p >= 1)
            target[idx] = d;
        else
        {
            u = seed / 2147483647.0;
            target[idx] = float(d * pow(double(u), double(1.0 / p)));
        }
    }

}

__global__ void normalize(float* target, int* matrix_scale , int* normalize_scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<normalize_scale[2])
    {
        float sum=0.0;
        int offset_alpha=int(idx/matrix_scale[1]);
        int offset_times=idx%matrix_scale[1];
        int offset=offset_alpha*normalize_scale[1]+offset_times;
        for (int i=0;i<normalize_scale[0];i++)
        {
            sum+=target[offset+i*matrix_scale[1]];
        }
        for (int i=0;i<normalize_scale[0];i++)
        {
            target[offset+i*matrix_scale[1]]=target[offset+i*matrix_scale[1]]/sum;
        }
    }
}

""")

def dirichlet(a, times=1):

    a=np.array(a)
    normalize_scale=[a.shape[-1],a.shape[-1]*times,int(a.size/a.shape[-1])*times]
    normal_block=int(500)
    normal_grid=int(normalize_scale[2]/normal_block+1)
    normalize_scale=gpuarray.to_gpu(np.array(normalize_scale,dtype=np.int32))
    a, b, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.float32, np.float32, a, 1.0)

    func = Sampler.get_function('rand_Gamma')
    func(randomseed, output, matrix_scale, a, b, grid=(partition[0], 1, 1), block=(partition[1], 1, 1))

    func = Sampler.get_function('normalize')
    func(output, matrix_scale,normalize_scale, grid=(normal_grid, 1, 1), block=(normal_block, 1, 1))

    output = output.get()
    return output