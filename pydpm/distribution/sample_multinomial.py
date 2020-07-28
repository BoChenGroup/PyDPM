"""
===========================================
Multinomial Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

check by chaojie

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.multinomial(10,[0.5, 0.2, 0.3], 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(np.sum(a,axis=0)), np.std(np.sum(a,axis=0))))

start_time = time.time()
b = np.random.multinomial(10, [0.5, 0.2, 0.3],1000000)
end_time = time.time()
print('numpy takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(b), np.std(b)))

start_time = time.time()
a = DSG.binomial(10,0.5, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(np.sum(a,axis=0)), np.std(np.sum(a,axis=0))))

start_time = time.time()
b = np.random.binomial(10, 0.5,1000000)
end_time = time.time()
print('numpy takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(b), np.std(b)))

"""

import numpy as np
from pycuda.compiler import SourceModule
from pycuda import gpuarray
from pydpm.distribution.pre_process import cuda_generator
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

__global__ void caculate_prob(float* target, int* matrix_scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<matrix_scale[2])
    {   
        int current_index=idx*matrix_scale[1];
        float sum=0.0;
        for (int i=0;i<matrix_scale[1];i++)
        {
            sum+=target[current_index+i];
            target[current_index+i]=sum;
        }   
        for (int i=0;i<matrix_scale[1];i++)
        {
            target[current_index+i]/=sum;
        }   
    }
}

__global__ void rand_Multinomial(float* randomseed, int* target, int* matrix_scale, float* p)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < matrix_scale[0])
    {
        float rand_prob = randomseed[idx];
        float sum = 0.0;
        int p_scale = matrix_scale[1];
        int current_index = int(idx/matrix_scale[3]);
        int prob_offset = (current_index%matrix_scale[2])*p_scale;
        int Embedding=p_scale-1;
        for (int k = 0; k < p_scale; k++)
        {
            float prob = p[prob_offset+k];
            if (rand_prob<=prob)
            {
                Embedding=k;
                atomicAdd(&target[current_index*p_scale + Embedding], 1);
                return;
            }
        }
    }
}
""")

def multinomial(count=1, prob=[0.5,0.5], times=1, device='cpu'):

    prob = np.array(prob, dtype=np.float32, order='C')
    prob_scale = prob.shape[-1]
    prob_num = int(prob.size/prob_scale)
    output_scale = (times,) + prob.shape
    prob = gpuarray.to_gpu(prob)
    output = gpuarray.empty(output_scale, dtype=np.int32)

    nx = int(output.size/prob_scale)*count
    matrix_scale = gpuarray.to_gpu(np.array([nx, prob_scale, prob_num, count], dtype=np.int32))
    randomseed = cuda_generator.gen_uniform([nx], dtype=np.float32)

    block_x = int(500)
    grid_x = int(np.floor(nx / block_x) + 1)
    func = Sampler.get_function('caculate_prob')
    func(prob, matrix_scale, grid=(int(prob_num/500)+1, 1, 1), block=(500, 1, 1))
    func = Sampler.get_function('rand_Multinomial')
    func(randomseed, output, matrix_scale, prob, grid=(grid_x, 1, 1), block=(block_x, 1, 1))

    if device == 'cpu':
        output = output.get()
    return output

def binomial(count=1, prob=0.5, times=1, device='cpu'):

    prob = np.array(prob, dtype=np.float32, order='C')

    if len(prob.shape) == 0 or prob.shape[-1]!=1:
        prob = np.expand_dims(prob, axis=-1)

    if prob.shape[-1] == 1:

        del_prob = 1 - prob
        prob_all = np.concatenate([prob, del_prob], axis=-1)

    output = multinomial(count, prob_all, times,device)

    return output

