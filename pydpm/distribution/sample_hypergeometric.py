"""
===========================================
Hypergeometric Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

need to be checked



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

__global__ void rand_Hypergeometric(float* randomseed, int* target, int* matrix_scale, float* prob , int * ns)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<matrix_scale[0])
    {
        int current_index=idx/matrix_scale[1];
        float p=prob[current_index];  
        int n_sample=ns[current_index];
        int rnd=0; 
        int seed=randomseed[idx]*2147483647.0;
        for (int i=0;i<n_sample;i++)
        {
            seed=cudarand(seed);
            if (seed/2147483647.0<p)
                rnd++;
        }
        target[idx]=rnd;
    }
}


""")

def hypergeometric(ng, nb, ns, times=1,device='cpu'):

    ng, nb, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.float32, np.int32, ng,nb)
    ns = gpuarray.to_gpu(np.array(ns, dtype=np.int32, order='C'))
    prob = ng /(ng+nb)
    func = Sampler.get_function('rand_Hypergeometric')
    func(randomseed, output, matrix_scale, prob, ns, grid=(partition[0], 1, 1), block=(partition[1], 1, 1))

    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output


