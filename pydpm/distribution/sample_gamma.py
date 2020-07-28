"""
===========================================
Gamma Sampler implemented on GPU
===========================================

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

check by chaojie

import pydpm.distribution as DSG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
a = DSG.gamma(0.5, 2, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

start_time = time.time()
b = np.random.gamma(0.5, 2, 1000000)
end_time = time.time()
print('numpy takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(b), np.std(b)))

plt.figure()
plt.hist(a, bins=50, density=True)
plt.title('DSG.gamma')
plt.show()

plt.figure()
plt.hist(b, bins=50, density=True)
plt.title('numpy.random.gamma')
plt.show()

start_time = time.time()
a = DSG.standard_gamma(0.5, 1000000)
end_time = time.time()
print('DSG takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(a), np.std(a)))

start_time = time.time()
b = np.random.standard_gamma(0.5, 1000000)
end_time = time.time()
print('numpy takes {:<8.4f} second, mean: {:<8.4f}, std: {:<8.4f}'.format(end_time - start_time, np.mean(b), np.std(b)))

plt.figure()
plt.hist(a, bins=50, density=True)
plt.title('DSG.standard_gamma')
plt.show()

plt.figure()
plt.hist(b, bins=50, density=True)
plt.title('numpy.random.standard_gamma')
plt.show()

"""

import time

import numpy as np
from pydpm.distribution.pre_process import para_preprocess
from pycuda.compiler import SourceModule
import pycuda.autoinit
import pydpm.distribution.compat

Sampler = SourceModule("""
    #include <stdio.h>
    #include <time.h>
    __device__ int cudarand(long long seed)
    {
        if (seed == 0)
        {
            seed = 1;
        }
        long long temp=(48271 * seed + 0) % 2147483647;
        return temp;
    }

    __device__ int cudaseed(float seed ,long idx)
    {
        clock_t start = clock();
        int iseed = int(seed*2147483647)%2147483647;
        long long nseed = iseed*idx%2147483647;
        nseed = nseed*(abs(start+idx)%2069)%2147483647;
        long long temp=(48271 * nseed + 0) % 2147483647;
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

    __global__ void rand_Gamma(float* randomseed, float* target, int* matrix_scale, float* shape, float* scale)
    {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx<matrix_scale[0])
        {
            float d, x, v, u, p;
            int current_position=idx/matrix_scale[1];
            float beta = scale[current_position];
            float alpha = shape[current_position];
            if (alpha<1) {
                alpha += 1.0;
                p = shape[current_position];
            }
            else p = 1;

            long long seed = cudarand(randomseed[idx]*2147483647);
            //printf("%d",seed);
            float dd = alpha - (1.0 / 3.0);
            float cc = (1.0 / 3.0) / sqrt(dd);
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

            d = beta * dd * v;
            if (p >= 1)
                target[idx] = d;
            else
            {
                u = seed / 2147483647.0;
                target[idx] = float(d * pow(double(u), double(1.0 / p)));
            }
        }

    }
    """)


def gamma(shape, scale=1.0, times=1,device='cpu'):

    shape, scale, output, matrix_scale, randomseed, partition, single_number = para_preprocess(times, np.float32,
                                                                                               np.float32, shape, scale)
    func = Sampler.get_function('rand_Gamma')
    func(randomseed, output, matrix_scale, shape, scale, grid=(partition[0], 1, 1), block=(partition[1], 1, 1))

    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output


def standard_gamma(shape, size=1, device='cpu'):

    if np.array(shape).size > 1:
        raise Exception('parameter shape should be a scalar!!')

    shape = shape * np.ones(size)
    shape, scale, output, matrix_scale, randomseed, partition, single_number = para_preprocess(1, np.float32,
                                                                                               np.float32, shape, 1.0)
    func = Sampler.get_function('rand_Gamma')
    func(randomseed, output, matrix_scale, shape, scale,
         grid=(partition[0], 1, 1), block=(partition[1], 1, 1))
    if device == 'cpu':
        output = output.get()
    if single_number:
        return output[0]
    return output


