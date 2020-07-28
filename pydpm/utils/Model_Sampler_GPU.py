"""
===========================================
Model Sampler implemented on GPU
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
realmin = 2.2e-10
import pydpm.distribution.compat

# ====================== CUDA Initial ======================#

mod = SourceModule("""
#include <stdio.h>

__device__ int cudarand(long long seed)
{
    long long temp=(48271 * seed + 0) % 2147483647;
    return temp;
}

__global__ void Multi_Sampler(float* randomseed, int* Para, int* X_value, int* X_rows, int* X_cols, float* Phi, float* Theta, float* XVK, float* XKJ)    //
{
    const int V = Para[0];
    const int K = Para[1];
    const int J = Para[2];
    const int N = Para[3];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seed = randomseed[idx]*2147483647.0;

    if (idx < N)
    {
        float cumsum = 0.0;
        float sum=0.0;
        int v = X_rows[idx];
        int j = X_cols[idx];

        for (int k = 0; k < K; k++)
        {
            sum += Phi[v*K + k] * Theta[k*J + j];
        }
        
        for (int token = 0; token<X_value[idx]; token++)
        {
            int Embedding_K=K-1;
            float sumprob=0.0;
            seed = cudarand(seed);
            float probrnd = ((float)(seed) / 2147483647.0) * sum;

            for (int k = 0; k < K; k++)
            {
                cumsum = Phi[v*K + k] * Theta[k*J + j];
                if (sumprob+cumsum>=probrnd)
                {
                    Embedding_K=k;
                    break;
                }
                sumprob+=cumsum;
            }
            
            atomicAdd(&XVK[v*K + Embedding_K], 1);
            atomicAdd(&XKJ[Embedding_K*J + j], 1);
        }
    }
}

__global__ void Conv_Multi_Sampler(int* para, int *row_index, int *column_index, int *page_index, float *value_index, float *Params_W1_nk1, float *Params_D1_k1, float *Params_W1_nk1_Aug, float *Params_D1_k1_Aug)
{
    int K1         = para[0];
    int K1_K1      = para[1];
    int K1_K2      = para[2];
    int K1_K3      = para[3];
    int K1_K4      = para[4];
    int word_total = para[5];

    int ix = blockDim.x * blockIdx.x + threadIdx.x; 
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int idx = iy* blockDim.x *gridDim.x+ ix;

    if ((idx < word_total))
    {
        int v1 = row_index[idx];                 // row_index
        int v2 = column_index[idx];              // col_index
        int n  = page_index[idx];                // file_index
        float value = value_index[idx];

        int word_k1_min = 0;
        int word_k1_max = 0;
        int word_k2_min = 0;
        int word_k2_max = 0;

        // word_k1
        if ((v1 - K1_K3 + 1) > 0)
            word_k1_min = v1 - K1_K3 + 1;
        else
            word_k1_min = 0;

        if (v1 > K1_K1 -1)
            word_k1_max = K1_K1 -1;
        else
            word_k1_max = v1;

        int l_word_k1 = word_k1_max - word_k1_min + 1;
        int *word_k1  = new int[l_word_k1];
        for (int i = 0; i < (l_word_k1); i++)
            word_k1[i] = word_k1_min + i;

        // word_k2
        if ((v2 - K1_K4 + 1) > 0)
            word_k2_min = v2 - K1_K4 + 1;
        else
            word_k2_min = 0;

        if (v2 > K1_K2 -1)
            word_k2_max = K1_K2 -1;
        else
            word_k2_max = v2;

        int l_word_k2 = word_k2_max - word_k2_min + 1;
        int *word_k2  = new int[l_word_k2];
        for (int i = 0; i < (l_word_k2); i++)
            word_k2[i] = word_k2_min + i;

        // word_k3
        int *word_k3 = new int[l_word_k1];
        for (int i = 0; i < (l_word_k1); i++)
            word_k3[i] = v1 - word_k1[i] ;

        // word_k4
        int *word_k4 = new int[l_word_k2];
        for (int i = 0; i < (l_word_k2); i++)
            word_k4[i] = v2 - word_k2[i] ;

        float MultRate_sum = 0;
        //word_aug_stack
        //MultRate_stack
        //Params_W1_nk1
        //Params_D1_k1
        int stack_start = idx * K1_K4 * K1;

        for (int i = 0; i < K1; i++)
        {
            for (int k = 0; k < (l_word_k1); k++)
            {
                for (int j = 0; j < (l_word_k2); j++)
                {
                    int temp_a = (n) * K1 * K1_K1 * K1_K2 + (i) * K1_K1 * K1_K2 + word_k1[k] * K1_K2 + (word_k2[j]);
                    int temp_b = (i) * K1_K3 * K1_K4 + word_k3[k] * K1_K4 + (word_k4[j]);
                    MultRate_sum = MultRate_sum + Params_W1_nk1[temp_a] * Params_D1_k1[temp_b];
                }
            }
        }

        for (int i = 0; i < K1; i++)
        {
            for (int k = 0; k < (l_word_k1); k++)
            {
                for (int j = 0; j < (l_word_k2); j++)
                {
                    int temp_a = (n) * K1 * K1_K1 * K1_K2 + (i) * K1_K1 * K1_K2 + word_k1[k] * K1_K2 + (word_k2[j]);
                    int temp_b = (i) * K1_K3 * K1_K4 + word_k3[k] * K1_K4 + (word_k4[j]);
                    float point=0;
                    if (MultRate_sum == 0)
                    {
                        point = (1 / (l_word_k1 * l_word_k2)) * value;
                    }
                    else
                    {
                        point = (Params_W1_nk1[temp_a] * Params_D1_k1[temp_b] / MultRate_sum) * value;
                    }

                    atomicAdd(&Params_W1_nk1_Aug[temp_a], point);
                    atomicAdd(&Params_D1_k1_Aug[temp_b], point);
                }
            }
        }

        delete[] word_k1;
        delete[] word_k2;
        delete[] word_k3;
        delete[] word_k4; 
    }

}

__global__ void Crt_Sampler(float* randomseed, int* scale, int* Xt_to_t1, float* p,  float* Xt1)
{
    const int N = scale[0];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seed = randomseed[idx] * 2147483647;
    
    if (idx < N && Xt_to_t1[idx] >= 0.5)
    {
        int token,table;
        float sum = 0.0;
        sum = p[idx];
        
        for ( token = 1, table = 1; token<Xt_to_t1[idx]; token++)
        {
            seed = cudarand(seed);
            float probrnd = ((float)(seed) / 2147483647.0);
            if (probrnd <= sum / (sum + token))
                table++;
        }
        
        atomicAdd(&Xt1[idx], table);
    }
}

__global__ void Crt_Multi_Sampler(float *randomseed, int* Para, int* Xt_to_t1, float* Phi_t1, float* Theta_t1, float* Xt1_VK, float* Xt1_KJ)
{

    const int V = Para[0];
    const int K = Para[1];
    const int J = Para[2];
    const int N = Para[3];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seed = randomseed[idx] * 2147483647;
    
    if (idx < N && Xt_to_t1[idx] >= 0.5 )
    {
        float sum = 0.0;
        float cumsum = 0.0;
        int token, table;
        int v = idx / J;   // row first
        int j = idx % J;   // row first
        
        for (int k = 0; k<K; k++)
        {
            sum += Phi_t1[v*K + k] * Theta_t1[k*J + j]; // C index is different of Matlab
        }
        
        for (token = 1, table = 1; token<Xt_to_t1[idx]; token++)
        {
            seed = cudarand(seed);
            float probrnd = ((float)(seed) / 2147483647.0);
            if (probrnd <= sum / (sum + token))
                table++;
        }
        
        for (token = 0; token<table; token++)
        {
            int Embedding_K = K - 1;
            float sumprob = 0.0;
            seed = cudarand(seed);
            float probrnd = ((float)(seed) / 2147483647.0) * sum;

            for (int k = 0; k < K; k++)
            {
                cumsum = Phi_t1[v*K + k] * Theta_t1[k*J + j];
                if (sumprob + cumsum >= probrnd)
                {
                    Embedding_K = k;
                    break;
                }
                sumprob += cumsum;
            }
            
            atomicAdd(&Xt1_VK[v*K + Embedding_K], 1);
            atomicAdd(&Xt1_KJ[Embedding_K*J + j], 1);
        }
    }
}

__device__ float rand_normal(float rand1,float rand2)
{
    float U, V;
    //static int phase = 0;
    float z;
    
    //if (phase == 0)
    //{
        U = rand1 / 2147483647.0;
        V = rand2 / 2147483647.0;
        z = sqrt(-2.0 * log(U))* sin(2.0 * 3.141592654 * V);
    /*}
    else
    {
        z = sqrt(-2.0 * log(U)) * cos(2.0 * 3.141592654 * V);
    }
    
    phase = 1 - phase;*/
    return z;
}

__global__ void rand_gamma(float *randomseed, float *target, float *shape, float *scale, int *nx) 
{
    float d, x, v, u,p;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float beta = scale[0];
    float alpha=shape[idx];
    float cc;
    if(idx<nx[0])
    {
        if (alpha<1) {
            alpha += 1.0;
            p = shape[idx];
        }
        else p = 1;
        float seed = cudarand(randomseed[idx]*2147483647.0);
        float dd=alpha-(1.0 / 3.0);
        cc = (1.0 / 3.0) / sqrt(dd);
        for (;;) {
            do {
                float r1=seed;
                float r2=cudarand(seed);
                seed=cudarand(r2);
                x = rand_normal(r1,r2);
                v = 1.0 + cc * x;
            } while (v <= 0);
            v = v * v*v;
            u = seed/2147483647.0;
            seed = cudarand(seed);
            if (u < 1 - 0.0331 *x*x*x*x)
                break;
            if (log(u) < 0.5 * x * x + dd * (1 - v + log(v)))
                break;
        }
        d = 1 / beta * dd * v;
        if (p >= 1)
            target[idx]=d;
        else {
            u = seed/2147483647.0;
            target[idx]=float(d * pow(double(u), double(1.0 / p)));
        }
    }
}

""")


def Multrnd_Matrix_GPU(X_t, Phi_t, Theta_t, dtype='dense'):

    if dtype=='dense':

        [V, J] = X_t.shape
        K = Theta_t.shape[0]
        [X_t_rows, X_t_cols] = np.where(X_t > 0.5)
        X_t_values = X_t[(X_t_rows, X_t_cols)]
        N = len(X_t_values)  # number of sample point
        Para = np.array([V, K, J, N], dtype=np.int32)

    X_t_values = np.array(X_t_values, dtype=np.int32)
    X_t_rows = np.array(X_t_rows, dtype=np.int32)
    X_t_cols = np.array(X_t_cols, dtype=np.int32)

    Xt_to_t1_t = np.zeros([K, J], dtype=np.float32, order='C')
    WSZS_t = np.zeros([V, K], dtype=np.float32, order='C')
    Phi_t = np.array(Phi_t, dtype=np.float32, order='C')
    Theta_t = np.array(Theta_t, dtype=np.float32, order='C')

    if N != 0:

        block_x = int(400)
        grid_x = int(np.floor(N / block_x) + 1)
        # print("block: {:<8.4f}, grid:{:<8.4f}".format(block_x, grid_x))

        randomseed = np.random.rand(N)
        randomseed = np.array(randomseed, dtype=np.float32, order='C')
        func = mod.get_function('Multi_Sampler')
        func(drv.In(randomseed), drv.In(Para), drv.In(X_t_values), drv.In(X_t_rows), drv.In(X_t_cols), drv.In(Phi_t),
             drv.In(Theta_t), drv.InOut(WSZS_t), drv.InOut(Xt_to_t1_t),
             grid=(grid_x, 1, 1), block=(block_x, 1, 1))

    return Xt_to_t1_t, WSZS_t

def Crt_Matrix_GPU(Xt_to_t1_t, p, dtype='dense'):

    if dtype=='dense':

        [K_t, J] = Xt_to_t1_t.shape
        N = K_t*J
        N = np.array(N, dtype=np.int32, order='C')

    Xt_to_t1_t = np.array(Xt_to_t1_t, dtype=np.int32, order='C')
    p = np.array(p, dtype=np.float32, order='C')
    X_t1 = np.zeros([K_t, J], dtype=np.float32, order='C')

    if N != 0:

        block_x = int(400)
        grid_x = int(np.floor(N / block_x) + 1)

        randomseed = np.random.rand(N)
        randomseed = np.array(randomseed, dtype=np.float32, order='C')
        func=mod.get_function('Crt_Sampler')
        func(drv.In(randomseed), drv.In(N), drv.In(Xt_to_t1_t), drv.In(p),  drv.InOut(X_t1),
             grid=(grid_x, 1, 1), block=(block_x, 1, 1))

    return X_t1

def Crt_Multirnd_Matrix_GPU(Xt_to_t1_t, Phi_t1, Theta_t1, dtype='dense'):

    if dtype=='dense':

        [K_t, J] = Xt_to_t1_t.shape
        K_t1 = Theta_t1.shape[0]
        N = K_t*J
        Para = np.array([K_t, K_t1, J, N], dtype=np.int32)


    Xt_to_t1_t = np.array(Xt_to_t1_t, dtype=np.int32, order='C')
    Xt_to_t1_t1 = np.zeros([K_t1, J], dtype=np.float32, order='C')
    WSZS_t1 = np.zeros([K_t, K_t1], dtype=np.float32, order='C')
    Phi_t1 = np.array(Phi_t1, dtype=np.float32, order='C')
    Theta_t1 = np.array(Theta_t1, dtype=np.float32, order='C')

    if N!=0:

        block_x = int(400)
        grid_x = int(np.floor(N / block_x) + 1)

        randomseed = np.random.rand(N)
        randomseed = np.array(randomseed, dtype=np.float32, order='C')

        func = mod.get_function('Crt_Multi_Sampler')
        func(drv.In(randomseed), drv.In(Para), drv.In(Xt_to_t1_t), drv.In(Phi_t1),
             drv.In(Theta_t1), drv.InOut(WSZS_t1), drv.InOut(Xt_to_t1_t1),
             grid=(grid_x, 1), block=(block_x, 1, 1))

    return Xt_to_t1_t1, WSZS_t1


def Sample_Phi(WSZS_t, Eta_t):

    Kt = WSZS_t.shape[0]
    Kt1 = WSZS_t.shape[1]
    Phi_t_shape = WSZS_t + Eta_t

    func = mod.get_function('rand_gamma')
    nx = Kt * Kt1
    Phi_t = np.zeros([Kt, Kt1], dtype=np.float32, order='C')
    Phi_t_shape = np.array(Phi_t_shape, dtype=np.float32, order='C')
    randomseed = np.random.rand(Kt * Kt1)
    func(drv.In(randomseed), drv.Out(Phi_t), drv.In(Phi_t_shape), drv.In(np.array([1], dtype=np.float32)),
         drv.In(np.array([nx], dtype=np.int32)),
         grid=(Kt, 1), block=(Kt1, 1, 1))
    Phi_t = Phi_t / Phi_t.sum(0)
    # print("{:.8f}".format(time.time()-timex))
    return Phi_t


def Sample_Theta(Xt_to_t1_t, c_j_t1, p_j_t, shape):

    Kt = Xt_to_t1_t.shape[0]
    N = Xt_to_t1_t.shape[1]
    nx = Kt * N
    Theta_t = np.zeros([Kt, N], dtype=np.float32, order='C')
    Theta_t_shape = Xt_to_t1_t + shape

    func = mod.get_function('rand_gamma')
    Theta_t_shape = np.array(Theta_t_shape, dtype=np.float32, order='C')
    randomseed = np.random.rand(nx)
    func(drv.In(randomseed), drv.InOut(Theta_t), drv.In(Theta_t_shape), drv.In(np.array([1], dtype=np.float32)),
         drv.In(np.array([nx], dtype=np.int32)),
         grid=(N, 1), block=(Kt, 1, 1))
    Theta_t[:, :] = Theta_t / (c_j_t1[0, :] - np.log(np.maximum(realmin, 1 - p_j_t[0, :])))
    return Theta_t

def conv_multi_sample(file, row, col, val, w1_nk1, d1_k1, Setting):

    X_rows = np.array(row, dtype='int32',order='C')
    X_cols = np.array(col, dtype='int32',order='C') + 1  # padding!!
    X_file_index = np.array(file, dtype='int32',order='C')
    X_value = np.array(val, dtype='float32',order='C')
    word_total = len(X_rows)

    block_x = 500
    grid_x = 128
    grid_y = word_total / (block_x * grid_x) + 1

    Batch_Para = np.array([Setting['K1'], Setting['K1_S1'], Setting['K1_S2'], Setting['K1_S3'], Setting['K1_S4'], word_total],
        dtype=np.int32, order='C')

    W1_nk1 = np.array(w1_nk1, dtype='float32', order='C')
    D1_k1 = np.array(d1_k1, dtype='float32', order='C')
    W1_nk1_Aug = np.zeros(W1_nk1.shape, dtype='float32', order='C')
    D1_k1_Aug = np.zeros(D1_k1.shape, dtype='float32', order='C')
    fuc = mod.get_function('Conv_Multi_Sampler')
    fuc(drv.In(Batch_Para), drv.In(X_rows), drv.In(X_cols), drv.In(X_file_index), drv.In(X_value), drv.In(W1_nk1), drv.In(D1_k1),
    drv.InOut(W1_nk1_Aug), drv.InOut(D1_k1_Aug),
    grid=(int(grid_x), int(grid_y), 1), block=(int(block_x), 1, 1))

    return W1_nk1_Aug, D1_k1_Aug



