"""
===========================================
Model Sampler implemented on GPU
===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu
# License: BSD-3-Clause

import pycuda.curandom as curandom
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda.curandom import XORWOWRandomNumberGenerator

import numpy as np
realmin = 2.2e-10

cuda_generator = XORWOWRandomNumberGenerator()

mod = SourceModule("""

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

#include <stdio.h>

__global__ void _conv_multi_augmentation(float* randomseed, int* para, int* row_index, int* col_index, int* n_index, float* value_index, float* W1_nk1, float* D1_k1, float* W1_nk1_Aug, float* D1_k1_Aug)
{   
    int K0         = para[0];
    int K1         = para[1];
    int K1_S1      = para[2];
    int K1_S2      = para[3];
    int K1_S3      = para[4];
    int K1_S4      = para[5];
    int word_total = para[6];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ((idx < word_total))
    {
        int v1 = row_index[idx];                 // row_index
        int v2 = col_index[idx];                 // col_index
        int n  = n_index[idx];                   // file_index
        float value = value_index[idx];          
        int seed = randomseed[idx] * 2147483647;

        int word_k1_min = 0;
        int word_k1_max = 0;
        int word_k2_min = 0;
        int word_k2_max = 0;

        // word_k1
        if ((v1 - K1_S3 + 1) > 0)
            word_k1_min = v1 - K1_S3 + 1;
        else
            word_k1_min = 0;

        if (v1 > K1_S1 -1)
            word_k1_max = K1_S1 -1;
        else
            word_k1_max = v1;

        int l_word_k1 = word_k1_max - word_k1_min + 1;

        // word_k2
        if ((v2 - K1_S4 + 1) > 0)
            word_k2_min = v2 - K1_S4 + 1;
        else
            word_k2_min = 0;

        if (v2 > K1_S2 -1)
            word_k2_max = K1_S2 -1;
        else
            word_k2_max = v2;

        int l_word_k2 = word_k2_max - word_k2_min + 1;

        // N*K0*K1_V1*K1_V2 => N*K1*K1_S1*K1_S2, K0*K1*K1_S3*K1_S4

        float MultRate_sum = 0;

        for (int i = 0; i < K1; i++)
        {
            for (int k = 0; k < (l_word_k1); k++)
            {
                for (int j = 0; j < (l_word_k2); j++)
                {
                    int word_k1 = word_k1_min + k;
                    int word_k2 = word_k2_min + j;
                    int temp_a = (n) * K1 * K1_S1 * K1_S2 + (i) * K1_S1 * K1_S2 + word_k1 * K1_S2 + (word_k2);
                    int temp_b = (i) * K1_S3 * K1_S4 + (v1 - word_k1) * K1_S4 + (v2 - word_k2);

                    MultRate_sum = MultRate_sum + W1_nk1[temp_a] * D1_k1[temp_b];
                }
            }
        }

        if (MultRate_sum == 0) 
        {
            return;
        }

        for (int token = 0; token<value; token++)
        {
            float cumsum=0.0;
            seed = cudarand(seed);
            float probrnd = ((float)(seed) / 2147483647.0) * MultRate_sum;
            int flag=0;

            for (int i = 0; i < K1; i++)
            {
                for (int k = 0; k < (l_word_k1); k++)
                {
                    for (int j = 0; j < (l_word_k2); j++)
                    {
                        int word_k1 = word_k1_min + k;
                        int word_k2 = word_k2_min + j;
                        int temp_a = (n) * K1 * K1_S1 * K1_S2 + (i) * K1_S1 * K1_S2 + word_k1 * K1_S2 + (word_k2);
                        int temp_b = (i) * K1_S3 * K1_S4 + (v1 - word_k1) * K1_S4 + (v2 - word_k2);

                        float prob = W1_nk1[temp_a] * D1_k1[temp_b];
                        cumsum += prob;
                        if (cumsum>=probrnd)
                        {
                             atomicAdd(&W1_nk1_Aug[temp_a], 1.0);
                             atomicAdd(&D1_k1_Aug[temp_b], 1.0);
                             flag = 1;
                             break;        
                        }
                    }

                    if (flag==1) break;
                }

                if (flag==1) break;
            }
        }

    }
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

            for (int k = 0; k < K; k++)//data
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

__global__ void Crt_Conv_Multi_Sampler(float* randomseed, int* para, int* n_index, int* k1_index, int* row_index, int* col_index, float* value_index, float* D2_k2, float* W2_nk2, float* Phi_Theta, float* D2_k2_Aug, float* W2_nk2_Aug)
{
    int N = para[0];
    int K1 = para[1];
    int K1_S1 = para[2];
    int K1_S2 = para[3];
    int K2 = para[4];
    int K2_S1 = para[5];
    int K2_S2 = para[6];
    int K2_S3 = para[7];
    int K2_S4 = para[8];
    int word_total = para[9];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ((idx < word_total))
    {
        int n = n_index[idx];
        int k1 = k1_index[idx];
        int row = row_index[idx];
        int col = col_index[idx];
        float value = value_index[idx];

        // Crt: N*K1*K1_S1*K1_S2

        int seed = randomseed[idx] * 2147483647;
        int temp_a = n*K1*K1_S1*K1_S2 + k1*K1_S1*K1_S2 + row*K1_S1 + col;
        float sum = Phi_Theta[temp_a]; // Phi_Theta: N*K1*K1_S1*K1_S2
        int table = 0;
        int token = 0;

        if (value<0.5)
        {
            table = 0;
            return;
        }
        else
        {
            for ( token = 1, table = 1; token<value; token++)
            {
                seed = cudarand(seed);
                float probrnd = ((float)(seed) / 2147483647.0);
                if (probrnd <= sum / (sum + token))
                    table++;
            }
        }

        //W1_nk1: N*K1*K1_S1*K1_S2 => W2_nk2: N*K2*K2_S1*K2_S2  D2_k2: K1*K2*K2_S3*K2_S4

        int word_k1_min = 0;
        int word_k1_max = 0;
        int word_k2_min = 0;
        int word_k2_max = 0;

        // word_k1
        if ((row - K2_S3 + 1) > 0)
            word_k1_min = row - K2_S3 + 1;
        else
            word_k1_min = 0;

        if (row > K2_S1 -1)
            word_k1_max = K2_S1 -1;
        else
            word_k1_max = row;

        int l_word_k1 = word_k1_max - word_k1_min + 1;

        // word_k2
        if ((col - K2_S4 + 1) > 0)
            word_k2_min = col - K2_S4 + 1;
        else
            word_k2_min = 0;

        if (col > K2_S2 -1)
            word_k2_max = K2_S2 -1;
        else
            word_k2_max = col;

        int l_word_k2 = word_k2_max - word_k2_min + 1;

        float MultRate_sum = 0;

        for (int i = 0; i < K2; i++)
        {
            for (int k = 0; k < (l_word_k1); k++)
            {
                for (int j = 0; j < (l_word_k2); j++)
                {
                    int word_k1 = word_k1_min + k;
                    int word_k2 = word_k2_min + j;
                    int temp_a = (n) * K2 * K2_S1 * K2_S2 + (i) * K2_S1 * K2_S2 + word_k1 * K2_S2 + (word_k2);
                    int temp_b = k1 * K2 * K2_S3 * K2_S4 +  (i) * K2_S3 * K2_S4 + (row - word_k1) * K2_S4 + (col - word_k2);
                    MultRate_sum = MultRate_sum + W2_nk2[temp_a] * D2_k2[temp_b];
                }
            }
        }

        if (MultRate_sum == 0) 
        {
            return;
        }


        for (int token = 0; token<table; token++)
        {
            float cumsum=0.0;
            seed = cudarand(seed);
            float probrnd = ((float)(seed) / 2147483647.0) * MultRate_sum;
            int flag=0;

            for (int i = 0; i < K2; i++)
            {
                for (int k = 0; k < (l_word_k1); k++)
                {
                    for (int j = 0; j < (l_word_k2); j++)
                    {
                        int word_k1 = word_k1_min + k;
                        int word_k2 = word_k2_min + j;
                        int temp_a = (n) * K2 * K2_S1 * K2_S2 + (i) * K2_S1 * K2_S2 + word_k1 * K2_S2 + (word_k2);
                        int temp_b = k1 * K2 * K2_S3 * K2_S4 +  (i) * K2_S3 * K2_S4 + (row - word_k1) * K2_S4 + (col - word_k2);

                        float prob = W2_nk2[temp_a] * D2_k2[temp_b];
                        cumsum += prob;
                        if (cumsum>=probrnd)
                        {
                             atomicAdd(&W2_nk2_Aug[temp_a], 1.0);
                             atomicAdd(&D2_k2_Aug[temp_b], 1.0);
                             flag = 1;
                             break;        
                        }
                    }

                    if (flag==1) break;
                }

                if (flag==1) break;
            }
        }

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

__global__ void Sum_Pooling(int* para, float* M_nk, float* M_nk_pool, float* M_nk_w)
{
    int N = para[0];
    int K1 = para[1];
    int K1_S1 = para[2];
    int K1_S2 = para[3];
    int word_total = para[4];
    int stride = para[5];
    int K1_S2_pool = para[6];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ((idx < word_total))
    {   
        int row_index = idx / K1_S2_pool;
        int remain = idx - row_index*K1_S2_pool;

        int idx_stride = stride;
        if (remain == K1_S2_pool-1)
        {
            idx_stride = K1_S2 - (K1_S2_pool-1)*stride;
        }

        //printf("%d, %d, %d", row_index, remain, idx_stride);

        for (int i=0; i<idx_stride; i++)
        {
            int temp_a = row_index*K1_S2 + remain*stride + i;
            atomicAdd(&M_nk_pool[idx], M_nk[temp_a] + 0.0000001); 
        }

        for (int i=0; i<idx_stride; i++)
        {            
            int temp_a = row_index*K1_S2 + remain*stride + i;
            float rate = (M_nk[temp_a] + 0.0000001) / (M_nk_pool[idx]);
            atomicAdd(&M_nk_w[temp_a], rate); 
        }
    }
}

__global__ void Up_Pooling(int* para, float* M_nk, float* M_nk_pool, float* M_nk_w)
{
    int N = para[0];
    int K1 = para[1];
    int K1_S1 = para[2];
    int K1_S2 = para[3];
    int word_total = para[4];
    int stride = para[5];
    int K1_S2_pool = para[6];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ((idx < word_total))
    {   
        int row_index = idx / K1_S2_pool;
        int remain = idx - row_index*K1_S2_pool;

        int idx_stride = stride;
        if (remain == K1_S2_pool-1)
        {
            idx_stride = K1_S2 - (K1_S2_pool-1)*stride;
        }

        for (int i=0; i<idx_stride; i++)
        {
            int temp_a = row_index*K1_S2 + remain*stride + i;
            float rate = M_nk_pool[idx] * M_nk_w[temp_a];
            if (rate <= 0.0000001)
            {
                rate = 0.0000001;
            }

/*            if (M_nk_pool[idx] == 0)
            {
                printf("value error");
            }*/
            atomicAdd(&M_nk[temp_a], rate); 
        }
    }
}
 """)

class model_sampler_gpu(object):

    def __init__(self, system_type='Windows', seed=0):
        """
        The basic class for sampling distribution on cpu
        """
        super(model_sampler_gpu, self).__init__()

        self.system_type = system_type
        self.seed = seed

    def multi_aug(self, X_t, Phi_t, Theta_t, dtype='dense'):

        if dtype=='dense':

            [V, J] = X_t.shape  # 词表，文档，主题
            K = Theta_t.shape[0]
            [X_t_rows, X_t_cols] = np.where(X_t > 0.5)
            X_t_values = X_t[(X_t_rows, X_t_cols)]
            N = len(X_t_values)  # number of sample point
            Para = np.array([V, K, J, N], dtype=np.int32)  #

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

    def crt_multi_aug(self, Xt_to_t1_t, Phi_t1, Theta_t1, dtype='dense'):

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

    def conv_multi_aug(self, row_index, col_index, n_index, value_index, D1_k1, W1_nk1):

        # check by chaojies
        [K1, K0, K1_S3, K1_S4] = D1_k1.shape  # K1*K0*K1_S3*K1_S4
        [N, K1, K1_S1, K1_S2] = W1_nk1.shape  # N*K1*K1_S1*K1_S2

        # 增广矩阵用于更新s12维度上增广
        X_rows = gpuarray.to_gpu(np.array(row_index, dtype=np.int32, order='C'))
        X_cols = gpuarray.to_gpu(np.array(col_index + 1, dtype=np.int32, order='C'))  # padding!!!
        X_file_index = gpuarray.to_gpu(np.array(n_index, dtype=np.int32, order='C'))
        X_value = gpuarray.to_gpu(np.array(value_index, dtype=np.float32, order='C'))
        word_total = int(X_rows.size)

        if word_total == 0:
            return np.zeros([N, K1, K1_S1, K1_S2]), np.zeros([K1, K0, K1_S3, K1_S4])
        else:
            W1_nk1 = gpuarray.to_gpu(np.array(W1_nk1, dtype=np.float32, order='C'))
            D1_k1 = gpuarray.to_gpu(np.array(np.swapaxes(D1_k1, 0, 1), dtype=np.float32, order='C'))  # K1*K0*K1_S3*K1_S4

            W1_nk1_Aug = gpuarray.zeros(W1_nk1.shape, dtype=np.float32, order='C')
            D1_k1_Aug = gpuarray.zeros(D1_k1.shape, dtype=np.float32, order='C')

            randomseed = cuda_generator.gen_uniform([word_total], dtype=np.float32)

            # 转化为GPU的输入形式
            fuc = mod.get_function("_conv_multi_augmentation")
            Batch_Para = gpuarray.to_gpu(np.array([K0, K1, K1_S1, K1_S2, K1_S3, K1_S4, word_total], dtype=np.int32, order='C'))

            block_x = int(500)
            grid_x = int(np.floor(word_total / block_x) + 1)

            fuc(randomseed, Batch_Para, X_rows, X_cols, X_file_index, X_value, W1_nk1, D1_k1,
                W1_nk1_Aug, D1_k1_Aug, grid=(grid_x, 1, 1), block=(block_x, 1, 1))  # 一般最多512个并行线程

            return W1_nk1_Aug.get(), np.swapaxes(D1_k1_Aug.get(), 0, 1)

