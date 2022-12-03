#include <curand_kernel.h>
#include "sampler_kernel_linux.h"

// ------------------------------------------------rand status ------------------------------------------
__global__ void _build_status_shared (size_t seed, curandStateXORWOW_t* status) {
    __shared__ curandStateXORWOW_t status_shared[blockDimY][blockDimX]; //共享内存，数据常驻缓存status_shared[2][32]
    size_t idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    curand_init(seed, idx, 0, &status_shared[threadIdx.y][threadIdx.x]);
    status[idx] = status_shared[threadIdx.y][threadIdx.x];
}
extern "C" void* _init_status(size_t seed) {
    // 线程数blockDim <= 1024，block大小为32倍数； gridDim <= 1<<32
    curandStateXORWOW_t* status = (curandStateXORWOW_t*)malloc(sizeof(curandStateXORWOW_t));
    cudaMalloc((void**)&status, nStatus * sizeof(curandStateXORWOW_t));

    dim3 grid(gridDimX), block(blockDimX, blockDimY);

    _build_status_shared << <grid, block >> > (seed, status);
    cudaDeviceSynchronize();
    return (void*)status;
}

// ================================================== sampler ============================================

// ------------------------------------------------sample gamma ------------------------------------------
__global__ void _rand_gamma(float* shape, float* scale, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {//根据输入采样，得到sst->output
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x; //grid(nElm/(32*4)) block(32*4)  均1维  range nElm
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats; // idx%(nElm/1)=idx 如果repeat是其他整数，则重复选取nElm/repeat元素
        float sh = shape[matrix_idx]; //matrix_idx可能会超出shape的索引范围，尽管可以引用(nElm肯能大于matrix_scale(lyw_g L34)): the reason:可以对冗余的值进行采样，因为return时会截取
        float sc = scale[matrix_idx];

        float U, V, X, Y;
        if (sh == 1.0) {
            size_t state_idx = idx % nStatus;
            U = curand_uniform(&status[state_idx]);
            output[idx] = -logf(U) * sc;
        } else if (sh <= 0) {
            output[idx] = 0.0;
            if (sh < 0) {
                printf("Warning: shape %f <= 0 in threads idx: %zu [thread:(%d, %d),  block:(%d, %d)]\n", sh, idx, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
            }
        } else if (sh < 1.0) {
            size_t state_idx = idx % (nStatus / 2);

            for (;;) {
                U = curand_uniform(&status[state_idx]);  // (0, 1]
                V = -logf(curand_uniform(&status[state_idx + nStatus / 2]));
                if (U <= 1.0 - sh) {
                    X = powf(U, 1. / sh);
                    if (X <= V) {
                        output[idx] = X * sc;
                        return;
                    }
                } else {
                    Y = -logf((U) / sh);
                    X = powf(1.0 - sh + sh * Y, 1. / sh);
                    if (X <= (V + Y)) {
                        output[idx] = X * sc;
                        return;
                    }
                }
            }
        } else {
            size_t state_idx = idx % (nStatus / 2);
            float b = sh - one_third;
            float c = one_third / sqrt(b);
            for (;;) {
                do {
                    X = curand_normal(&status[state_idx]);
                    V = 1.0 + c*X;
                } while (V <= 0.0);
                V = V*V*V;
                U = curand_uniform(&status[state_idx + nStatus / 2]);
                if ((U < 1.0 - 0.0331*(X*X)*(X*X)) || (logf(U) < 0.5*X*X + b*(1. - V + logf(V)))) {
                    output[idx] = b*V*sc;
                    return;
                }
            }
        }
    }
}
extern "C" void _sample_gamma(float* shape_host, float* scale_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* shape_device, * scale_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&shape_device, nBytes);
    cudaMalloc((void**)&scale_device, nBytes);
    cudaMemcpy(shape_device, shape_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(scale_device, scale_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma); // grid(nElm/(32*4)) block(32*4)   nThreads_g-xy

    _rand_gamma <<<grid, block>>> (shape_device, scale_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);
//    printf("matrix_scale:%zu, require:%zu  grid:%zu, block:%d\n\n", matrix_scale, sst_device->nElems, size_t(ceil(float(sst_device ->nElems) / float(nThreads_gamma))), nThreads_gamma); //=========================

    cudaFree(shape_device);
    cudaFree(scale_device);
    cudaFree(output_device);
}

// -------------------------------------------sample standrad_gamma --------------------------------------
// cupy/random/_kernels.py: the difference of gamma and standdrad gamma mainly in the allocation of status and the for loop
__global__ void _rand_standard_gamma(float* shape, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        float sh = shape[matrix_idx];
        //        output[idx] = curand_normal(&status[state_idx]) * scale[matrix_idx] + loc[matrix_idx];

        float U, V, X, Y;
        if (sh == 1.0) {
            size_t state_idx = idx % nStatus;
            U = curand_uniform(&status[state_idx]);
            output[idx] = -logf(U);
        } else if (sh <= 0) {
            output[idx] = 0.0;
            if (sh < 0) {
                printf("Warning: shape %f <= 0 in threads idx: %zu [thread:(%d, %d),  block:(%d, %d)]\n", sh, idx, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
            }
        } else if (sh < 1.0) {
            size_t state_idx = idx % (nStatus / 2);

            for (;;) {
                U = curand_uniform(&status[state_idx]);
                V = -logf(curand_uniform(&status[state_idx + nStatus / 2]));
                if (U <= 1.0 - sh) {
                    X = powf(U, 1. / sh);
                    if (X <= V) {
                        output[idx] = X;
                        return;
                    }
                } else {
                    Y = -logf((U) / sh);
                    X = powf(1.0 - sh + sh * Y, 1. / sh);
                    if (X <= (V + Y)) {
                        output[idx] = X;
                        return;
                    }
                }
            }
        } else {
            size_t state_idx = idx % (nStatus / 2);
            float b = sh - one_third;
            float c = one_third / sqrt(b);
            for (;;) {
                do {
                    X = curand_normal(&status[state_idx]);
                    V = 1.0 + c*X;
                } while (V <= 0.0);
                V = V*V*V;
                U = curand_uniform(&status[state_idx + nStatus / 2]);
                if ((U < 1.0 - 0.0331*(X*X)*(X*X)) || (logf(U) < 0.5*X*X + b*(1. - V + logf(V)))) {
                    output[idx] = b*V;
                    return;
                }
            }
        }
    }
}
extern "C" void _sample_standard_gamma(float* shape_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* shape_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&shape_device, nBytes);
    cudaMemcpy(shape_device, shape_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_standard_gamma <<<grid, block>>> (shape_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(shape_device);
    cudaFree(output_device);
}

// -------------------------------------------------sample beta ------------------------------------------
__global__ void _rand_beta(float* a, float* b, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        float sh1 = a[matrix_idx];
        float sh2 = b[matrix_idx];

        if ((sh1 <= 1.0) && (sh2 <= 1.0)) {
            double U, V, X, Y;
            /* Use Johnk's algorithm */
            while (1) {
                size_t state_idx = idx % (nStatus / 2);
                U = curand_uniform(&status[state_idx]);
                V = curand_uniform(&status[state_idx + nStatus / 2]);

                X = pow(U, 1.0 / sh1);
                Y = pow(V, 1.0 / sh2);

                if ((X + Y) <= 1.0) {
                    if (X + Y > 0) {
                        output[idx] = X / (X + Y);
                        return;
                    } else {
                        double logX = log(U) / sh1;
                        double logY = log(V) / sh2;
                        double logM = logX > logY ? logX: logY;

                        logX -= logM;
                        logY -= logM;
                        output[idx] = exp(logX - log(exp(logX) + exp(logY)));
                        return;
                    }
                }
            }
        } else{
            float gamma1, gamma2;
            float U, V, X, Y;

            // gamma 1
            if (sh1 == 1.0) {
                size_t state_idx = idx % nStatus;
                U = curand_uniform(&status[state_idx]);
                gamma1 = -logf(U);
            } else if (sh1 <= 0) {
                gamma1 = 0.0;
//                 if (sh1 < 0){
//                     printf("Warning: shape %f <= 0 in threads idx: %zu [thread:(%d, %d),  block:(%d, %d)]\n", sh1, idx, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
//                 }
            } else if (sh1 < 1.0) {
                for (;;) {
                    size_t state_idx = idx % (nStatus / 2);
                    U = curand_uniform(&status[state_idx]);
                    V = -logf(curand_uniform(&status[state_idx + nStatus / 2]));
                    if (U <= 1.0 - sh1) {
                        X = powf(U, 1. / sh1);
                        if (X <= V) {
                            gamma1 = X;
                            break;
                        }
                    } else {
                        Y = -logf((1 - U) / sh1);
                        X = powf(1.0 - sh1 + sh1 * Y, 1. / sh1);
                        if (X <= (V + Y)) {
                            gamma1 = X;
                            break;
                        }
                    }
                }
            } else {
                size_t state_idx = idx % (nStatus / 2);
                float b = sh1 - one_third;
                float c = one_third / sqrt(b);
                for (;;) {
                    do {
                        X = curand_normal(&status[state_idx]);
                        V = 1.0 + c*X;
                    } while (V <= 0.0);
                    V = V*V*V;
                    U = curand_uniform(&status[state_idx + nStatus / 2]);
                    if ((U < 1.0 - 0.0331*(X*X)*(X*X)) || (logf(U) < 0.5*X*X + b*(1. - V + logf(V)))) {
                        gamma1 = b*V;
                        break;
                    }
                }
            }

            // gamma2
            if (sh2 == 1.0) {
                size_t state_idx = idx % nStatus;
                U = curand_uniform(&status[state_idx]);
                gamma2 = -logf(U);
            } else if (sh2 <= 0) {
                gamma2 = 0.0;
//                 if (sh2 < 0){
//                     printf("Warning: shape %f <= 0 in threads idx: %zu [thread:(%d, %d),  block:(%d, %d)]\n", sh2, idx, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
//                 }
            } else if (sh2 < 1.0) {
                for (;;) {
                    size_t state_idx = idx % (nStatus / 2);
                    U = curand_uniform(&status[state_idx]);
                    V = -logf(curand_uniform(&status[state_idx + nStatus / 2]));
                    if (U <= 1.0 - sh2) {
                        X = powf(U, 1. / sh2);
                        if (X <= V) {
                            gamma2 = X;
                            break;
                        }
                    } else {
                        Y = -logf((1 - U) / sh2);
                        X = powf(1.0 - sh2 + sh2 * Y, 1. / sh2);
                        if (X <= (V + Y)) {
                            gamma2 = X;
                            break;
                        }
                    }
                }
            } else {
                size_t state_idx = idx % (nStatus / 2);
                float b = sh2 - one_third;
                float c = one_third / sqrt(b);
                for (;;) {
                    do {
                        X = curand_normal(&status[state_idx]);
                        V = 1.0 + c*X;
                    } while (V <= 0.0);
                    V = V*V*V;
                    U = curand_uniform(&status[state_idx + nStatus / 2]);
                    if ((U < 1.0 - 0.0331*(X*X)*(X*X)) || (logf(U) < 0.5*X*X + b*(1. - V + logf(V)))) {
                        gamma2 = b*V;
                        break;
                    }
                }
            }

            // sample beta
            output[idx] = gamma1 / (gamma1 + gamma2);
        }
    }
}
extern "C" void _sample_beta(float* a_host, float* b_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* a_device, * b_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&a_device, nBytes);
    cudaMalloc((void**)&b_device, nBytes);
    cudaMemcpy(a_device, a_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_beta <<<grid, block>>> (a_device, b_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(output_device);
}

// ------------------------------------------sample standrad_normal --------------------------------------
__global__ void _rand_standard_normal(float* output, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t state_idx = idx % nStatus;
        output[idx] = curand_normal(&status[state_idx]);
    }
}
extern "C" void _sample_standard_normal(void* output, size_t nElems, curandStateXORWOW_t* status) {
    float* output_device;
    size_t nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_standard_normal <<<grid, block>>> (output_device, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(output_device);
}

// -----------------------------------------------sample normal ------------------------------------------
__global__ void _rand_normal(float* loc, float* scale, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        output[idx] = curand_normal(&status[state_idx]) * scale[matrix_idx] + loc[matrix_idx];
    }
}
extern "C" void _sample_normal(float* loc_host, float* scale_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* loc_device, * scale_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&loc_device, nBytes);
    cudaMalloc((void**)&scale_device, nBytes);
    cudaMemcpy(loc_device, loc_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(scale_device, scale_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_normal <<<grid, block>>> (loc_device, scale_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(loc_device);
    cudaFree(scale_device);
    cudaFree(output_device);
}

// ------------------------------------------sample standrad_uniform --------------------------------------
__global__ void _rand_standard_uniform(float* output, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t state_idx = idx % nStatus;
        output[idx] = curand_uniform(&status[state_idx]);
    }
}
extern "C" void _sample_standard_uniform(void* output, size_t nElems, curandStateXORWOW_t* status) {
    float* output_device;
    size_t nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_standard_uniform <<<grid, block>>> (output_device, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(output_device);
}

// -----------------------------------------------sample uniform -----------------------------------------
__global__ void _rand_uniform(float* low, float* high, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        output[idx] = curand_uniform(&status[state_idx]) * (high[matrix_idx] - low[matrix_idx]) + low[matrix_idx];
    }
}
extern "C" void _sample_uniform(float* low_host, float* high_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* low_device, * high_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&low_device, nBytes);
    cudaMalloc((void**)&high_device, nBytes);
    cudaMemcpy(low_device, low_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(high_device, high_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_uniform <<<grid, block>>> (low_device, high_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(low_device);
    cudaFree(high_device);
    cudaFree(output_device);
}

// ------------------------------------------sample negative_binomial ------------------------------------
__global__ void _rand_negative_binomial(int* r, float* p, int* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;

        int suc = 0;
        float fail = 0.0;
        int total_r = r[matrix_idx];
        float prob = p[matrix_idx];
        while (total_r > fail) {
            if (curand_uniform(&status[state_idx]) < prob)
                suc++;
            else
                fail++;
        }
        output[idx] = suc;
    }
}
extern "C" void _sample_negative_binomial(int* r_host, float* p_host, int* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    int* r_device;
    size_t nBytes = matrix_scale * sizeof(int);
    cudaMalloc((void**)&r_device, nBytes);
    cudaMemcpy(r_device, r_host, nBytes, cudaMemcpyHostToDevice);
    float* p_device;
    nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&p_device, nBytes);
    cudaMemcpy(p_device, p_host, nBytes, cudaMemcpyHostToDevice);

    int* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(int);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_negative_binomial <<<grid, block>>> (r_device, p_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(r_device);
    cudaFree(p_device);
    cudaFree(output_device);
}

// -----------------------------------------------sample poisson -----------------------------------------
__global__ void _rand_poisson(float* lam, int* output, size_t matrix_scale, size_t nElems, curandStateXORWOW_t* status) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nElems) {
        int k = 0;
        size_t state_idx = idx % nStatus;
        float lamb = lam[int(idx / matrix_scale)];
        float p = 1.0;
        float l = exp(-lamb);
        while (p >= l) {
            float u = curand_uniform(&status[state_idx]);
            p *= u;
            k++;
        }
        output[idx] = k-1;
    }
}
extern "C" void _sample_poisson(float* lam_host, int* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* lam_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&lam_device, nBytes);
    cudaMemcpy(lam_device, lam_host, nBytes, cudaMemcpyHostToDevice);

    int* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(int);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_poisson <<<grid, block>>> (lam_device, output_device, matrix_scale, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(lam_device);
    cudaFree(output_device);
}

// -------------------------------------------------sample crt -------------------------------------------
__global__ void _rand_crt(float* point, float* p, int* output, size_t matrix_scale, size_t nElems, curandStateXORWOW_t* status) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nElems) {
        size_t state_idx = idx % nStatus;
        int token, table;
        int current_scale = idx / matrix_scale;
        float num = point[current_scale];
        float cum_sum = p[current_scale];
        if(num < 0.5) {
            table = 0;
        } else {
            for (token = 1, table = 1; token < num; token++) {
                float u = curand_uniform(&status[state_idx]);
                if (u <= cum_sum / (cum_sum + token))
                    table++;
            }
        }
        output[idx] = table;
    }
}
extern "C" void _sample_crt(float* point_host, float* p_host, int* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* point_device, * p_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&point_device, nBytes);
    cudaMalloc((void**)&p_device, nBytes);
    cudaMemcpy(point_device, point_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(p_device, p_host, nBytes, cudaMemcpyHostToDevice);

    int* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(int);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_crt <<<grid, block>>> (point_device, p_device, output_device, matrix_scale, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(point_device);
    cudaFree(p_device);
    cudaFree(output_device);
}

// -----------------------------------------------sample cauchy ------------------------------------------
__global__ void _rand_cauchy(float* loc, float* scale, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        float u = curand_uniform(&status[state_idx]) - 0.5;
        output[idx] = loc[matrix_idx] + tanf(u * Pi) * scale[matrix_idx]; // rows first
    }
}
extern "C" void _sample_cauchy(float* loc_host, float* scale_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* loc_device, * scale_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&loc_device, nBytes);
    cudaMalloc((void**)&scale_device, nBytes);
    cudaMemcpy(loc_device, loc_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(scale_device, scale_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_cauchy <<<grid, block>>> (loc_device, scale_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(loc_device);
    cudaFree(scale_device);
    cudaFree(output_device);
}

// -------------------------------------------sample standard_cauchy -------------------------------------
__global__ void _rand_standard_cauchy(float* output, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t state_idx = idx % nStatus;
        float u = curand_uniform(&status[state_idx]) - 0.5;
        output[idx] = tanf(u * Pi); // rows first
    }
}
extern "C" void _sample_standard_cauchy(float* output, size_t nElems, curandStateXORWOW_t* status) {
    float* output_device;
    size_t nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_standard_cauchy <<<grid, block>>> (output_device, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(output_device);
}

// ---------------------------------------------sample chisquare -----------------------------------------
__global__ void _rand_chisquare(int* degrees, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        float sum = 0;
        for (int i = 0; i < degrees[matrix_idx]; i++) {
            float x = pow(curand_normal(&status[state_idx]), 2);
            sum += x;
        }
        output[idx] = sum;
    }
}
extern "C" void _sample_chisquare(int* degrees_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    int* degrees_device;
    size_t nBytes = matrix_scale * sizeof(int);
    cudaMalloc((void**)&degrees_device, nBytes);
    cudaMemcpy(degrees_device, degrees_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_chisquare <<<grid, block>>> (degrees_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(degrees_device);
    cudaFree(output_device);
}

// ----------------------------------------sample noncentral_chisquare ------------------------------------
__global__ void _rand_noncentral_chisquare(int* df_, float* nonc_, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        int df = df_[matrix_idx];
        float nonc = nonc_[matrix_idx];
        if (nonc == 0) {
            float sum = 0;
            for (int i = 0; i < df; i++) {
                float x = pow(curand_normal(&status[state_idx]), 2);
                sum += x;
            }
            output[idx] = sum;
        } else {
            float N = curand_normal(&status[state_idx]) + sqrt(nonc);
            if (df > 1) {
                float Chi2 = 0;
                for (int i = 0; i < df-1; i++) {
                    float x = pow(curand_normal(&status[state_idx]), 2);
                    Chi2 += x;
                }
                output[idx] = Chi2 + N*N;
            } else {
                output[idx] = N*N;
            }
        }
    }
}
extern "C" void _sample_noncentral_chisquare(int* df_host, float* nonc_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    int* df_device;
    size_t nBytes = matrix_scale * sizeof(int);
    cudaMalloc((void**)&df_device, nBytes);
    cudaMemcpy(df_device, df_host, nBytes, cudaMemcpyHostToDevice);
    float* nonc_device;
    nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&nonc_device, nBytes);
    cudaMemcpy(nonc_device, nonc_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_noncentral_chisquare <<<grid, block>>> (df_device, nonc_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(df_device);
    cudaFree(nonc_device);
    cudaFree(output_device);
}

// --------------------------------------------sample exponential ----------------------------------------
__global__ void _rand_exponential(float* Lambda, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        output[idx] = -Lambda[matrix_idx] * log(curand_uniform(&status[state_idx]));
    }
}
extern "C" void _sample_exponential(float* Lambda_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* Lambda_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&Lambda_device, nBytes);
    cudaMemcpy(Lambda_device, Lambda_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_exponential <<<grid, block>>> (Lambda_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(Lambda_device);
    cudaFree(output_device);
}

// -------------------------------------------------sample f ---------------------------------------------
__global__ void _rand_f(int* n1, int* n2, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;

        float sum1 = 0;
        for (int i = 0; i < n1[matrix_idx]; i++) {
            float x = pow(curand_normal(&status[state_idx]), 2);
            sum1 += x;
        }

        float sum2 = 0;
        for (int i = 0; i < n2[matrix_idx]; i++) {
            float x = pow(curand_normal(&status[state_idx]), 2);
            sum2 += x;
        }

        output[idx] = (sum1 / n1[matrix_idx]) / (sum2 / n2[matrix_idx]);
    }
}
extern "C" void _sample_f(int* n1_host, int* n2_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    int* n1_device, * n2_device;
    size_t nBytes = matrix_scale * sizeof(int);
    cudaMalloc((void**)&n1_device, nBytes);
    cudaMalloc((void**)&n2_device, nBytes);
    cudaMemcpy(n1_device, n1_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(n2_device, n2_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_f <<<grid, block>>> (n1_device, n2_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(n1_device);
    cudaFree(n2_device);
    cudaFree(output_device);
}

// -------------------------------------------------sample noncentral_f ---------------------------------------------
__global__ void _rand_noncentral_f(int* dfnum_, int* dfden_, float* nonc_, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        int dfnum = dfnum_[matrix_idx];
        int dfden = dfden_[matrix_idx];
        float nonc = nonc_[matrix_idx];

        // get num_ncchi2: noncentral_chisquare(dfnum, nonc)
        float num_ncchi2;
        if (nonc == 0) {
            float sum = 0;
            for (int i = 0; i < dfnum; i++) {
                float x = pow(curand_normal(&status[state_idx]), 2);
                sum += x;
            }
            num_ncchi2 = sum;
        } else {
            float N = curand_normal(&status[state_idx]) + sqrt(nonc);
            if (dfnum > 1) {
                float Chi2 = 0;
                for (int i = 0; i < dfnum-1; i++) {
                    float x = pow(curand_normal(&status[state_idx]), 2);
                    Chi2 += x;
                }
                num_ncchi2 = Chi2 + N*N;
            } else {
                num_ncchi2 = N*N;
            }
        }

        float t = num_ncchi2 * dfden;

        // get den_chi2: chisquare(dfden)
        float den_chi2 = 0;
        for (int i = 0; i < dfden; i++) {
            float x = pow(curand_normal(&status[state_idx]), 2);
            den_chi2 += x;
        }

        output[idx] = t / (den_chi2 * dfnum);
    }
}
extern "C" void _sample_noncentral_f(int* dfnum_host, int* dfden_host, float* nonc_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    int* dfnum_device, * dfden_device;
    size_t nBytes = matrix_scale * sizeof(int);
    cudaMalloc((void**)&dfnum_device, nBytes);
    cudaMalloc((void**)&dfden_device, nBytes);
    cudaMemcpy(dfnum_device, dfnum_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dfden_device, dfden_host, nBytes, cudaMemcpyHostToDevice);
    float* nonc_device;
    nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&nonc_device, nBytes);
    cudaMemcpy(nonc_device, nonc_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_noncentral_f <<<grid, block>>> (dfnum_device, dfden_device, nonc_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(dfnum_device);
    cudaFree(dfden_device);
    cudaFree(nonc_device);
    cudaFree(output_device);
}

// ---------------------------------------------sample geometric -----------------------------------------
__global__ void _rand_geometric(float* p, int* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        int rnd = 0;
        while (++rnd) {
            if (curand_uniform(&status[state_idx]) < p[matrix_idx]) {
                output[idx] = rnd;
                return;
            }
            if (rnd > 1000) {
                output[idx] = rnd;
                return;
            }
        }
    }
}
extern "C" void _sample_geometric(float* p_host, int* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* p_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&p_device, nBytes);
    cudaMemcpy(p_device, p_host, nBytes, cudaMemcpyHostToDevice);

    int* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(int);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_geometric <<<grid, block>>> (p_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(p_device);
    cudaFree(output_device);
}

// -----------------------------------------------sample gumbel ------------------------------------------
__global__ void _rand_gumbel(float* loc, float* scale, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        output[idx] = loc[matrix_idx] - scale[matrix_idx] * log(-log(curand_uniform(&status[state_idx])));
    }
}
extern "C" void _sample_gumbel(float* loc_host, float* scale_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* loc_device, * scale_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&loc_device, nBytes);
    cudaMalloc((void**)&scale_device, nBytes);
    cudaMemcpy(loc_device, loc_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(scale_device, scale_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_gumbel <<<grid, block>>> (loc_device, scale_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(loc_device);
    cudaFree(scale_device);
    cudaFree(output_device);
}

// -------------------------------------------sample hypergeometric --------------------------------------
static __device__ float loggam(float x) {
    double x0, x2, xp, gl, gl0;
    long k, n;
    double a[10] = {8.333333333333333e-02,-2.777777777777778e-03,
                    7.936507936507937e-04,-5.952380952380952e-04,
                    8.417508417508418e-04,-1.917526917526918e-03,
                    6.410256410256410e-03,-2.955065359477124e-02,
                    1.796443723688307e-01,-1.39243221690590e+00};
    x0 = x;
    n = 0;
    if ((x == 1.0) || (x == 2.0)) {
        return 0.0;
    } else if (x <= 7.0) {
        n = (long)(7 - x);
        x0 = x + n;
    }
    x2 = 1.0/(x0*x0);
    xp = 2*M_PI;
    gl0 = a[9];
    for (k=8; k>=0; k--) {
        gl0 *= x2;
        gl0 += a[k];
    }
    gl = gl0/x0 + 0.5*log(xp) + (x0-0.5)*log(x0) - x0;
    if (x <= 7.0) {
        for (k=1; k<=n; k++) {
            gl -= log(x0-1.0);
            x0 -= 1.0;
        }
    }
    return (float)gl;
}
__global__ void _rand_hypergeometric(int* ngood, int* nbad, int* nsample, int* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        int good = ngood[matrix_idx];
        int bad = nbad[matrix_idx];
        int sample = nsample[matrix_idx];
        if (sample > 10) {
            // rk_hypergeometric_hrua in cupy.random._kernels.py
            /* D1 = 2*sqrt(2/e) */
            /* D2 = 3 - 2*sqrt(3/e) */
            float D1 = 1.7155277699214135;
            float D2 = 0.8989161620588988;

            int mingoodbad, maxgoodbad, popsize, m, d9;
            float d4, d5, d6, d7, d8, d10, d11;
            int Z;
            float T, W, X, Y;

            mingoodbad = min(good, bad);
            popsize = good + bad;
            maxgoodbad = max(good, bad);
            m = min(sample, popsize - sample);
            d4 = ((float)mingoodbad) / popsize;
            d5 = 1.0 - d4;
            d6 = m*d4 + 0.5;
            d7 = sqrt((float)(popsize - m) * sample * d4 * d5 / (popsize - 1) + 0.5);
            d8 = D1*d7 + D2;
            d9 = (int)floor((float)(m + 1) * (mingoodbad + 1) / (popsize + 2));
            d10 = (loggam(d9+1) + loggam(mingoodbad-d9+1) + loggam(m-d9+1) +
                   loggam(maxgoodbad-m+d9+1));
            d11 = min(min(m, mingoodbad)+1.0, floor(d6+16*d7));
            /* 16 for 16-decimal-digit precision in D1 and D2 */

            while (1)
            {
                X = curand_uniform(&status[state_idx]);
                Y = curand_uniform(&status[state_idx]);
                W = d6 + d8*(Y- 0.5)/X;

                /* fast rejection: */
                if ((W < 0.0) || (W >= d11)) continue;

                Z = (int)floor(W);
                T = d10 - (loggam(Z+1) + loggam(mingoodbad-Z+1) + loggam(m-Z+1) +
                           loggam(maxgoodbad-m+Z+1));

                /* fast acceptance: */
                if ((X*(4.0-X)-3.0) <= T) break;

                /* fast rejection: */
                if (X*(X-T) >= 1) continue;

                if (2.0*log(X) <= T) break;  /* acceptance */
            }

            /* this is a correction to HRUA* by Ivan Frohne in rv.py */
            if (good > bad) Z = m - Z;

            /* another fix from rv.py to allow sample to exceed popsize/2 */
            if (m < sample) Z = good - Z;

            output[idx] = Z;
        } else {
            // rk_hypergeometric_hyp in cupy
            int d1, K, Z;
            float d2, U, Y;

            d1 = bad + good - sample;
            d2 = min(bad, good);

            Y = d2;
            K = sample;
            while (Y > 0.0) {
                U = curand_uniform(&status[state_idx]); // rk_double(state) [0, 1) in cupy. differ in cuda curand_uniform(&status[state_idx])
                Y -= floor(U + Y/(d1 + K));
                K--;
                if (K == 0) break;
            }
            Z = (d2 - Y);
            if (good > bad) Z = sample - Z;
            output[idx] = Z;
        }
    }
}
extern "C" void _sample_hypergeometric(int* ngood_host, int* nbad_host, int* nsample_host, int* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    int* ngood_device, * nbad_device, * nsample_device;
    size_t nBytes = matrix_scale * sizeof(int);
    cudaMalloc((void**)&ngood_device, nBytes);
    cudaMalloc((void**)&nbad_device, nBytes);
    cudaMalloc((void**)&nsample_device, nBytes);
    cudaMemcpy(ngood_device, ngood_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(nbad_device, nbad_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(nsample_device, nsample_host, nBytes, cudaMemcpyHostToDevice);

    int* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(int);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_hypergeometric <<<grid, block>>> (ngood_device, nbad_device, nsample_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(ngood_device);
    cudaFree(nbad_device);
    cudaFree(nsample_device);
    cudaFree(output_device);
}

// -----------------------------------------------sample laplace ------------------------------------------
__global__ void _rand_laplace(float* loc, float* scale, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        float u = curand_uniform(&status[state_idx]) - 0.5;
        int sign = (u == 0? 0: u / abs(u));
        output[idx] = loc[matrix_idx] - scale[matrix_idx] * sign * log(1 - 2 * abs(u));
    }
}
extern "C" void _sample_laplace(float* loc_host, float* scale_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* loc_device, * scale_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&loc_device, nBytes);
    cudaMalloc((void**)&scale_device, nBytes);
    cudaMemcpy(loc_device, loc_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(scale_device, scale_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_laplace <<<grid, block>>> (loc_device, scale_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(loc_device);
    cudaFree(scale_device);
    cudaFree(output_device);
}

// ----------------------------------------------sample logistic -----------------------------------------
__global__ void _rand_logistic(float* loc, float* scale, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        float u = curand_uniform(&status[state_idx]);
        output[idx] = loc[matrix_idx] + scale[matrix_idx] * (log(u) - log(1 - u));
    }
}
extern "C" void _sample_logistic(float* loc_host, float* scale_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* loc_device, * scale_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&loc_device, nBytes);
    cudaMalloc((void**)&scale_device, nBytes);
    cudaMemcpy(loc_device, loc_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(scale_device, scale_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_logistic <<<grid, block>>> (loc_device, scale_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(loc_device);
    cudaFree(scale_device);
    cudaFree(output_device);
}

// ------------------------------------------------sample power ------------------------------------------
__global__ void _rand_power(float* a, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        output[idx] = pow(double(1 - curand_uniform(&status[state_idx])), double(1 / a[matrix_idx]));
    }
}
extern "C" void _sample_power(float* a_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* a_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&a_device, nBytes);
    cudaMemcpy(a_device, a_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_power <<<grid, block>>> (a_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(a_device);
    cudaFree(output_device);
}

// -------------------------------------------------sample zipf ------------------------------------------
__global__ void _rand_zipf(float* a, int* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        double am1, b;
        am1 = a[matrix_idx] - 1.0;
        b = powf(2.0, am1);
        while (1) {
            double T, U, V, X;
            U = curand_uniform(&status[state_idx]);
            V = curand_uniform(&status[state_idx]);
            X = floor(powf(U, -1.0/am1));
            if (X < 1.0) {
                continue;
            }
            T = powf(1.0 + 1.0/X, am1);
            if (V*X*(T - 1.0)/(b - 1.0) <= T/b) {
                output[idx] = X;
                return;
            }
        }
    }
}
extern "C" void _sample_zipf(float* a_host, int* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* a_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&a_device, nBytes);
    cudaMemcpy(a_device, a_host, nBytes, cudaMemcpyHostToDevice);

    int* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(int);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_zipf <<<grid, block>>> (a_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(a_device);
    cudaFree(output_device);
}

// ------------------------------------------------sample pareto ------------------------------------------
__global__ void _rand_pareto(float* k, float* xm, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        output[idx] = xm[matrix_idx] / powf(1 - curand_uniform(&status[state_idx]), 1. / k[matrix_idx]) - 1;
    }
}
extern "C" void _sample_pareto(float* k_host, float* xm_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* k_device, * xm_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&k_device, nBytes);
    cudaMalloc((void**)&xm_device, nBytes);
    cudaMemcpy(k_device, k_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(xm_device, xm_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_pareto <<<grid, block>>> (k_device, xm_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(k_device);
    cudaFree(xm_device);
    cudaFree(output_device);
}

// ----------------------------------------------sample rayleigh -----------------------------------------
__global__ void _rand_rayleigh(float* scale, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        float sc = scale[matrix_idx];
        float U, V;
        float z, x;
        U = curand_normal(&status[state_idx]);
        V = curand_normal(&status[state_idx]);
        z = sqrt(-2.0 * logf(U)) * sin(2.0 * Pi * V) * sc;
        U = curand_normal(&status[state_idx]);
        V = curand_normal(&status[state_idx]);
        x = sqrt(-2.0 * logf(U)) * sin(2.0 * Pi * V) * sc;
        output[idx] = sqrt(z * z + x * x);
    }
}
extern "C" void _sample_rayleigh(float* scale_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* scale_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&scale_device, nBytes);
    cudaMemcpy(scale_device, scale_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_rayleigh <<<grid, block>>> (scale_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(scale_device);
    cudaFree(output_device);
}

// --------------------------------------------------sample t --------------------------------------------
__global__ void _rand_t(float* df, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        float N = df[matrix_idx];
        float sum = 0;
        for (int i = 0; i < N; i++) {
            float x = pow(double(curand_normal(&status[state_idx])), 2);
            sum += x;
        }
        output[idx] = curand_normal(&status[state_idx]) / sqrt(sum / N);
    }
}
extern "C" void _sample_t(float* df_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* df_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&df_device, nBytes);
    cudaMemcpy(df_device, df_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_t <<<grid, block>>> (df_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(df_device);
    cudaFree(output_device);
}

// ---------------------------------------------sample triangular ----------------------------------------
__global__ void _rand_triangular(float* left, float* mode, float* right, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        float l = left[matrix_idx];
        float r = right[matrix_idx];
        float m = mode[matrix_idx];
        float randx, randy, reject;
        while (true) {
            randx = curand_uniform(&status[state_idx]) * (r - l) + l;
            randy = curand_uniform(&status[state_idx]) * (2 / (r - l));
            reject = (randx <= m? 2 * (randx - l) / ((r - l) * (m - l)): 2 * (r - randx) / ((r - l) * (r - m)));
            if (randy <= reject) {
                output[idx] = randx;
                return;
            }
        }
    }
}
extern "C" void _sample_triangular(float* left_host, float* mode_host, float* right_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* left_device, * mode_device, * right_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&left_device, nBytes);
    cudaMalloc((void**)&mode_device, nBytes);
    cudaMalloc((void**)&right_device, nBytes);
    cudaMemcpy(left_device, left_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(mode_device, mode_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(right_device, right_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_triangular <<<grid, block>>> (left_device, mode_device, right_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(left_device);
    cudaFree(mode_device);
    cudaFree(right_device);
    cudaFree(output_device);
}

// ----------------------------------------------sample weibull ------------------------------------------
__device__ float log_max(float x) {
    return log(max(x, float(2.2e-10)));
}
__global__ void _rand_weibull(float* shape, float* scale, float* output, size_t repeats, size_t nElems, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nElems) {
        size_t matrix_idx = idx / repeats;
        size_t state_idx = idx % nStatus;
        output[idx] = float(scale[matrix_idx] * pow(double(-log_max(1 - curand_uniform(&status[state_idx]))), double(1.0 / shape[matrix_idx])));
    }
}
extern "C" void _sample_weibull(float* shape_host, float* scale_host, float* output, size_t matrix_scale, size_t repeats, curandStateXORWOW_t* status) {
    float* shape_device, * scale_device;
    size_t nBytes = matrix_scale * sizeof(float);
    cudaMalloc((void**)&shape_device, nBytes);
    cudaMalloc((void**)&scale_device, nBytes);
    cudaMemcpy(shape_device, shape_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(scale_device, scale_host, nBytes, cudaMemcpyHostToDevice);

    float* output_device;
    size_t nElems = matrix_scale * repeats;
    nBytes = nElems * sizeof(float);
    cudaMalloc((void**)&output_device, nBytes);

    dim3 grid(size_t(ceil(float(nElems) / float(nThreads_gamma)))), block(nThreads_gamma);
    _rand_weibull <<<grid, block>>> (shape_device, scale_device, output_device, repeats, nElems, status);
    cudaMemcpy(output, output_device, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(shape_device);
    cudaFree(scale_device);
    cudaFree(output_device);
}

// ----------------------------------------------sample multinomial ------------------------------------------
__global__ void _calculate_probs(float* prob, size_t matrix_scale_1, size_t matrix_scale_2) {
    // calculate the cumsum of prob for each multi distribution.
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < matrix_scale_1){
        size_t K = int(float(matrix_scale_2) / float(matrix_scale_1));
        float sum = 0.0;
        for (int i=0; i<K; i++){
            sum += prob[idx*K + i];
            prob[idx*K + i] = sum;
        }
        for (int i=0; i<K; i++){
            prob[idx*K + i] /= sum;
        }
    }
}
__global__ void _rand_multinomial(int* count, float* prob, int* output, size_t matrix_scale_1, size_t matrix_scale_2, size_t repeats, curandStateXORWOW_t* status) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < matrix_scale_1*repeats) {
        size_t K = int(float(matrix_scale_2) / float(matrix_scale_1));
        size_t matrix_idx = idx / repeats;  // row idx in prob_device
        size_t repeat_idx = idx % repeats;  // times
        size_t state_idx = idx % nStatus;

        for (int num=0; num<count[matrix_idx]; num++)
        {
            float rand_prob = curand_uniform(&status[state_idx]);
            for (int i=0 ; i<K; i++){
                if (rand_prob < prob[matrix_idx*K + i])  //(matrix_idx, i, repeat_idx)
                {
                    output[matrix_idx*K*repeats + i*repeats + repeat_idx] += 1;
                    break;
                }
            }
        }
    }
}
extern "C" void _sample_multinomial(int* count_host, float* prob_host, int* output, size_t matrix_scale_1, size_t matrix_scale_2, size_t repeats, curandStateXORWOW_t* status) {
    int* count_device;
    float* prob_device;

    size_t count_nBytes = matrix_scale_1 * sizeof(int);
    size_t prob_nBytes = matrix_scale_2 * sizeof(float);

    cudaMalloc((void**)&count_device, count_nBytes);
    cudaMalloc((void**)&prob_device, prob_nBytes);
    cudaMemcpy(count_device, count_host, count_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(prob_device, prob_host, prob_nBytes, cudaMemcpyHostToDevice);

    int* output_device;
    size_t output_nBytes = matrix_scale_2 * repeats * sizeof(int);
    cudaMalloc((void**)&output_device, output_nBytes);
    cudaMemcpy(output_device, output, output_nBytes, cudaMemcpyHostToDevice);

    dim3 grid(size_t(ceil(float(matrix_scale_1) / float(nThreads_gamma)))), block(nThreads_gamma); // grid(nElm/(32*4)) block(32*4)   nThreads_g-xy

    _calculate_probs <<<grid, block>>> (prob_device, matrix_scale_1, matrix_scale_2);

//     cudaMemcpy(prob_host, prob_device, prob_nBytes, cudaMemcpyDeviceToHost);
//     printf("%f, %f, %f, %f", prob_host[0], prob_host[1], prob_host[2], prob_host[3]);

    grid.x = size_t(ceil(float(matrix_scale_1 * repeats) / float(nThreads_gamma)));
    block.x = nThreads_gamma;
    _rand_multinomial <<<grid, block>>> (count_device, prob_device, output_device, matrix_scale_1, matrix_scale_2, repeats, status);

    cudaMemcpy(output, output_device, output_nBytes, cudaMemcpyDeviceToHost);

    cudaFree(count_device);
    cudaFree(prob_device);
    cudaFree(output_device);
}
