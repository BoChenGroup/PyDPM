#include <curand_kernel.h>
#include "model_sampler_win.h"

// ------------------------------------------------rand status ------------------------------------------
__global__ void _build_status_shared (size_t seed, curandStateXORWOW_t* status) {
    __shared__ curandStateXORWOW_t status_shared[blockDimY][blockDimX]; //共享内存，数据常驻缓存status_shared[2][32]
    size_t idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    curand_init(seed, idx, 0, &status_shared[threadIdx.y][threadIdx.x]);
    status[idx] = status_shared[threadIdx.y][threadIdx.x];
}
DLLEXPORT void* _init_status(size_t seed) {
    // 线程数blockDim <= 1024，block大小为32倍数； gridDim <= 1<<32
    curandStateXORWOW_t* status = (curandStateXORWOW_t*)malloc(sizeof(curandStateXORWOW_t));
    cudaMalloc((void**)&status, nStatus * sizeof(curandStateXORWOW_t));

    dim3 grid(gridDimX), block(blockDimX, blockDimY);

    _build_status_shared << <grid, block >> > (seed, status);
    cudaDeviceSynchronize();
    return (void*)status;
}

// ================================================== sampler ============================================
// ------------------------------------------------sample crt multinomial augmentation ------------------------------------------
__global__ void _rand_crt_multi_aug(int V, int K, int J, int N, float* X_Values, float* X_Rows, float* X_Cols, float* Phi, float* Theta, float* XVK, float* XKJ, curandStateXORWOW_t* status) {//根据输入采样，得到sst->output

    size_t idx = threadIdx.x + blockDim.x * blockIdx.x; //grid(nElm/(32*4)) block(32*4)  均1维  range nElm
    if (idx < N) {
//         printf("V:%d, K:%d, J:%d, N:%d\n", V, K, J, N);
        size_t state_idx = idx % nStatus;
        size_t word_value = X_Values[idx];
        size_t word_row = X_Rows[idx];
        size_t word_col = X_Cols[idx];

        // calculate sum(Phi[v, :]*Theta[:, j])
        float prob_sum = 0.0;
        for (int k=0; k<K; k++){
            prob_sum += Phi[word_row*K + k] * Theta[k*J + word_col];
        }

        // crt sample table num from customer num x
        int table = 1;
        for (int token = 1; token<word_value; token++) {
            float prob_threshold = curand_uniform(&status[state_idx]);
            if (prob_threshold <= prob_sum / (prob_sum + token)){
                table += 1;
            }
        }
//         printf("table num: %d", table);

        // augment table to a count vector
        float prob = 0.0;
        size_t topic_index = 0;
        for (int token=0; token<table; token++){
            float prob_threshold = curand_uniform(&status[state_idx]) * prob_sum; // It may return from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded.
            float prob_cumulated = 0.0;

            for(int k=0; k<K; k++){
                prob = Phi[word_row*K + k] * Theta[k*J + word_col];
                if (prob_cumulated + prob >=prob_threshold){
                    topic_index = k;
                    break;
                }
                prob_cumulated += prob;
            }

            atomicAdd(&XVK[word_row*K + topic_index], 1.0);
            atomicAdd(&XKJ[topic_index*J + word_col], 1.0);
        }
    }
}
DLLEXPORT void _crt_multi_aug(int* Params, float* X_Values, float* X_Rows, float* X_Cols, float* Phi, float* Theta, float* XVK, float* XKJ, curandStateXORWOW_t* status){
    const int V = Params[0];  // the vocabulary length
    const int K = Params[1];  // the number of topics
    const int J = Params[2];  // the number of documents
    const int N = Params[3];  // the number of non-zero elements
//     printf("V:%d, K:%d, J:%d, N:%d\n", V, K, J, N); //

    // input variables
    float* X_Values_device, * X_Rows_device, * X_Cols_device, * Phi_device, * Theta_device;
    cudaMalloc((void**)&X_Values_device, N * sizeof(float));
    cudaMemcpy(X_Values_device, X_Values, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&X_Rows_device, N * sizeof(float));
    cudaMemcpy(X_Rows_device, X_Rows, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&X_Cols_device, N * sizeof(float));
    cudaMemcpy(X_Cols_device, X_Cols, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&Phi_device, V*K * sizeof(float));
    cudaMemcpy(Phi_device, Phi, V*K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&Theta_device, K*J * sizeof(float));
    cudaMemcpy(Theta_device, Theta, K*J * sizeof(float), cudaMemcpyHostToDevice);

    // output variables
    float* XVK_device, * XKJ_device;
    cudaMalloc((void**)&XVK_device, V*K * sizeof(float));
    cudaMemcpy(XVK_device, XVK, V*K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&XKJ_device, K*J * sizeof(float));
    cudaMemcpy(XKJ_device, XKJ, K*J * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(size_t(ceil(float(N) / float(nThreads)))), block(nThreads);
//     printf("grid:%zd, block:%zd\n", size_t(ceil(float(N) / float(nThreads_gamma))), nThreads_gamma);
   _rand_crt_multi_aug <<<grid, block>>> (V, K, J, N, X_Values_device, X_Rows_device, X_Cols_device, Phi_device, Theta_device, XVK_device, XKJ_device, status);

    cudaMemcpy(XVK, XVK_device, V*K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(XKJ, XKJ_device, K*J * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(X_Values_device);
    cudaFree(X_Rows_device);
    cudaFree(X_Cols_device);
    cudaFree(Phi_device);
    cudaFree(Theta_device);
    cudaFree(XVK_device);
    cudaFree(XKJ_device);
}

// ------------------------------------------------sample multinomial augmentation ------------------------------------------
__global__ void _rand_multi_aug(int V, int K, int J, int N, float* X_Values, float* X_Rows, float* X_Cols, float* Phi, float* Theta, float* XVK, float* XKJ, curandStateXORWOW_t* status) {//根据输入采样，得到sst->output

    size_t idx = threadIdx.x + blockDim.x * blockIdx.x; //grid(nElm/(32*4)) block(32*4)  均1维  range nElm
    if (idx < N) {
//         printf("V:%d, K:%d, J:%d, N:%d \n", V, K, J, N);
        size_t state_idx = idx % nStatus;
        size_t word_value = X_Values[idx];
        size_t word_row = X_Rows[idx];
        size_t word_col = X_Cols[idx];

        // calculate sum(Phi[v, :]*Theta[:, j])
        float prob_sum = 0.0;
        for (int k=0; k<K; k++){
            prob_sum += Phi[word_row*K + k] * Theta[k*J + word_col];
        }

        // augment x to a count vector
        float prob = 0.0;
        size_t topic_index = 0;
        for (int token=0; token<word_value; token++){
            float prob_threshold = curand_uniform(&status[state_idx]) * prob_sum; // It may return from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded.
            float prob_cumulated = 0.0;

            for(int k=0; k<K; k++){
                prob = Phi[word_row*K + k] * Theta[k*J + word_col];
                if (prob_cumulated + prob >=prob_threshold){
                    topic_index = k;
                    break;
                }
                prob_cumulated += prob;
            }

            atomicAdd(&XVK[word_row*K + topic_index], 1.0);
            atomicAdd(&XKJ[topic_index*J + word_col], 1.0);
        }
    }
}
DLLEXPORT void _multi_aug(int* Params, float* X_Values, float* X_Rows, float* X_Cols, float* Phi, float* Theta, float* XVK, float* XKJ, curandStateXORWOW_t* status){

    const size_t V = Params[0];  // the vocabulary length
    const size_t K = Params[1];  // the number of topics
    const size_t J = Params[2];  // the number of documents
    const size_t N = Params[3];  // the number of non-zero elements
//     printf("V:%d, K:%d, J:%d, N:%d \n", V, K, J, N); //

    // input variables
    float* X_Values_device, * X_Rows_device, * X_Cols_device, * Phi_device, * Theta_device;
    cudaMalloc((void**)&X_Values_device, N * sizeof(float));
    cudaMemcpy(X_Values_device, X_Values, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&X_Rows_device, N * sizeof(float));
    cudaMemcpy(X_Rows_device, X_Rows, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&X_Cols_device, N * sizeof(float));
    cudaMemcpy(X_Cols_device, X_Cols, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&Phi_device, V*K * sizeof(float));
    cudaMemcpy(Phi_device, Phi, V*K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&Theta_device, K*J * sizeof(float));
    cudaMemcpy(Theta_device, Theta, K*J * sizeof(float), cudaMemcpyHostToDevice);

    // output variables
    float* XVK_device, * XKJ_device;
    cudaMalloc((void**)&XVK_device, V*K * sizeof(float));
    cudaMemcpy(XVK_device, XVK, V*K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&XKJ_device, K*J * sizeof(float));
    cudaMemcpy(XKJ_device, XKJ, K*J * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(size_t(ceil(float(N) / float(nThreads)))), block(nThreads);
//     printf("grid:%zd, block:%zd\n", size_t(ceil(float(N) / float(nThreads_gamma))), nThreads_gamma);
   _rand_multi_aug <<<grid, block>>> (V, K, J, N, X_Values_device, X_Rows_device, X_Cols_device, Phi_device, Theta_device, XVK_device, XKJ_device, status);

    cudaMemcpy(XVK, XVK_device, V*K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(XKJ, XKJ_device, K*J * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(X_Values_device);
    cudaFree(X_Rows_device);
    cudaFree(X_Cols_device);
    cudaFree(Phi_device);
    cudaFree(Theta_device);
    cudaFree(XVK_device);
    cudaFree(XKJ_device);
}

// ------------------------------------------------sample convolutional multinomial augmentation ------------------------------------------
__global__ void _rand_conv_multi_aug(int K0, int K1, int K1_S1, int K1_S2, int K1_S3, int K1_S4, int N, float* X_Rows, float* X_Cols, float* X_Pages, float* X_Values, float* D1_k1, float* W1_nk1, float* D1_k1_Aug, float* W1_nk1_Aug, curandStateXORWOW_t* status) {//根据输入采样，得到sst->output

    size_t idx = threadIdx.x + blockDim.x * blockIdx.x; //grid(nElm/(32*4)) block(32*4)  均1维  range nElm
    if (idx < N) {
        size_t state_idx = idx % nStatus;

        int word_row = int(X_Rows[idx]);
        int word_col = int(X_Cols[idx]);
        int word_page = int(X_Pages[idx]);
        int word_value = int(X_Values[idx]);

//         printf("row:%d, col:%d, page:%d, value:%d \n", word_row, word_col, word_page, word_value);

        int word_row_left = 0;
        int word_row_right = 0;
        int word_col_left = 0;
        int word_col_right = 0;

        // word_row
        if ((word_row - K1_S3 + 1) > 0)
            word_row_left = word_row - K1_S3 + 1;
        else
            word_row_left = 0;

        if (word_row > K1_S1 - 1)
            word_row_right = K1_S1 - 1;
        else
            word_row_right = word_row;

        int word_row_size = word_row_right - word_row_left + 1;

        // word_col
        if ((word_col - K1_S4 +1)>0)
            word_col_left = word_col - K1_S4 + 1;
        else
            word_col_left = 0;

        if (word_col > K1_S2 -1)
            word_col_right = K1_S2 - 1;
        else
            word_col_right = word_col;

        int word_col_size = word_col_right - word_col_left + 1;

        // N*K0*K1_V1*K1_V2 => N*K1*K1_S1*K1_S2, K0*K1*K1_S3*K1_S4
        float prob_sum = 0.0;

        for (int i=0; i<K1; i++){
            for (int k=0; k<word_row_size; k++){
                for (int j=0; j<word_col_size; j++){
                    int row_idx = word_row_left + k;
                    int col_idx = word_col_left + j;
                    int W1_idx = (word_page) * K1 * K1_S1 * K1_S2 + (i) * K1_S1 * K1_S2 + row_idx * K1_S2 + (col_idx);
                    int D1_idx = (i) * K1_S3 * K1_S4 + (word_row - row_idx) * K1_S4 + (word_col - col_idx);

                    prob_sum += W1_nk1[W1_idx] * D1_k1[D1_idx];
                }
            }
        }
        if (prob_sum > 0.0){
            for (int token=0; token<word_value; token++){
                float prob_cumulated = 0.0;
                float prob_threshold = curand_uniform(&status[state_idx]) * prob_sum;

                int Stop_Flag = 0;

                for (int i=0; i<K1; i++){
                    for (int k=0; k<word_row_size; k++){
                        for (int j=0; j<word_col_size; j++){
                            int row_idx = word_row_left + k;
                            int col_idx = word_col_left + j;
                            int W1_idx = (word_page) * K1 * K1_S1 * K1_S2 + (i) * K1_S1 * K1_S2 + row_idx * K1_S2 + (col_idx);
                            int D1_idx = (i) * K1_S3 * K1_S4 + (word_row - row_idx) * K1_S4 + (word_col - col_idx);

                            prob_cumulated += W1_nk1[W1_idx] * D1_k1[D1_idx];

                            if (prob_cumulated >= prob_threshold){
                                atomicAdd(&D1_k1_Aug[D1_idx], 1.0);
                                atomicAdd(&W1_nk1_Aug[W1_idx], 1.0);
                                Stop_Flag = 1;
                            }
                            if (Stop_Flag == 1)
                                break;
                        }
                        if (Stop_Flag == 1)
                            break;
                    }
                    if (Stop_Flag == 1)
                        break;
                }
            }
        }
    }
}
DLLEXPORT void _conv_multi_aug(int* Params, float* X_Rows, float* X_Cols, float* X_Pages, float* X_Values, float* D1_k1, float* W1_nk1, float* D1_k1_Aug, float* W1_nk1_Aug, curandStateXORWOW_t* status){

    const size_t K0 = Params[0];
    const size_t K1 = Params[1];
    const size_t K1_S1 = Params[2];
    const size_t K1_S2 = Params[3];
    const size_t K1_S3 = Params[4];
    const size_t K1_S4 = Params[5];
    const size_t N = Params[6];  // the number of non-zeros elements
    const size_t Num_Doc = Params[7];

//     printf("%d, %d, %d, %d, %d, %d, %d, %d\n", K0, K1, K1_S1, K1_S2, K1_S3, K1_S4, N, Num_Doc);

    // input variables
    float* X_Rows_device, * X_Cols_device, * X_Pages_device, * X_Values_device, * D1_k1_device, * W1_nk1_device;
    cudaMalloc((void**)&X_Rows_device, N * sizeof(float));
    cudaMemcpy(X_Rows_device, X_Rows, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&X_Cols_device, N * sizeof(float));
    cudaMemcpy(X_Cols_device, X_Cols, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&X_Pages_device, N * sizeof(float));
    cudaMemcpy(X_Pages_device, X_Pages, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&X_Values_device, N * sizeof(float));
    cudaMemcpy(X_Values_device, X_Values, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&D1_k1_device, K0*K1*K1_S3*K1_S4 * sizeof(float));
    cudaMemcpy(D1_k1_device, D1_k1, K0*K1*K1_S3*K1_S4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&W1_nk1_device, Num_Doc*K1*K1_S1*K1_S2 * sizeof(float));
    cudaMemcpy(W1_nk1_device, W1_nk1, Num_Doc*K1*K1_S1*K1_S2 * sizeof(float), cudaMemcpyHostToDevice);

    // output variables
    float* D1_k1_Aug_device, * W1_nk1_Aug_device;
    cudaMalloc((void**)&D1_k1_Aug_device, K0*K1*K1_S3*K1_S4 * sizeof(float));
    cudaMemcpy(D1_k1_Aug_device, D1_k1_Aug, K0*K1*K1_S3*K1_S4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&W1_nk1_Aug_device, Num_Doc*K1*K1_S1*K1_S2 * sizeof(float));
    cudaMemcpy(W1_nk1_Aug_device, W1_nk1_Aug, Num_Doc*K1*K1_S1*K1_S2 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(size_t(ceil(float(N) / float(nThreads)))), block(nThreads);
//     printf("grid:%zd, block:%zd\n", size_t(ceil(float(N) / float(nThreads_gamma))), nThreads_gamma);
    _rand_conv_multi_aug <<<grid, block>>> (K0, K1, K1_S1, K1_S2, K1_S3, K1_S4, N, X_Rows_device, X_Cols_device, X_Pages_device, X_Values_device, D1_k1_device, W1_nk1_device, D1_k1_Aug_device, W1_nk1_Aug_device, status);
    cudaDeviceSynchronize();

    cudaMemcpy(D1_k1_Aug, D1_k1_Aug_device, K0*K1*K1_S3*K1_S4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(W1_nk1_Aug, W1_nk1_Aug_device, Num_Doc*K1*K1_S1*K1_S2 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(X_Rows_device);
    cudaFree(X_Cols_device);
    cudaFree(X_Pages_device);
    cudaFree(X_Values_device);
    cudaFree(D1_k1_device);
    cudaFree(W1_nk1_device);
    cudaFree(D1_k1_Aug_device);
    cudaFree(W1_nk1_Aug_device);
}