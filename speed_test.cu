#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>


#define CHECK_CUDA(func) {                                                      \
    cudaError_t status = (func);                                                \
    if (status != cudaSuccess) {                                                \
        std::cerr << "CUDA Error at line " << __LINE__ << ": "                  \
                  << cudaGetErrorString(status) << std::endl;                   \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

#define CHECK_CUBLAS(func) {                                                    \
    cublasStatus_t status = (func);                                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                                      \
        std::cerr << "cuBLAS Error at line " << __LINE__ << ": "                \
                  << status << std::endl;                                       \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

#define CHECK_CUSPARSE(func) {                                                  \
    cusparseStatus_t status = (func);                                           \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                    \
        std::cerr << "cuSPARSE Error at line " << __LINE__ << ": "              \
                  << status << std::endl;                                       \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}



void runSpmmTest(int A_ell_blocksize, int pattern){
    // -------------------------------------------------------------------
    // Parameters for matrices:
    constexpr int A_num_rows = 4096;
    constexpr int A_num_cols = 4096;
    constexpr int B_num_rows = 4096;
    constexpr int B_num_cols = 2048;
    constexpr int ldc = A_num_rows;       // leading dimension of C
    constexpr int ldb = B_num_rows;       // leading dimension of B
    const int C_size = ldc * B_num_cols;  // number of elements in result

    // -------------------------------------------------------------------
    // Create a sparse matrix A in blocked ELL format.
    // For a 4096x4096 matrix, there are 4096/4 = 1024 blocks per dimension.
    const int mb = A_num_rows / A_ell_blocksize;  // number of block rows (1024)
    const int nb = A_num_cols / A_ell_blocksize;  // number of block columns (1024)
    const int A_num_blocks = mb * nb;
    // For simplicity, we allocate storage for one block per dense block position.
    // In a typical ELL format, you might overallocate; here we assume each block is present.
    int *hA_columns = (int*)malloc(A_num_blocks * sizeof(int));
    for (int i = 0; i < mb; i++) {
        for (int j = 0; j < nb; j++) {
            hA_columns[i * nb + j] = j;
        }
    }
    // Each block is 4x4.
    __half *hA_values = (__half*)malloc(A_num_blocks * A_ell_blocksize * A_ell_blocksize * sizeof(__half));
    srand((unsigned)time(NULL));
    for (int i = 0; i < 4096 * 4096; i += pattern) {
        // Create an array with indices 0, 1, 2, 3.
        std::vector<int> indices(pattern);
        std::iota(indices.begin(), indices.end(), 0);
        // Shuffle indices using the Fisher–Yates algorithm.
        for (int j = 0; j < pattern; j++) {
            int rdx = j + rand() % (pattern - j);
            int temp = indices[j];
            indices[j] = indices[rdx];
            indices[rdx] = temp;
        }
        // Use the first two indices in the shuffled array as the positions for 1.
        int pos1 = indices[0];
        int pos2 = indices[1];
        
        // For each of the 4 positions, set to 1 if its index is pos1 or pos2, else 0.
        for (int j = 0; j < pattern; j++) {
            if (j == pos1 || j == pos2)
                hA_values[i + j] = __float2half(1.0f);
            else
                hA_values[i + j] = __float2half(0.0f);
        }
    }
    
    // Optional: Print the first 16 elements (4 groups of 4) to verify the pattern.
    //printf("First 16 elements (4 groups of 4):\n");
    //for (int i = 0; i < 32; i++) {
    //    printf("%8.2f ", __half2float(hA_values[i]));
    //}

    // Create dense matrix B with random values.
    const int B_size = ldb * B_num_cols;
    __half *hB = (__half*)malloc(B_size * sizeof(__half));
    for (int i = 0; i < B_size; i++) {
        hB[i] = __float2half(static_cast<float>(1));
    }
    
    // -------------------------------------------------------------------
    // Allocate device memory for the sparse matrix multiplication.
    int *dA_columns;
    __half *dA_values, *dB, *dC;
    CHECK_CUDA(cudaMalloc((void**)&dA_columns, A_num_blocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_values, A_num_blocks * A_ell_blocksize * A_ell_blocksize * sizeof(__half)));
    CHECK_CUDA(cudaMalloc((void**)&dB, B_size * sizeof(__half)));
    CHECK_CUDA(cudaMalloc((void**)&dC, C_size * sizeof(__half)));

    // Copy the host data to device.
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_num_blocks * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_num_blocks * A_ell_blocksize * A_ell_blocksize * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, B_size * sizeof(__half), cudaMemcpyHostToDevice));
    // Initialize result matrix dC to zero.
    CHECK_CUDA(cudaMemset(dC, 0, C_size * sizeof(__half)));

    // -------------------------------------------------------------------
    // Create cuSPARSE descriptors and perform the sparse SpMM.
    cusparseHandle_t cusparseHandle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // Create the sparse matrix descriptor in blocked ELL format.
    CHECK_CUSPARSE(cusparseCreateBlockedEll(&matA,
                                             A_num_rows, A_num_cols,
                                             A_ell_blocksize,
                                             A_num_cols,  // maximum blocks per row (here each row block is full)
                                             dA_columns,
                                             dA_values,
                                             CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO,
                                             CUDA_R_16F));

    // Create dense matrix descriptors for B and C.
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_16F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_16F, CUSPARSE_ORDER_COL));

    // SpMM parameters.
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           matA, matB,
                                           &beta,
                                           matC,
                                           CUDA_R_16F,
                                           CUSPARSE_SPMM_ALG_DEFAULT,
                                           &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    //warm-up
    CHECK_CUSPARSE(cusparseSpMM(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA, matB,
        &beta,
        matC,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        dBuffer));
    CHECK_CUDA(cudaDeviceSynchronize()); // Ensure the warm-up is finished

    cudaEvent_t startSpMM, stopSpMM;
    CHECK_CUDA(cudaEventCreate(&startSpMM));
    CHECK_CUDA(cudaEventCreate(&stopSpMM));
    CHECK_CUDA(cudaEventRecord(startSpMM, 0));

    // Execute the sparse SpMM.
    CHECK_CUSPARSE(cusparseSpMM(cusparseHandle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                matA, matB,
                                &beta,
                                matC,
                                CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT,
                                dBuffer));

    // Clean up the cuSPARSE descriptors.
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));

    CHECK_CUDA(cudaEventRecord(stopSpMM, 0));
    CHECK_CUDA(cudaEventSynchronize(stopSpMM));
    float spmmTime = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&spmmTime, startSpMM, stopSpMM));
    printf("Block size: %3d, Pattern: %3d, cuSPARSE SpMM execution time: %f ms\n", A_ell_blocksize, pattern, spmmTime);

    // Copy the sparse multiplication result from device to host.
    __half *hC_sparse = (__half*)malloc(C_size * sizeof(__half));
    CHECK_CUDA(cudaMemcpy(hC_sparse, dC, C_size * sizeof(__half), cudaMemcpyDeviceToHost));
    //printf("First 16 elements (4 groups of 4):\n");
    //for (int i = 0; i < 32; i++) {
    //    printf("%8.2f ", __half2float(hC_sparse[i]));
    //}
    if(A_ell_blocksize == 1024 && pattern == 1024){
    // Allocate device memory for the dense version of A and for the reference result.
    __half *dA_dense, *dC_dense;
    CHECK_CUDA(cudaMalloc(&dA_dense, A_num_rows * A_num_cols * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_dense, C_size * sizeof(__half)));

    // Copy the dense A from host to device.
    CHECK_CUDA(cudaMemcpy(dA_dense, hA_values, A_num_rows * A_num_cols * sizeof(__half), cudaMemcpyHostToDevice));
    // Initialize dC_dense to zero.
    CHECK_CUDA(cudaMemset(dC_dense, 0, C_size * sizeof(__half)));

    // Create a cuBLAS handle.
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
    __half h_alpha = __float2half(1.0f);
    __half h_beta  = __float2half(0.0f);
    CHECK_CUBLAS(cublasGemmEx(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        A_num_rows, B_num_cols, A_num_cols,
        &h_alpha,
        dA_dense, CUDA_R_16F, A_num_cols,
        dB,       CUDA_R_16F, A_num_cols,
        &h_beta,
        dC_dense, CUDA_R_16F, A_num_rows,
        CUDA_R_16F,
        CUBLAS_GEMM_DEFAULT));
    CHECK_CUDA(cudaDeviceSynchronize()); 
    // Perform dense GEMM: dC_dense = dA_dense * dB.
    cudaEvent_t startDense, stopDense;
    CHECK_CUDA(cudaEventCreate(&startDense));
    CHECK_CUDA(cudaEventCreate(&stopDense));
    CHECK_CUDA(cudaEventRecord(startDense));
    CHECK_CUBLAS(cublasGemmEx(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
                              A_num_rows, B_num_cols, A_num_cols,
                              &h_alpha,
                              dA_dense, CUDA_R_16F, A_num_cols,
                              dB, CUDA_R_16F, A_num_cols,
                              &h_beta,
                              dC_dense, CUDA_R_16F, A_num_rows,
                              CUDA_R_16F,
                              CUBLAS_GEMM_DEFAULT));
    CHECK_CUDA(cudaEventRecord(stopDense));
    CHECK_CUDA(cudaEventSynchronize(stopDense));
    float denseTime = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&denseTime, startDense, stopDense));
   printf("Dense GEMM execution time for reference: %f ms\n", denseTime);

    // Copy the dense result from device to host.
    __half *hC_dense = (__half*)malloc(C_size * sizeof(__half));
    CHECK_CUDA(cudaMemcpy(hC_dense, dC_dense, C_size * sizeof(__half), cudaMemcpyDeviceToHost));


    free(hC_dense);

    CHECK_CUDA(cudaFree(dA_dense));
    CHECK_CUDA(cudaFree(dC_dense));

    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaEventDestroy(startDense));
    CHECK_CUDA(cudaEventDestroy(stopDense));
}
    // printf("First 16 elements (4 groups of 4):\n");
    // for (int i = 0; i < 32; i++) {
    //     printf("%8.2f ", __half2float(hC_dense[i]));
    // }
    // -------------------------------------------------------------------
    // Clean up all allocated resources.
    free(hA_columns);
    free(hA_values);
    free(hB);
    free(hC_sparse);
    

    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(dA_columns));
    CHECK_CUDA(cudaFree(dA_values));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    

    

    CHECK_CUDA(cudaEventDestroy(startSpMM));
    CHECK_CUDA(cudaEventDestroy(stopSpMM));
    

}



int main(){
    int blockSizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int patternSizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
    //runSpmmTest(4,64);
    
    
     for(int i = 0; i < 9; i++){
         for(int j = 0; j < 9; j++){
             runSpmmTest(blockSizes[i], patternSizes[j]);
         }
     }
    return 0;
}
