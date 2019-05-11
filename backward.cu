#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "model.cuh"
#include "activation.cuh"

#define THREAD_NUM 256

void run_backward_step(
    cublasHandle_t handle, 
    cudaStream_t stream, 
    Activation activation, 
    float *minus_learning_rate,
    float *upper_grads, unsigned int batch_size, unsigned int upper_size,
    float *upper_values, float *lower_values,
    float *weight_matrix, float *bias,
    float *lower_grads, unsigned int lower_size,
    const float *ones
) {
    // TODO: Implement activation derivation
    // Update biases
    cublasSgemv(
        handle, CUBLAS_OP_T,
        batch_size, upper_size,
        minus_learning_rate,
        upper_grads, batch_size,
        ones, 1,
        ones,
        bias, 1
    );

    // Update weight matrix
    
}

void run_backward(SubModel *submodel, float *input_values, float *upper_grads, unsigned int batch_size, float *minus_learning_rate, cudaStream_t stream) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    
}

