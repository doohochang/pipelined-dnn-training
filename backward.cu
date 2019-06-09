#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "model.cuh"
#include "activation.cuh"

#define THREAD_NUM 256

__global__ void sigmoid_derivative(float *upper_grads, float *upper_values, unsigned int upper_size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < upper_size)
        upper_grads[index] *= upper_values[index]*(1.0f - upper_values[index]);
}

__global__ void relu_derivative(float *upper_grads, float *upper_values, unsigned int upper_size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < upper_size)
        if (upper_values[index] == 0)
            upper_grads[index] = 0;
}

// learning_weight = -(learning_rate / batch_size)

void run_backward_step(
    cublasHandle_t handle, 
    cudaStream_t stream, 
    Activation activation, 
    float *learning_weight,
    float *upper_grads, unsigned int batch_size, unsigned int upper_size,
    float *upper_values, float *lower_values,
    float *weight_matrix, float *bias,
    float *lower_grads, unsigned int lower_size,
    const float *ones, const float *zero
) {
    // TODO: Implement activation derivation
    switch (activation){
        case ACTIVATION_LINEAR:
            break;
        case ACTIVATION_SIGMOID:
            // derivative of sigmoid sig'(x) = sig(x) * (1 - sig(x))
            //upper_grads = upper_grads * upper_values * (1 - upper_values)
            sigmoid_derivative<<<(batch_size * upper_size + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM, 0, stream>>>(upper_grads, upper_values, upper_size * batch_size);
            break;
        case ACTIVATION_RELU:
            ////upper_grads = upper_grads * (upper_values > 0 ? 1 : 0)
            relu_derivative<<<(batch_size * upper_size + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM, 0, stream>>>(upper_grads, upper_values, upper_size * batch_size);
            break;
    }
    // Update biases
    cublasSgemv(
        handle, CUBLAS_OP_T,
        batch_size, upper_size,
        learning_weight,
        upper_grads, batch_size,
        ones, 1,
        ones,
        bias, 1
    );

    // Update lower grads
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        batch_size, lower_size, upper_size,
        ones,
        upper_grads, batch_size,
        weight_matrix, lower_size,
        zero,
        lower_grads, batch_size);
    
    // Update weight matrix
    cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        lower_size, upper_size, batch_size,
        ones,
        lower_values, batch_size,
        upper_grads, batch_size,
        learning_weight,
        weight_matrix, lower_size);
     
}

void run_backward(SubModel *submodel, int number_of_upper_nodes, float *upper_values, float *upper_grads, unsigned int batch_size, float *learning_weight, cudaStream_t stream, float *one, float* zero) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    
    int last = submodel->spec.number_of_layers - 1;
    
    run_backward_step(
        handle, stream, submodel->spec.layers[last].activation,
        learning_weight,
        upper_grads, batch_size, number_of_upper_nodes,
        upper_values, submodel->forward_values[last],
        submodel->weight_matrices[last], submodel->biases[last],
        submodel->gradients[last], submodel->spec.layers[last].number_of_nodes,
        one, zero
    );
    if(last > 0){
        for (int i = submodel->spec.number_of_layers - 2; i >= 0 ; i--){
            run_backward_step(
                handle, stream, submodel->spec.layers[i].activation,
                learning_weight,
                submodel->gradients[i + 1], batch_size, submodel->spec.layers[i + 1].number_of_nodes,
                submodel->forward_values[i + 1], submodel->forward_values[i],
                submodel->weight_matrices[i], submodel->biases[i],
                submodel->gradients[i], submodel->spec.layers[i].number_of_nodes,
                one, zero
            );
        }
    } 
}
