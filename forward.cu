#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "forward.cuh"
#include "model.cuh"
#include "activation.cuh"

#define THREAD_NUM 256

__global__ void times(float *input, unsigned int input_size, float *output, unsigned int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < n * input_size)
        output[index] = input[index % input_size];
}

void run_forward_step(
    cublasHandle_t handle,
    cudaStream_t stream,
    Activation activation,
    float *input, unsigned int batch_size, unsigned int input_size,
    float *weight_matrix, float *bias,
    float *output, unsigned int output_size,
    float *one
) {
    // weight_matrix: input_size * output_size
    // bias: output_size
 
    times<<<(batch_size * output_size + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM, 0, stream>>>(bias, output_size, output, batch_size);

    // input(batch_size, input_size) * weight_matrix(input_size, output_size) = output(batch_size, output_size)
    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        batch_size, input_size, output_size,
        one,
        input, batch_size,
        weight_matrix, input_size,
        one,
        output, batch_size
    );

    switch (activation) {
        case ACTIVATION_LINEAR:
            break;
        case ACTIVATION_SIGMOID:
            sigmoid_kernel<<<(batch_size * output_size + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM, 0, stream>>>(output, batch_size * output_size);
            break;
        case ACTIVATION_RELU:
            relu_kernel<<<(batch_size * output_size + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM, 0, stream>>>(output, batch_size * output_size);
            break;
    }
}

void run_forward(SubModel *submodel, float *input, unsigned int batch_size, cudaStream_t stream, float *one) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    run_forward_step(
        handle, stream, submodel->spec.layers[0].activation,
        input, batch_size, submodel->spec.number_of_input_nodes,
        submodel->weight_matrices[0], submodel->biases[0],
        submodel->forward_values[0], submodel->spec.layers[0].number_of_nodes,
        one
    );

    for (int i = 1; i < submodel->spec.number_of_layers; i++){
        run_forward_step(
            handle, stream, submodel->spec.layers[i].activation,
            submodel->forward_values[i - 1], batch_size, submodel->spec.layers[i - 1].number_of_nodes,
            submodel->weight_matrices[i], submodel->biases[i],
            submodel->forward_values[i], submodel->spec.layers[i].number_of_nodes,
            one
        );
    }
}

__global__ void exp_kernel(float *array, unsigned int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		array[index] = exp(array[index]);
}

__global__ void set_value(float value, float *array, unsigned int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        array[index] = value;
}

__global__ void divide_by_vector(float *matrix, float *vector, unsigned int row, unsigned int col) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < row * col)
        matrix[index] /= vector[index / col];
}

__global__ void minus_one(float *matrix, unsigned int *indices, unsigned int row, unsigned int col) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < row)
        matrix[index * col + indices[index]] -= 1;
}

__global__ void pick_minus_log_ps(float *matrix, float *minus_log_ps, unsigned int *indices, unsigned int row, unsigned int col) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < row)
        minus_log_ps[index] = -log(matrix[index * col + indices[index]]);
}

void run_softmax_cross_entropy(float *scores, unsigned int batch_size, unsigned int number_of_scores, unsigned int *answers, float *loss, float *grad_scores, cudaStream_t stream, const float *ones, float *batch_size_buffer) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    cudaMemcpyAsync(grad_scores, scores, sizeof(float) * batch_size * number_of_scores, cudaMemcpyDeviceToDevice, stream);
    exp_kernel<<<(batch_size * number_of_scores + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM, 0, stream>>>(grad_scores, batch_size * number_of_scores);

    /*
        float *ones;
        cudaMalloc(&ones, sizeof(float) * number_of_scores);
        set_value<<<(number_of_scores + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM, 0, stream>>>(1.0f, ones, number_of_scores);
    */

    set_value<<<(batch_size + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM, 0, stream>>>(0, batch_size_buffer, batch_size);
    
    cublasSgemv(
        handle, CUBLAS_OP_N,
        batch_size, number_of_scores,
        ones,
        grad_scores, batch_size,
        ones, 1,
        ones,
        batch_size_buffer, 1
    );

    divide_by_vector<<<(batch_size * number_of_scores + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM, 0, stream>>>(grad_scores, batch_size_buffer, batch_size, number_of_scores);

    pick_minus_log_ps<<<(batch_size + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM, 0, stream>>>(grad_scores, batch_size_buffer, answers, batch_size, number_of_scores);
    
    cublasSdot(handle, batch_size, batch_size_buffer, 1, ones, 1, loss);

    minus_one<<<(batch_size + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM, 0, stream>>>(grad_scores, answers, batch_size, number_of_scores);
}

void run_output_layer(OutputLayer layer, float *input, unsigned int batch_size, void *answers, float *loss, float *grad_input, cudaStream_t stream, float *ones, float *batch_size_buffer) {
    switch (layer.loss) {
        case LOSS_SOFTMAX_CROSS_ENTROPY:
            run_softmax_cross_entropy(input, batch_size, layer.number_of_nodes, (unsigned int *)answers, loss, grad_input, stream, ones, batch_size_buffer);
            break;
    }
}

