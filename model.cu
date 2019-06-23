#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>
#include "model.cuh"
#include <stdio.h>

#define THREAD_NUM 256

__global__ void scale_values(float *num, size_t size, float abs_max)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		num[idx] = (abs_max + abs_max) * num[idx] - abs_max;
}

void alloc_rand_values(float **dev_ptr, size_t size, curandGenerator_t *generator, float abs_max) {
    cudaMalloc(dev_ptr, size);
    curandGenerateUniform(*generator, *dev_ptr, size);
    scale_values<<<(size + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(*dev_ptr, size, abs_max);
}

void alloc_zero_values(float **dev_ptr, size_t size) {
    cudaMalloc(dev_ptr, size);
    cudaMemset(*dev_ptr, 0, size);
}

SubModel::SubModel(SubModelSpec spec) {
    this->spec = spec;

    // Initialize Curand generator
    curandGenerator_t randGen;
	curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(randGen, (unsigned long long)clock());

    // Alloc weight matrices
    //cudaMalloc(&(this->weight_matrices), sizeof(float *) * spec.number_of_layers); //이렇게하면 alloc_rand에서 오류발생

    cudaMalloc(&(this->weight_matrices), sizeof(float *) * spec.number_of_layers);
    this->weight_matrices_buffer = (float**)malloc(sizeof(float *) * spec.number_of_layers);

    alloc_rand_values(
        &(this->weight_matrices_buffer[0]),
        sizeof(float) * spec.number_of_input_nodes * spec.layers[0].number_of_nodes,
        &randGen,
        sqrt(6.0f / (spec.number_of_input_nodes + spec.layers[0].number_of_nodes))
    );
  
    for (int i = 1; i < spec.number_of_layers; i++) {
        alloc_rand_values(
            &(this->weight_matrices_buffer[i]),
            sizeof(float) * spec.layers[i - 1].number_of_nodes * spec.layers[i].number_of_nodes,
            &randGen,
            sqrt(6.0f / (spec.layers[i - 1].number_of_nodes + spec.layers[i].number_of_nodes))
        );
    }
    
    cudaMemcpy(this->weight_matrices, this->weight_matrices_buffer, sizeof(float *) * spec.number_of_layers, cudaMemcpyHostToDevice);

    // Alloc biases & forward values
    /* //이렇게하면 오류발생
    cudaMalloc(&(this->biases), sizeof(float *) * spec.number_of_layers);
    cudaMalloc(&(this->forward_values), sizeof(float *) * spec.number_of_layers);
    cudaMalloc(&(this->gradients), sizeof(float *) * spec.number_of_layers);*/

    this->biases_buffer = (float**)malloc(sizeof(float *) * spec.number_of_layers);
    this->forward_values_buffer = (float**)malloc(sizeof(float *) * spec.number_of_layers);
    this->gradients_buffer = (float**)malloc(sizeof(float *) * spec.number_of_layers);
    
    cudaMalloc(&(this->biases), sizeof(float *) * spec.number_of_layers);
    cudaMalloc(&(this->forward_values), sizeof(float *) * spec.number_of_layers);
    cudaMalloc(&(this->gradients), sizeof(float *) * spec.number_of_layers);
    
    for (int i = 0; i < spec.number_of_layers; i++) {
        alloc_zero_values(&(this->biases_buffer[i]), sizeof(float) * spec.layers[i].number_of_nodes);
        cudaMalloc(&(this->forward_values_buffer[i]), sizeof(float) * spec.layers[i].number_of_nodes);
        cudaMalloc(&(this->gradients_buffer[i]), sizeof(float) * spec.layers[i].number_of_nodes);
    }
    
    cudaMemcpy(this->biases, this->biases_buffer, sizeof(float *) * spec.number_of_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->forward_values, this->forward_values_buffer, sizeof(float *) * spec.number_of_layers, cudaMemcpyHostToDevice);
    cudaMemcpy(this->gradients, this->gradients_buffer, sizeof(float *) * spec.number_of_layers, cudaMemcpyHostToDevice);
}

SubModel::~SubModel() {
    
    for (int i = 0; i < spec.number_of_layers; i++) {
        cudaFree(this->weight_matrices_buffer[i]);
        cudaFree(this->biases_buffer[i]);
        cudaFree(this->forward_values_buffer[i]);
        cudaFree(this->gradients_buffer[i]);
    }

    cudaFree(this->weight_matrices);
    cudaFree(this->biases);
    cudaFree(this->forward_values);
    cudaFree(this->gradients);
}
