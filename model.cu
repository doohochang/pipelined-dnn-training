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

SubModel::SubModel(SubModelSpec spec, int batch_size) {
    this->spec = spec;

    // Initialize Curand generator
    curandGenerator_t randGen;
	curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(randGen, (unsigned long long)clock());

    // Alloc weight matrices
    //cudaMalloc(&(this->weight_matrices), sizeof(float *) * spec.number_of_layers); //이렇게하면 alloc_rand에서 오류발생
    
    this->weight_matrices_size = spec.number_of_input_nodes * spec.layers[0].number_of_nodes;
    
    this->biases_size = spec.layers[0].number_of_nodes;
    this->forward_values_size = batch_size * spec.layers[0].number_of_nodes;
    

    for(int i = 1; i < spec.number_of_layers; i++){
        this->weight_matrices_size += spec.layers[i-1].number_of_nodes * spec.layers[i].number_of_nodes;
        this->biases_size += spec.layers[i].number_of_nodes;
        this->forward_values_size += batch_size * spec.layers[i].number_of_nodes;
    }
    
    this->gradients_size = this->weight_matrices_size;
    
    alloc_rand_values(
        &(this->weight_matrices),
        sizeof(float) * this->weight_matrices_size,
        &randGen,
        sqrt(6.0f / (spec.layers[0].number_of_nodes + spec.layers[1].number_of_nodes))
    );

    cudaMalloc(&(this->forward_values), sizeof(float) * this->forward_values_size);
    cudaMalloc(&(this->gradients), sizeof(float) * this->gradients_size);
    
    alloc_zero_values(&(this->biases), sizeof(float) * this->biases_size);

}

SubModel::~SubModel() {
    cudaFree(this->weight_matrices);
    cudaFree(this->biases);
    cudaFree(this->forward_values);
    cudaFree(this->gradients);
}
