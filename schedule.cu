#include <cuda_runtime.h>

#include "model.cuh"

void schedule(int num_devices, SubModelSpec *submodel_specs) {
    SubModel **submodels;

    // Initialize submodels
    submodels = (SubModel *)malloc(sizeof(SubModel *) * num_devices);
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(i);
        cudaMalloc(submodels + i, sizeof(SubModel) * num_devices);
        for (int j = 0; j < num_devices; j++) {
            submodels[i][j] = SubModel(submodel_specs[i]);
        }
    }


}
