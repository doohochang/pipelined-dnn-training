#ifndef _model_h_
#define _model_h_

#include "hparams.cuh"

class SubModel {
    public:
        // SubModel specification
        SubModelSpec spec;

        /*
            Weight matrices of layers in model.
            i th matrix spec:
                row = number of nodes in i-1 th layer
                col = number of nodes in i th layer
        */
        
        
        
        float *weight_matrices;
        int weight_matrices_size;

        // Bias vectors of hidden layers
        float *biases;
        int biases_size;
        
        // Computed forward values of each node
        float *forward_values;
        int forward_values_size;
        
        float *gradients;
        int gradients_size;

        SubModel(SubModelSpec spec, int batch_size);
        ~SubModel();
};

#endif
