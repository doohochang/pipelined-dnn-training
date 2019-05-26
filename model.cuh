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
        float **weight_matrices;

        // Bias vectors of hidden layers
        float **biases; 

        // Computed forward values of each node
        float **forward_values;
        
        float **gradients;

        SubModel(SubModelSpec spec);
        ~SubModel();
};

#endif
