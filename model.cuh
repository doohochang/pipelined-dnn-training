#ifndef _model_h_
#define _model_h_

#include "hparams.cuh"

typedef struct model {
    ModelSpec spec;

    /*
        Weight matrices of hidden layers in model.
        i th matrix spec:
            row = number of nodes in i-1 th layer
            col = number of nodes in i th layer
    */
    float **weight_matrices;

    // bias vectors of hidden layers
    float **biases; 
} Model; 

#endif
