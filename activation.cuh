#ifndef _activation_h_
#define _activation_h_

__global__ void sigmoid_kernel(float *vec, int len);

__global__ void relu_kernel(float *vec, int len);

#endif
