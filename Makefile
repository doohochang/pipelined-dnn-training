test: main.cu model.cu activation.cu forward.cu backward.cu
	nvcc -lcublas -lcurand -o test main.cu model.cu activation.cu forward.cu backward.cu

