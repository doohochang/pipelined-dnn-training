test: main.cu model.cu activation.cu forward.cu
	nvcc -lcublas -lcurand -o test main.cu model.cu activation.cu forward.cu

