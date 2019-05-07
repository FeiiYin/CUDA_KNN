
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstring>
#include <stdlib.h>


const int THREADS = 16;
const int MOD = 1000;

/**
 方阵乘法
*/
__global__ void mulMatrixKernel(int *c, const int *a, const int *b, int size, size_t pitch)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = i / size;
	const int column = i % size;
	if (row < size && column < size) {
		int tmp = 0;
		// 应当使用对齐后的行长度作为单位
		for (int i = 0; i < size; ++i)
			tmp += a[row * pitch + i] * b[i * pitch + column];
		c[row * pitch + column] = tmp;
	}
}

__host__ void mulMatrix2D(int *c, const int *a, const int *b, int size)
{
	int *matrix_a;
	int *matrix_b;
	int *matrix_c;
	size_t pitch;
	// 保证分配的内存是合理对齐的，满足物理上的内存访问，因此可以保证对行访问时具有最优的效率
	cudaMallocPitch((void **)&matrix_a, &pitch, sizeof(int) * size, size);
	cudaMallocPitch((void **)&matrix_b, &pitch, sizeof(int) * size, size);
	cudaMallocPitch((void **)&matrix_c, &pitch, sizeof(int) * size, size);

	cudaMemcpy2D(matrix_a, sizeof(int) * size, a, pitch, sizeof(int) * size, size, cudaMemcpyHostToDevice);
	cudaMemcpy2D(matrix_b, sizeof(int) * size, b, pitch, sizeof(int) * size, size, cudaMemcpyHostToDevice);

	int threadsPerBlock = THREADS;
	int blocksPerGrid = (size * size + THREADS - 1) / THREADS;

	mulMatrixKernel <<<blocksPerGrid, threadsPerBlock>>> (matrix_c, matrix_a, matrix_b, size, pitch);

	cudaMemcpy2D(c, sizeof(int) * size, matrix_c, pitch, sizeof(int) * size, size, cudaMemcpyDeviceToHost);
	
	// cpu的内存是未被对齐的
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j)
			printf("%d ", c[i * size + j]);
		printf("\n");
	}

	cudaFree(matrix_a);
	cudaFree(matrix_b);
	cudaFree(matrix_c);
}

/**
 整形矩阵随机生成
*/
__host__ void integerMatrixGenerate(int *a, int size)
{
	a = (int *)malloc(sizeof(int) * size * size);
	for (int i = 0; i < size; i++) 
		for (int j = 0; j < size; j++) 
			a[i * size + j] = rand() % MOD;
}

int main()
{
	int *a, *b, *c;
	const int size = 4;
	integerMatrixGenerate(a, size);
	integerMatrixGenerate(b, size);
	mulMatrix2D(c, a, b, size);

    return 0;
}



/**
 模板提供的向量加法方法
*/
__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}


__host__ void testAddVector()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return;
	}

	return;
}

