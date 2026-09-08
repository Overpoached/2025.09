#include "kernel.h"
#include <cuda_runtime.h>
#include <stdio.h>

void PrintDeviceProperties() { 
    int deviceCount = 0; 
    cudaError_t err = cudaGetDeviceCount(&deviceCount); 
    if (err != cudaSuccess) { 
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err)); 
        return; 
    } 
    for (int dev = 0; dev < deviceCount; ++dev) { 
        cudaDeviceProp prop; 
        cudaGetDeviceProperties(&prop, dev); 
        printf("Device %d: %s\n  Compute capability: %d.%d\n  Global mem: %zu MB\n", dev, prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024)); 
    } 
}

// CUDA kernel: Each thread adds 1 to one element
__global__ void addOneKernel(int* d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        d_data[idx] = d_data[idx] + 1;
    }
}

// CUDA kernel: Parallel reduction to sum array elements
__global__ void sumReductionKernel(int* d_data, int* d_partialSums, int size) {
    extern __shared__ int sharedData[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sharedData[tid] = (idx < size) ? d_data[idx] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        d_partialSums[blockIdx.x] = sharedData[0];
    }
}

int addOneAndSum(int* data, int size) {
    int* d_data = nullptr;
    int* d_partialSums = nullptr;

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory
    cudaMalloc((void**)&d_data, size * sizeof(int));
    cudaMalloc((void**)&d_partialSums, blocksPerGrid * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to add 1 to each element
    printf("Launching addOneKernel with %d blocks and %d threads per block\n",
           blocksPerGrid, threadsPerBlock);
    addOneKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        cudaFree(d_partialSums);
        return -1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy modified data back to host
    cudaMemcpy(data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Launch reduction kernel to sum all elements
    printf("Launching sumReductionKernel with %d blocks and %d threads per block\n",
           blocksPerGrid, threadsPerBlock);
    sumReductionKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(
        d_data, d_partialSums, size);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA reduction kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        cudaFree(d_partialSums);
        return -1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy partial sums back to host
    int* partialSums = new int[blocksPerGrid];
    cudaMemcpy(partialSums, d_partialSums, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

    // Final sum on CPU (sum the partial results from each block)
    int totalSum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        totalSum += partialSums[i];
    }

    // Cleanup
    delete[] partialSums;
    cudaFree(d_data);
    cudaFree(d_partialSums);

    return totalSum;
}
