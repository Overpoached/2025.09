#include <cuda_runtime.h>
#include <stdio.h>

void PrintDeviceProperties() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("Error getting device count: %s\n", cudaGetErrorString(err));
        return;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        printf("Device %d: %s\n", device, prop.name);
        printf("  Total Global Memory: %zu bytes\n", prop.totalGlobalMem);
        printf("  Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    }
}