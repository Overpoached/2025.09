#include "tensor.h"

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

namespace ohy {

	DeviceBuffer::DeviceBuffer(size_t n) : bytes(n) {
		cudaMalloc((void**)&ptr, bytes);

		// Check for kernel launch errors
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
			cudaFree(ptr);
			throw std::runtime_error("Create DeviceBuffer failed");
		}
	}
	DeviceBuffer::DeviceBuffer(const char* data, size_t n) : bytes(n) {
		cudaMalloc((void**)&ptr, bytes);
		cudaMemcpy(ptr, data, bytes, cudaMemcpyHostToDevice);

		// Check for kernel launch errors
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
			cudaFree(ptr);
			throw std::runtime_error("Create DeviceBuffer failed");
		}
	}

	DeviceBuffer::~DeviceBuffer() {
		if (ptr) {
			cudaFree(ptr);
		}
	}

	int64_t Tensor::numel() const {
		int64_t total = 1;
		for (const auto& dim : shape) {
			total *= dim;
		}
		return total;
	}

	TensorPtr CreateTensor(DataType dtype, const std::vector<int64_t>& shape) {
		TensorPtr tensor = std::make_shared<Tensor>();
		tensor->dtype = dtype;
		tensor->shape = shape;
		return tensor;
	}
	TensorPtr CreateTensor(DataType dtype, const std::vector<int64_t>& shape, const char* data, size_t n) {
		TensorPtr tensor = std::make_shared<Tensor>();
		tensor->dtype = dtype;
		tensor->shape = shape;
		if (data) {
			tensor->storage = std::make_shared<DeviceBuffer>(data, n);
		}
		else {
			tensor->storage = std::make_shared<DeviceBuffer>(n);
		}
		return tensor;
	}
}