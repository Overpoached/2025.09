#include "tensor.h"

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

namespace ohy {

	Buffer::Buffer(Location loc, size_t n) : bytes(n), location(loc) {
		switch (location) {
		case Location::Device: {
			cudaMalloc((void**)&ptr, bytes);

			// Check for kernel launch errors
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
				cudaFree(ptr);
				throw std::runtime_error("Create Buffer failed");
			}
		} break;
		case Location::Host: {
			ptr = malloc(bytes);
		} break;
		default: {
			throw std::runtime_error("Unsupported buffer location");
		}
		}
	}
	Buffer::Buffer(Location loc, const char* data, size_t n) : bytes(n), location(loc) {
		switch (location) {
		case Location::Device: {
			cudaMalloc((void**)&ptr, bytes);
			cudaMemcpy(ptr, data, bytes, cudaMemcpyHostToDevice);

			// Check for kernel launch errors
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
				cudaFree(ptr);
				throw std::runtime_error("Create Buffer failed");
			}
		} break;
		case Location::Host: {
			ptr = malloc(bytes);
			memcpy_s(ptr, bytes, data, bytes);
		} break;
		default: {
			throw std::runtime_error("Unsupported buffer location");
		}
		}
	}

	Buffer::~Buffer() {
		switch (location) {
		case Location::Device: {
			if (ptr) {
				cudaFree(ptr);
			}
		} break;
		case Location::Host: {
			if (ptr) {
				free(ptr);
			}
		} break;
		default: {
			throw std::runtime_error("Unsupported buffer location");
		}
		}
	}

	int64_t Tensor::numel() const {
		int64_t total = 1;
		for (const auto& dim : shape) {
			total *= dim;
		}
		return total;
	}

	//TensorPtr CreateTensor(DataType dtype, const std::vector<int64_t>& shape) {
	//	TensorPtr tensor = std::make_shared<Tensor>();
	//	tensor->dtype = dtype;
	//	tensor->shape = shape;
	//	return tensor;
	//}
	TensorPtr CreateTensor(Location location, DataType dtype, const std::vector<int64_t>& shape, const char* data, size_t n) {
		TensorPtr tensor = std::make_shared<Tensor>();
		tensor->dtype = dtype;
		tensor->shape = shape;
		if (data) {
			tensor->storage = std::make_shared<Buffer>(location, data, n);
		}
		else {
			tensor->storage = std::make_shared<Buffer>(location, n);
		}
		return tensor;
	}
}