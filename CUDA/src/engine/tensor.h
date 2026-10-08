#pragma once

#include "datatypes.h"
#include <vector>
#include <memory>

namespace ohy {

    struct DeviceBuffer {                       // owns GPU memory; cudaFree in destructor, non-copyable
        void* ptr = nullptr;
        size_t bytes = 0;
        explicit DeviceBuffer(size_t n);
        explicit DeviceBuffer(const char* data, size_t n);
        ~DeviceBuffer();
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    };

    struct Tensor {
        DataType                      dtype;
        std::vector<int64_t>          shape;
        std::shared_ptr<DeviceBuffer> storage;  // shared, so Reshape can reuse the same memory
        std::vector<int64_t>          host_i64; // only for small int64 initializers: Reshape targets,
        // Squeeze/Unsqueeze axes, Gather indices
        int64_t numel() const;                  // product of shape
        float* f32() const { return static_cast<float*>(storage->ptr); }
    };
    using TensorPtr = std::shared_ptr<Tensor>;

    // Generic create functions
	TensorPtr CreateTensor(DataType dtype, const std::vector<int64_t>& shape);
    TensorPtr CreateTensor(DataType dtype, const std::vector<int64_t>& shape, const char* data, size_t n);
}