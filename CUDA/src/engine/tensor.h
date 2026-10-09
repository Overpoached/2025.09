#pragma once

#include "datatypes.h"
#include <vector>
#include <memory>
#include <map>
#include <string>

namespace ohy {

    inline std::map<std::string, std::vector<int64_t>> host_op_index_map {
        { "Reshape", { 1 } },
        { "Slice", { 1,2,3,4 } },
        { "Unsqueeze", { 1 } },
        { "Squeeze", { 1 } },
        { "Expand", { 1 } },
        { "Tile", { 1 } },
        { "ConstantOfShape", { 0 } },
        { "Split", { 1 } },
        { "ReduceMean", { 1 } },
    };

    enum class Location { Host, Device };

    struct Buffer {                       // owns CPU/GPU memory; cudaFree (GPU) in destructor, non-copyable
        void* ptr = nullptr;
        size_t bytes = 0;
        Location location = Location::Device;

        explicit Buffer(Location loc, size_t n);
        explicit Buffer(Location loc, const char* data, size_t n);
        ~Buffer();
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
    };


    struct Tensor {
        DataType                      dtype;
        std::vector<int64_t>          shape;
        std::shared_ptr<Buffer> storage;  // shared, so Reshape can reuse the same memory

        int64_t numel() const;                  // product of shape
        char* data() const { return static_cast<char*>(storage->ptr); }
        float* f32() const { return static_cast<float*>(storage->ptr); }
        int* i32() const { return static_cast<int*>(storage->ptr); }
        int64_t* i64() const { return static_cast<int64_t*>(storage->ptr); }
    };
    using TensorPtr = std::shared_ptr<Tensor>;

    // Generic create functions
	//TensorPtr CreateTensor(DataType dtype, const std::vector<int64_t>& shape);
    TensorPtr CreateTensor(Location location, DataType dtype, const std::vector<int64_t>& shape, const char* data, size_t n);
}