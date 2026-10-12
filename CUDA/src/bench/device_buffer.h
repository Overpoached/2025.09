#ifndef BENCH_DEVICE_BUFFER_H
#define BENCH_DEVICE_BUFFER_H

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

#include "bench/cuda_check.h"

namespace bench {

// Move-only RAII wrapper around a cudaMalloc allocation.
//
// Benchmarks allocate hundreds of megabytes and throw on failure; leaking a
// buffer per failed run would exhaust an 8 GB card quickly.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) { Allocate(count); }

    ~DeviceBuffer() { Free(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            Free();
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    void Allocate(std::size_t count) {
        Free();
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
            count_ = count;
        }
    }

    void Free() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
        }
        count_ = 0;
    }

    void Fill(int byte_value) { CUDA_CHECK(cudaMemset(data_, byte_value, bytes())); }

    void CopyToHost(T* destination, std::size_t count, std::size_t element_offset = 0) const {
        CUDA_CHECK(cudaMemcpy(destination, data_ + element_offset, count * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace bench

#endif  // BENCH_DEVICE_BUFFER_H
