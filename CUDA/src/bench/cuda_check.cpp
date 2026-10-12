#include "bench/cuda_check.h"

#include <sstream>

namespace bench {

void ThrowCudaError(cudaError_t code, const char* expr, const char* file, int line) {
    std::ostringstream out;
    out << "CUDA error " << static_cast<int>(code) << " (" << cudaGetErrorName(code) << "): "
        << cudaGetErrorString(code) << "\n  at " << file << ":" << line << "\n  in  " << expr;
    throw CudaError(code, out.str());
}

}  // namespace bench
