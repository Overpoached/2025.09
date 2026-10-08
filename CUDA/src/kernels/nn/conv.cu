#include "nn.h"

#include "../kernel_utility.h"

namespace kernels {
    __global__ void Conv(const float* x,
        const float* __restrict__ w,
        float* y, 
        ConvShape shapes,
        const int64_t* __restrict__ pad,
        const int64_t* __restrict__ stride)
    {
        int64_t halfwcols = shapes.wcols / 2;
        int64_t halfwrows = shapes.wrows / 2;

        int64_t ix = blockIdx.x * int64_t(blockDim.x) + threadIdx.x * stride[1];
        int64_t iy = blockIdx.y * int64_t(blockDim.y) + threadIdx.y * stride[0];

        if (ix >= 0 && ix + halfwcols - pad[1] < shapes.xcols  // right pad
            && iy >= 0 && iy + halfwrows - pad[2] < shapes.xrows) // down pad
        {
            float val = 0.0;

            int64_t startCol = ix - halfwcols + halfwcols - pad[3]; // adds left pad;
            int64_t startRow = iy - halfwrows + halfwrows - pad[0]; // adds up pad;

            // Get the surrounding box
            for (int64_t j = 0; j < shapes.wrows; ++j) {
                for (int64_t i = 0; i < shapes.wcols; ++i) {

                    int64_t curCol = startCol + i;
                    int64_t curRow = startRow + j;

                    if (curRow > -1 && curRow < shapes.xrows && curCol > -1 && curCol < shapes.xcols) {
                        val += x[curRow * shapes.xcols + curCol] * w[j * shapes.wcols + i];
                    }
                }
            }


            // Write value
            int64_t yx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
            int64_t yy = blockIdx.y * int64_t(blockDim.y) + threadIdx.y;
            if (yx < shapes.ycols && yy < shapes.yrows) {
                y[yy * shapes.ycols + yx] = val;
            }
        }

    }
    void Conv(const float* x,
        const float* w,
        float* y,
        const ConvShape& shapes,
        const int64_t* pad,
        int64_t npad,
        const int64_t* stride,
        int64_t nstride,
        cudaStream_t stream) 
    {
        constexpr int kDimThreads = 16;
        if (npad != 4) {
            throw std::runtime_error("Expected pad size = 4. Result: " + std::to_string(npad));
        }
        if (nstride != 2) {
            throw std::runtime_error("Expected stride size = 2. Result: " + std::to_string(npad));
        }
		dim3 grid((shapes.xrows + kDimThreads - 1) / kDimThreads, (shapes.xcols + kDimThreads - 1) / kDimThreads, 1);
		dim3 block(kDimThreads, kDimThreads, 1);
        Conv << <grid, block, 0, stream >> > (x, w, y, shapes, pad, stride);
        CUDA_CHECK(cudaGetLastError());              // catches bad launch configs immediately
    }
}