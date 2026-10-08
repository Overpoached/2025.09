#include "nn.h"

#include "../kernel_utility.h"

namespace kernels {
    __global__ void Conv(const float* x, int64_t xrows, int64_t xcols,
        const float* __restrict__ w, int64_t wrows, int64_t wcols,
        float* y, int64_t yrows, int64_t ycols,
        const int64_t* __restrict__ pad,
        const int64_t* __restrict__ stride)
    {
        int64_t halfwcols = wcols / 2;
        int64_t halfwrows = wrows / 2;

        int64_t ix = blockIdx.x * int64_t(blockDim.x) + threadIdx.x * stride[1];
        int64_t iy = blockIdx.y * int64_t(blockDim.y) + threadIdx.y * stride[0];

        if (ix >= 0 && ix + halfwcols - pad[1] < xcols  // right pad
            && iy >= 0 && iy + halfwrows - pad[2] < xrows) // down pad
        {
            float val = 0.0;

            int64_t startCol = ix - halfwcols + halfwcols - pad[3]; // adds left pad;
            int64_t startRow = iy - halfwrows + halfwrows - pad[0]; // adds up pad;

            // Get the surrounding box
            for (int64_t j = 0; j < wrows; ++j) {
                for (int64_t i = 0; i < wcols; ++i) {

                    int64_t curCol = startCol + i;
                    int64_t curRow = startRow + j;

                    if (curRow > -1 && curRow < xrows && curCol > -1 && curCol < xcols) {
                        val += x[curRow * xcols + curCol] * w[j * wcols + i];
                    }
                }
            }


            // Write value
            int64_t yx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
            int64_t yy = blockIdx.y * int64_t(blockDim.y) + threadIdx.y;
            if (yx < ycols && yy < yrows) {
                y[yy * ycols + yx] = val;
            }
        }

    }
    void Conv(const float* x, int64_t xrows, int64_t xcols,
        const float* w, int64_t wrows, int64_t wcols,
        float* y, int64_t yrows, int64_t ycols,
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
		dim3 grid((xrows + kDimThreads - 1) / kDimThreads, (xcols + kDimThreads - 1) / kDimThreads, 1);
		dim3 block(kDimThreads, kDimThreads, 1);
        Conv << <grid, block, 0, stream >> > (x, xrows, xcols, w, wrows, wcols, y, yrows, ycols, pad, stride);
        CUDA_CHECK(cudaGetLastError());              // catches bad launch configs immediately
    }
}