#include "nn.h"

#include "../kernel_utility.h"

namespace kernels {
    __global__ void Conv(const float* x,
        const float* __restrict__ w,
        float* y, 
        ConvShape shapes)
    {
        int64_t yx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;   // output col
        int64_t yy = blockIdx.y * int64_t(blockDim.y) + threadIdx.y;   // output row
        int64_t oc = blockIdx.z * int64_t(blockDim.z) + threadIdx.z;   // output channel

        if (yx >= shapes.ycols || yy >= shapes.yrows || oc >= shapes.wcount) {
            return;
        }

        const int64_t xdim = shapes.xrows * shapes.xcols;
        const int64_t xbatchDim = shapes.xchannels * xdim;
        const int64_t wdim = shapes.wrows * shapes.wcols;
        const int64_t wfilterDim = shapes.wchannels * wdim;
        const int64_t ydim = shapes.yrows * shapes.ycols;
        const int64_t ybatchDim = shapes.wcount * ydim;

        const int64_t rowStart = yy * shapes.rowstride - shapes.uppad;
        const int64_t colStart = yx * shapes.colstride - shapes.leftpad;

        for (int64_t n = 0; n < shapes.xbatch; ++n) {
            float val = 0.0;

            for (int64_t m = 0; m < shapes.xchannels; ++m) {
                // Get the surrounding box
                for (int64_t j = 0; j < shapes.wrows; ++j) {
                    const int64_t curRow = rowStart + j;
                    if (curRow < 0 || curRow >= shapes.xrows) 
                        continue;

                    for (int64_t i = 0; i < shapes.wcols; ++i) {
                        const int64_t curCol = colStart + i;
                        if (curCol < 0 || curCol >= shapes.xcols)
                            continue;
                        val += x[n * xbatchDim + m * xdim + curRow * shapes.xcols + curCol] * w[oc * wfilterDim + m * wdim + j * shapes.wcols + i];
                    }
                }

                // Write value
                y[n * ybatchDim + oc * ydim + yy * shapes.ycols + yx] = val;
            }
        }
          

        

    }
    void Conv(const float* x,
        const float* w,
        float* y,
        const ConvShape& shapes,
        cudaStream_t stream) 
    {
        constexpr int kDimThreads = 8;
		dim3 grid((shapes.ycols + kDimThreads - 1) / kDimThreads, 
            (shapes.yrows + kDimThreads - 1) / kDimThreads, 
            (shapes.wcount + kDimThreads - 1) / kDimThreads);
		dim3 block(kDimThreads, kDimThreads, kDimThreads);
        Conv << <grid, block, 0, stream >> > (x, w, y, shapes);
        CUDA_CHECK(cudaGetLastError());              // catches bad launch configs immediately
    }
}