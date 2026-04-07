#include "headers.cu"

#ifndef TANH_KERNELS
#define TANH_KERNELS


/**
 * @brief Forward pass for Tanh activation: out = tanh(x).
 */
template <typename DType> __global__ void tanhForward(
    const DType *input,
    DType *output,
    int inputDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int flatIdx    = blockIdx.y * inputDim + featureIdx;
    if (featureIdx < inputDim && blockIdx.y < totalBatchSize) {
        output[flatIdx] = (DType)tanh((double)input[flatIdx]);
    }
}

/**
 * @brief Backward pass for Tanh: grad_in = (1 - tanh(x)^2) * grad_out.
 */
template <typename DType> __global__ void tanhBackward(
    const DType *input,
    DType *gradInput,
    const DType *gradOutput,
    int inputDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int flatIdx    = blockIdx.y * inputDim + featureIdx;
    if (featureIdx < inputDim && blockIdx.y < totalBatchSize) {
        double t = tanh((double)input[flatIdx]);
        gradInput[flatIdx] += (DType)((1.0 - t * t) * (double)gradOutput[flatIdx]);
    }
}

#endif // TANH_KERNELS