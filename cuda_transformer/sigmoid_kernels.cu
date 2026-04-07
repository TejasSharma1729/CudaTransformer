#include "headers.cu"

#ifndef SIGMOID_KERNELS
#define SIGMOID_KERNELS

/**
 * @brief Forward pass for the Sigmoid activation: out = 1 / (1 + exp(-x)).
 */
template <typename DType> __global__ void sigmoidForward(
    const DType *input,
    DType *output,
    int inputDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int flatIdx    = blockIdx.y * inputDim + featureIdx;
    if (featureIdx < inputDim && blockIdx.y < totalBatchSize) {
        double x = (double)input[flatIdx];
        output[flatIdx] = (DType)(1.0 / (1.0 + exp(-x)));
    }
}

/**
 * @brief Backward pass for Sigmoid: grad_in = sigmoid(x) * (1 - sigmoid(x)) * grad_out.
 */
template <typename DType> __global__ void sigmoidBackward(
    const DType *input,
    DType *gradInput,
    const DType *gradOutput,
    int inputDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int flatIdx    = blockIdx.y * inputDim + featureIdx;
    if (featureIdx < inputDim && blockIdx.y < totalBatchSize) {
        double x   = (double)input[flatIdx];
        double sig = 1.0 / (1.0 + exp(-x));
        gradInput[flatIdx] += (DType)(sig * (1.0 - sig) * (double)gradOutput[flatIdx]);
    }
}

#endif // SIGMOID_KERNELS