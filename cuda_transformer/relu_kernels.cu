#include "headers.cu"

#ifndef RELU_KERNELS
#define RELU_KERNELS

/**
 * @brief Forward pass for the ReLU activation function.
 * 
 * @tparam DType The data type for computations.
 * @param input Pointer to input data.
 * @param output Pointer to output data.
 * @param inputDim Size of the feature dim.
 * @param totalBatchSize Total number of elements across all batches.
 */
template <typename DType = float> __global__ void reluForward(
    const DType *input,
    DType *output,
    int inputDim,
    int totalBatchSize

) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int flatIdx = blockIdx.y * inputDim + featureIdx;
    if (featureIdx < inputDim && blockIdx.y < totalBatchSize) {
        output[flatIdx] = input[flatIdx] > static_cast<DType>(0) ? input[flatIdx] : static_cast<DType>(0);
    }
}

/**
 * @brief Backward pass for the ReLU activation function.
 * 
 * @tparam DType The data type for computations.
 * @param input Original input data.
 * @param gradInput Pointer to accumulate input grad.
 * @param gradOutput Grad with respect to the output.
 * @param inputDim Size of the feature dim.
 * @param totalBatchSize Total number of elements across all batches.
 */
template <typename DType = float> __global__ void reluBackward(
    const DType *input,
    DType *gradInput,
    const DType *gradOutput,
    int inputDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int flatIdx = blockIdx.y * inputDim + featureIdx;
    if (featureIdx < inputDim && blockIdx.y < totalBatchSize) {
        gradInput[flatIdx] += (input[flatIdx] > static_cast<DType>(0))
            ? gradOutput[flatIdx]
            : static_cast<DType>(0);
    }
}

#endif // RELU_KERNELS