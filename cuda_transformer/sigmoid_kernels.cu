#include "headers.cu"

#ifndef SIGMOID_KERNELS
#define SIGMOID_KERNELS

/**
 * @brief Forward pass for the Sigmoid activation: output[i] = 1 / (1 + exp(-input[i])).
 *
 * Each thread handles one element.  Intermediate arithmetic uses ComputeType<DType>
 * (float for half/bfloat16/float; double when DType is double), then casts back to DType.
 *
 * Grid:
 *   gridDim.x = (inputDim       + blockDim.x - 1) / blockDim.x  — feature tiles
 *   gridDim.y = totalBatchSize                                   — one block row per sample
 * Block: (blockDim.x, 1)  —  typically (256, 1)
 *
 * No shared memory used.
 *
 * @tparam DType Floating-point data type (float, double, __half, __nv_bfloat16).
 *
 * @param input          Device pointer (read);  shape [totalBatchSize, inputDim].
 * @param output         Device pointer (write); shape [totalBatchSize, inputDim].
 * @param inputDim       Number of features per sample (last dimension).
 * @param totalBatchSize Number of rows (batchSize * sequenceLength, or just batchSize).
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
        using CT = ComputeType<DType>;
        CT x = (CT)input[flatIdx];
        output[flatIdx] = (DType)((CT)1 / ((CT)1 + exp(-x)));
    }
}

/**
 * @brief Backward pass for the Sigmoid activation.
 *
 * Computes dL/dinput[i] += sigmoid(input[i]) * (1 - sigmoid(input[i])) * gradOutput[i].
 * Uses accumulation (+=); zero gradInput before the first backward call if needed.
 * Sigmoid is recomputed from the original input to avoid storing the forward output.
 *
 * Grid / Block: identical to sigmoidForward.
 *   gridDim.x = (inputDim + blockDim.x - 1) / blockDim.x
 *   gridDim.y = totalBatchSize
 * Block: (blockDim.x, 1)  —  typically (256, 1)
 *
 * @tparam DType Floating-point data type.
 *
 * @param input          Device pointer (read);       shape [totalBatchSize, inputDim].
 *                       Original forward-pass input; sigmoid is recomputed from it.
 * @param gradInput      Device pointer (accumulate); shape [totalBatchSize, inputDim].
 * @param gradOutput     Device pointer (read);       shape [totalBatchSize, inputDim].
 *                       Upstream gradient dL/d(output).
 * @param inputDim       Number of features per sample.
 * @param totalBatchSize Number of rows.
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
        using CT = ComputeType<DType>;
        CT x   = (CT)input[flatIdx];
        CT sig = (CT)1 / ((CT)1 + exp(-x));
        gradInput[flatIdx] += (DType)(sig * ((CT)1 - sig) * (CT)gradOutput[flatIdx]);
    }
}

#endif // SIGMOID_KERNELS