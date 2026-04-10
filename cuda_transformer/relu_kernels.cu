#include "headers.cu"

#ifndef RELU_KERNELS
#define RELU_KERNELS

/**
 * @brief Forward pass for the ReLU activation: output[i] = max(0, input[i]).
 *
 * Each thread handles one element; blockIdx.y selects the row (batch/sequence
 * position) and threadIdx.x + blockIdx.x*blockDim.x selects the feature column.
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
template <typename DType = float> __global__ void reluForward(
    const DType *input,
    DType *output,
    int inputDim,
    int totalBatchSize

) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int flatIdx = blockIdx.y * inputDim + featureIdx;
    if (featureIdx < inputDim && blockIdx.y < totalBatchSize) {
        output[flatIdx] = input[flatIdx] > (DType)0 ? input[flatIdx] : (DType)0;
    }
}

/**
 * @brief Backward pass for the ReLU activation.
 *
 * Computes dL/dinput[i] += gradOutput[i] if input[i] > 0, else 0.
 * Uses accumulation (+=) so the caller must zero gradInput before the first call
 * if it is shared across multiple upstream branches.
 *
 * Grid / Block: identical to reluForward.
 *   gridDim.x = (inputDim + blockDim.x - 1) / blockDim.x
 *   gridDim.y = totalBatchSize
 * Block: (blockDim.x, 1)  —  typically (256, 1)
 *
 * @tparam DType Floating-point data type.
 *
 * @param input          Device pointer (read);       shape [totalBatchSize, inputDim].
 *                       Original forward-pass input; used as the ReLU mask.
 * @param gradInput      Device pointer (accumulate); shape [totalBatchSize, inputDim].
 * @param gradOutput     Device pointer (read);       shape [totalBatchSize, inputDim].
 *                       Upstream gradient dL/d(output).
 * @param inputDim       Number of features per sample.
 * @param totalBatchSize Number of rows.
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