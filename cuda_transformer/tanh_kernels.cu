#include "headers.cu"

#ifndef TANH_KERNELS
#define TANH_KERNELS


/**
 * @brief Forward pass for the Tanh activation: output[i] = tanh(input[i]).
 *
 * Each thread handles one element.  Computation uses double precision tanh
 * for accuracy before casting back to DType.
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
 * @brief Backward pass for the Tanh activation.
 *
 * Computes dL/dinput[i] += (1 - tanh(input[i])^2) * gradOutput[i].
 * Tanh is recomputed from the original input; the forward output need not be cached.
 * Uses accumulation (+=); zero gradInput before the first backward call if needed.
 *
 * Grid / Block: identical to tanhForward.
 *   gridDim.x = (inputDim + blockDim.x - 1) / blockDim.x
 *   gridDim.y = totalBatchSize
 * Block: (blockDim.x, 1)  —  typically (256, 1)
 *
 * @tparam DType Floating-point data type.
 *
 * @param input          Device pointer (read);       shape [totalBatchSize, inputDim].
 *                       Original forward-pass input; tanh is recomputed from it.
 * @param gradInput      Device pointer (accumulate); shape [totalBatchSize, inputDim].
 * @param gradOutput     Device pointer (read);       shape [totalBatchSize, inputDim].
 *                       Upstream gradient dL/d(output).
 * @param inputDim       Number of features per sample.
 * @param totalBatchSize Number of rows.
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