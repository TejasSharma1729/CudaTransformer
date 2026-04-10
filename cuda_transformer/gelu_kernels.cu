#include "headers.cu"

#ifndef GELU_KERNELS
#define GELU_KERNELS

/**
 * @brief Forward pass for the GELU activation function.
 *
 * Applies the tanh approximation of the Gaussian Error Linear Unit:
 *   output[i] = x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * where x = input[i].  This matches PyTorch's default GELU approximation.
 * All intermediate arithmetic is done in double precision for numerical stability.
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
template <typename DType = float> __global__ void geluForward(
    const DType *input,
    DType *output,
    int inputDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int flatIdx = blockIdx.y * inputDim + featureIdx;

    if (featureIdx < inputDim && blockIdx.y < totalBatchSize) {
        double x = (double)input[flatIdx];

        // GELU approximation using double precision: x * Φ(x)
        // Using: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        double cdf_approx = 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));

        output[flatIdx] = (DType)(x * cdf_approx);
    }
}

/**
 * @brief Backward pass for the GELU activation function.
 *
 * Computes dL/dinput[i] += geluGrad(input[i]) * gradOutput[i], where:
 *   let u   = √(2/π) * (x + 0.044715 * x³)
 *   let Φ   = 0.5 * (1 + tanh(u))           (CDF approximation)
 *   let Φ'  = 0.5 * (1 - tanh²(u)) * √(2/π) * (1 + 3*0.044715*x²)
 *   geluGrad(x) = Φ + x * Φ'
 *
 * Uses accumulation (+=); zero gradInput before the first backward call if needed.
 * All arithmetic is done in double precision.
 *
 * Grid / Block: identical to geluForward.
 *   gridDim.x = (inputDim + blockDim.x - 1) / blockDim.x
 *   gridDim.y = totalBatchSize
 * Block: (blockDim.x, 1)  —  typically (256, 1)
 *
 * @tparam DType Floating-point data type.
 *
 * @param input          Device pointer (read);       shape [totalBatchSize, inputDim].
 *                       Original forward-pass input; GELU derivative is recomputed from it.
 * @param gradInput      Device pointer (accumulate); shape [totalBatchSize, inputDim].
 * @param gradOutput     Device pointer (read);       shape [totalBatchSize, inputDim].
 *                       Upstream gradient dL/d(output).
 * @param inputDim       Number of features per sample.
 * @param totalBatchSize Number of rows.
 */
template <typename DType = float> __global__ void geluBackward(
    const DType *input,
    DType *gradInput,
    const DType *gradOutput,
    int inputDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int flatIdx = blockIdx.y * inputDim + featureIdx;

    if (featureIdx < inputDim && blockIdx.y < totalBatchSize) {
        double x = (double)input[flatIdx];
        double grad_out = (double)gradOutput[flatIdx];

        // Compute CDF approximation using double precision: Φ(x) ≈ 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        double coeff = 0.7978845608; // sqrt(2/π)
        double x_cubed = x * x * x;
        double cdf_arg = coeff * (x + 0.044715 * x_cubed);
        double cdf_approx = 0.5 * (1.0 + tanh(cdf_arg));

        // Compute tanh derivative: d/dx[tanh(u)] = (1 - tanh(u)^2) * du/dx
        double tanh_val = tanh(cdf_arg);
        double tanh_deriv = (1.0 - tanh_val * tanh_val);

        // du/dx where u = sqrt(2/π) * (x + 0.044715 * x^3)
        double du_dx = coeff * (1.0 + 3.0 * 0.044715 * x * x);

        // Compute PDF derivative: φ(x) ≈ (1/sqrt(2π)) * exp(-x^2/2) * d/dx[Φ(x)]
        // More efficient: φ(x) is the derivative of the CDF approximation
        double cdf_deriv = 0.5 * tanh_deriv * du_dx;

        // Full gradient: [Φ(x) + x * φ(x)] * grad_out
        double gelu_grad = cdf_approx + x * cdf_deriv;

        gradInput[flatIdx] += (DType)(grad_out * gelu_grad);
    }
}

#endif // GELU_KERNELS
