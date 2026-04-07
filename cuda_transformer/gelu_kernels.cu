#include "headers.cu"

#ifndef GELU_KERNELS
#define GELU_KERNELS

/**
 * @brief Forward pass for the GELU activation function.
 * Uses the approximate formula: x * Φ(x) ≈ x * (0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))))
 * This is numerically stable and matches PyTorch's approximation.
 *
 * @tparam DType The data type for computations.
 * @param input Pointer to input data.
 * @param output Pointer to output data.
 * @param inputDim Size of the feature dimension.
 * @param totalBatchSize Total number of elements across all batches.
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
 * Gradient derivation:
 * d/dx[x * Φ(x)] = Φ(x) + x * φ(x)
 * where:
 *   Φ(x) = CDF of standard normal (computed via tanh approximation)
 *   φ(x) = PDF of standard normal = exp(-x^2/2) / sqrt(2π)
 *
 * @tparam DType The data type for computations.
 * @param input Original input data.
 * @param gradInput Pointer to accumulate input gradient.
 * @param gradOutput Gradient with respect to the output.
 * @param inputDim Size of the feature dimension.
 * @param totalBatchSize Total number of elements across all batches.
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
