#include "headers.cu"

#ifndef LAYERNORM_LAYER_KERNELS
#define LAYERNORM_LAYER_KERNELS

/**
 * @brief Layer normalization forward pass: normalize each row, then scale and shift.
 *
 * One CUDA block handles one row (one token position across the batch×sequence dimension).
 * The kernel performs three sequential reduction sweeps over the feature dimension D using
 * shared memory, computing:
 *   1. mean  = (1/D) * sum(x)
 *   2. var   = (1/D) * sum((x - mean)^2)
 *   3. y[i]  = (x[i] - mean) * rsqrt(var + epsilon) * weight[i] + bias[i]
 *
 * The computed mean and inverse standard deviation are written to `cache_mean` and
 * `cache_inv_std` for reuse in the backward kernel.
 *
 * Launch config: <<<N, threads, threads * sizeof(DType)>>>
 *   N       — number of rows (totalBatchSize = batchSize * seqLen)
 *   threads — number of threads per block (must be a power of 2, typically 256)
 *
 * @tparam DType    Floating-point data type (float, double, __half, __nv_bfloat16).
 * @param input       Device pointer to input tensor [N, D].
 * @param output      Device pointer to output tensor [N, D].
 * @param weight      Device pointer to learnable scale (gamma) vector [D].
 * @param bias        Device pointer to learnable shift (beta) vector [D].
 * @param D           Feature dimension size (the dimension that is normalized).
 * @param N           Total number of rows (batchSize * sequenceLength).
 * @param epsilon     Small constant added to the variance for numerical stability.
 * @param cache_mean  Device pointer to write per-row means [N] (read by backward kernel).
 * @param cache_inv_std Device pointer to write per-row inverse std-devs [N] (read by backward kernel).
 */
template <typename DType = float> __global__ void layerNormForwardKernel(
    const DType* input,
    DType* output,
    const DType* weight,
    const DType* bias,
    int D,
    int N,
    DType epsilon,
    DType* cache_mean,
    DType* cache_inv_std
) {
    int idx = blockIdx.x; // Each block handles one row (batch * seq_len)
    if (idx >= N) return;
    
    const DType* x = input + idx * D;
    DType* y = output + idx * D;

    // Shared memory for reduction
    extern __shared__ char smem[];
    DType* sdata = reinterpret_cast<DType*>(smem);

    // 1. Mean
    DType sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += x[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    DType mean = sdata[0] / (DType)D;

    // 2. Variance
    DType var_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        DType diff = x[i] - mean;
        var_sum += diff * diff;
    }
    sdata[threadIdx.x] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    DType var = sdata[0] / (DType)D;
    DType inv_std = rsqrt(var + epsilon);

    if (threadIdx.x == 0) {
        cache_mean[idx] = mean;
        cache_inv_std[idx] = inv_std;
    }

    // 3. Normalize and scale/shift
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        DType norm_x = (x[i] - mean) * inv_std;
        y[i] = norm_x * weight[i] + bias[i];
    }
}

/**
 * @brief Layer normalization backward pass: compute gradients for input, weight, and bias.
 *
 * One CUDA block handles one row (one token position).  Using the cached forward-pass
 * statistics (mean, inv_std) the kernel computes the analytically exact gradient:
 *
 *   x_hat[i]          = (x[i] - mean) * inv_std
 *   dx_hat[i]         = gradOutput[i] * weight[i]
 *   mean_dx_hat       = (1/D) * sum(dx_hat)
 *   mean_dx_hat_x_hat = (1/D) * sum(dx_hat * x_hat)
 *   gradInput[i]      = inv_std * (dx_hat[i] - mean_dx_hat - x_hat[i] * mean_dx_hat_x_hat)
 *
 * Weight and bias gradients are accumulated via atomicAdd across blocks so that multiple
 * rows can update the same feature-dimension index concurrently.
 *
 * Launch config: <<<N, threads, 2 * threads * sizeof(DType)>>>
 *   N       — number of rows (totalBatchSize = batchSize * seqLen)
 *   threads — number of threads per block (must be a power of 2, typically 256)
 *
 * @tparam DType       Floating-point data type.
 * @param gradOutput   Device pointer to upstream gradients [N, D].
 * @param input        Device pointer to original forward-pass input [N, D].
 * @param gradInput    Device pointer to output input-gradient buffer [N, D].
 * @param weight       Device pointer to learnable scale (gamma) vector [D].
 * @param gradWeight   Device pointer to weight-gradient accumulator [D] (atomicAdd).
 * @param gradBias     Device pointer to bias-gradient accumulator [D] (atomicAdd).
 * @param cache_mean   Device pointer to per-row means cached during forward pass [N].
 * @param cache_inv_std Device pointer to per-row inverse std-devs cached during forward [N].
 * @param D            Feature dimension size.
 * @param N            Total number of rows.
 */
template <typename DType = float> __global__ void layerNormBackwardKernel(
    const DType* gradOutput,
    const DType* input,
    DType* gradInput,
    const DType* weight,
    DType* gradWeight,
    DType* gradBias,
    const DType* cache_mean,
    const DType* cache_inv_std,
    int D, int N)
{
    int idx = blockIdx.x; // one block per row
    if (idx >= N) return;

    const DType* gout = gradOutput + idx * D;
    const DType* x = input + idx * D;
    DType* gin = gradInput + idx * D;

    DType mean = cache_mean[idx];
    DType inv_std = cache_inv_std[idx];

    extern __shared__ char smem[];
    DType* sdata1 = reinterpret_cast<DType*>(smem);
    DType* sdata2 = sdata1 + blockDim.x;

    // compute sum(dx_hat) and sum(dx_hat * x_hat)
    DType sum_dx_hat = 0.0f;
    DType sum_dx_hat_x_hat = 0.0f;

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        DType x_hat = (x[i] - mean) * inv_std;
        DType dx_hat = gout[i] * weight[i];
        sum_dx_hat += dx_hat;
        sum_dx_hat_x_hat += dx_hat * x_hat;
        
        // accumulate gradWeight and gradBias
        // Since multiple blocks update this, use atomicAdd
        atomicAdd(&gradWeight[i], gout[i] * x_hat);
        atomicAdd(&gradBias[i], gout[i]);
    }

    sdata1[threadIdx.x] = sum_dx_hat;
    sdata2[threadIdx.x] = sum_dx_hat_x_hat;
    __syncthreads();

    // reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata1[threadIdx.x] += sdata1[threadIdx.x + s];
            sdata2[threadIdx.x] += sdata2[threadIdx.x + s];
        }
        __syncthreads();
    }

    DType mean_dx_hat = sdata1[0] / (DType)D;
    DType mean_dx_hat_x_hat = sdata2[0] / (DType)D;

    // dx = inv_std * (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat)
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        DType x_hat = (x[i] - mean) * inv_std;
        DType dx_hat = gout[i] * weight[i];
        gin[i] = inv_std * (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat);
    }
}

/**
 * @brief In-place SGD update for layer-norm scale (gamma) and shift (beta) parameters.
 *
 * Applies the standard SGD rule element-wise:
 *   w[i] -= lr * wg[i]
 *   b[i] -= lr * bg[i]
 *
 * Launch config: <<<(D + 255) / 256, 256>>>
 *
 * @tparam DType  Floating-point data type.
 * @param w   Device pointer to the scale (gamma) parameter vector [D].
 * @param b   Device pointer to the shift (beta) parameter vector [D].
 * @param wg  Device pointer to the scale gradient vector [D] (read-only).
 * @param bg  Device pointer to the shift gradient vector [D] (read-only).
 * @param D   Feature dimension size (total number of elements to update).
 * @param lr  Learning rate scalar.
 */
template <typename DType> __global__ void layerNormUpdateKernel(
    DType* w,
    DType* b,
    const DType* wg,
    const DType* bg,
    int D,
    DType lr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < D) {
        w[idx] -= lr * wg[idx];
        b[idx] -= lr * bg[idx];
    }
}

#endif // LAYERNORM_LAYER_KERNELS