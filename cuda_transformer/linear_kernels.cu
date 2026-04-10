#include "headers.cu"

#ifndef LINEAR_KERNELS
#define LINEAR_KERNELS

/**
 * @brief Forward pass for a Linear layer: output = input × W^T + bias.
 *
 * Computes output[row, col] = Σ_k input[row, k] * weights[col, k] + biases[col]
 * using BLOCKDIM×BLOCKDIM shared-memory tiles over the inputDim reduction axis.
 * The bias is pre-loaded into the accumulator before the tile loop.
 * biases may be nullptr, in which case no bias is added.
 *
 * Grid:
 *   gridDim.x = (outputDim + blockDim.x - 1) / blockDim.x  — output-column tiles
 *   gridDim.y = (batchSize + blockDim.y - 1) / blockDim.y  — row tiles
 * Block: (blockDim.x, blockDim.y)  —  typically (BLOCKDIM, BLOCKDIM)
 *
 * Shared memory layout (contiguous extern char[]):
 *   sharedInput   [blockDim.y * blockDim.x]  — tile of input rows
 *   sharedWeights [blockDim.x * blockDim.y]  — tile of weight columns (transposed load)
 * Total shared: 2 * blockDim.x * blockDim.y * sizeof(DType)
 *
 * @tparam DType Floating-point data type (float, double, __half, __nv_bfloat16).
 *
 * @param input     Device pointer (read);  shape [batchSize, inputDim].
 * @param output    Device pointer (write); shape [batchSize, outputDim].
 * @param weights   Device pointer (read);  shape [outputDim, inputDim].
 *                  Row-major; weight row `o` corresponds to output feature `o`.
 * @param biases    Device pointer (read, nullable); shape [outputDim].
 *                  Pass nullptr to skip bias addition.
 * @param inputDim  Number of input features.
 * @param outputDim Number of output features.
 * @param batchSize Number of rows (batchSize * sequenceLength for sequence inputs).
 */
template <typename DType = float> __global__ void linearForward(
    const DType *input,
    DType *output,
    const DType *weights,
    const DType *biases,
    int inputDim,
    int outputDim,
    int batchSize
) {
    extern __shared__ char block_shared_mem[];
    DType *sharedInput = reinterpret_cast<DType*>(block_shared_mem);
    DType *sharedWeights = sharedInput + blockDim.y * blockDim.x;

    int colIdx = blockIdx.x * blockDim.x + threadIdx.x; // outputDim index
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y; // batchSize index

    DType accum = (biases != nullptr && colIdx < outputDim) ? biases[colIdx] : static_cast<DType>(0);

    for (int tile = 0; tile < inputDim; tile += blockDim.x) {
        if (rowIdx < batchSize && (tile + threadIdx.x) < inputDim) {
            sharedInput[threadIdx.y * blockDim.x + threadIdx.x] = 
                input[rowIdx * inputDim + tile + threadIdx.x];
        } else {
            sharedInput[threadIdx.y * blockDim.x + threadIdx.x] = static_cast<DType>(0);
        }

        if (colIdx < outputDim && (tile + threadIdx.y) < inputDim) {
            sharedWeights[threadIdx.x * blockDim.y + threadIdx.y] = 
                weights[colIdx * inputDim + tile + threadIdx.y];
        } else {
            sharedWeights[threadIdx.x * blockDim.y + threadIdx.y] = static_cast<DType>(0);
        }
        __syncthreads();

        for (int k = 0; k < blockDim.x; k++) {
            accum += sharedInput[threadIdx.y * blockDim.x + k] * sharedWeights[threadIdx.x * blockDim.x + k];
        }
        __syncthreads();
    }

    if (rowIdx < batchSize && colIdx < outputDim) {
        output[rowIdx * outputDim + colIdx] = accum;
    }
}

/**
 * @brief Backward pass for Linear layer: accumulates weight and bias gradients.
 *
 * Computes:
 *   dW[outputIdx, inputIdx] += Σ_{batch} outputGrad[batch, outputIdx] * input[batch, inputIdx]
 *   db[outputIdx]           += Σ_{batch} outputGrad[batch, outputIdx]
 *
 * Iterates over the batchSize axis in tiles of blockDim.x, loading outputGrad and
 * input into shared memory.  Gradient accumulation uses atomicAdd.
 * Bias gradient is reduced within each warp using __shfl_down_sync before the
 * atomicAdd, so only one atomicAdd per warp is issued (blockIdx.x == 0 guard).
 *
 * Grid:
 *   gridDim.x = (inputDim  + blockDim.x - 1) / blockDim.x  — input-dimension tiles
 *   gridDim.y = (outputDim + blockDim.y - 1) / blockDim.y  — output-dimension tiles
 * Block: (blockDim.x, blockDim.y)  —  typically (BLOCKDIM, BLOCKDIM)
 *
 * Shared memory layout (contiguous extern char[]):
 *   sharedOutputGrad [blockDim.y * blockDim.x]  — tile of outputGrad rows
 *   sharedInput      [blockDim.x * blockDim.y]  — tile of input rows (transposed load)
 * Total shared: 2 * blockDim.x * blockDim.y * sizeof(DType)
 *
 * @tparam DType Floating-point data type.
 *
 * @param input       Device pointer (read);              shape [batchSize, inputDim].
 * @param outputGrad  Device pointer (read);              shape [batchSize, outputDim].
 *                    Upstream gradient dL/d(output).
 * @param weightsGrad Device pointer (accumulate via atomicAdd); shape [outputDim, inputDim].
 * @param biasesGrad  Device pointer (accumulate via atomicAdd, nullable); shape [outputDim].
 *                    Pass nullptr to skip bias gradient.
 * @param inputDim    Number of input features.
 * @param outputDim   Number of output features.
 * @param batchSize   Number of rows.
 */
template <typename DType = float> __global__ void linearBackwardWB(
    const DType *input,
    const DType *outputGrad,
    DType *weightsGrad,
    DType *biasesGrad,
    int inputDim,
    int outputDim,
    int batchSize
) {
    extern __shared__ char block_shared_mem[];
    DType *sharedOutputGrad = reinterpret_cast<DType*>(block_shared_mem);
    DType *sharedInput = sharedOutputGrad + blockDim.y * blockDim.x;

    int inputIdx = blockIdx.x * blockDim.x + threadIdx.x;  // inputDim index
    int outputIdx = blockIdx.y * blockDim.y + threadIdx.y; // outputDim index

    DType weightAccum = static_cast<DType>(0);
    DType biasAccum = static_cast<DType>(0);

    for (int tile = 0; tile < batchSize; tile += blockDim.x) {
        if (outputIdx < outputDim && (tile + threadIdx.x) < batchSize) {
            sharedOutputGrad[threadIdx.y * blockDim.x + threadIdx.x] = 
                outputGrad[(tile + threadIdx.x) * outputDim + outputIdx];
        } else {
            sharedOutputGrad[threadIdx.y * blockDim.x + threadIdx.x] = static_cast<DType>(0);
        }

        if (inputIdx < inputDim && (tile + threadIdx.y) < batchSize) {
            sharedInput[threadIdx.x * blockDim.y + threadIdx.y] = 
                input[(tile + threadIdx.y) * inputDim + inputIdx];
        } else {
            sharedInput[threadIdx.x * blockDim.y + threadIdx.y] = static_cast<DType>(0);
        }

        __syncthreads();

        for (int k = 0; k < blockDim.x; k++) {
            weightAccum += sharedOutputGrad[threadIdx.y * blockDim.x + k] * sharedInput[threadIdx.x * blockDim.x + k];
        }
        biasAccum += sharedOutputGrad[threadIdx.y * blockDim.x + threadIdx.x];
        __syncthreads();
    }

    if (outputIdx < outputDim && inputIdx < inputDim) {
        atomicAdd(&weightsGrad[outputIdx * inputDim + inputIdx], weightAccum);
    }

    // Accumulate bias grad using warp shuffle reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        biasAccum += __shfl_down_sync(0xffffffff, biasAccum, offset);
    }
    if (threadIdx.x == 0 && outputIdx < outputDim && blockIdx.x == 0 && biasesGrad != nullptr) {
        atomicAdd(&biasesGrad[outputIdx], biasAccum);
    }
}

/**
 * @brief Backward pass for Linear layer: computes input gradient.
 *
 * Computes dL/dinput[row, col] = Σ_k outputGrad[row, k] * weights[k, col],
 * which is the matrix product outputGrad × W.  Uses BLOCKDIM×BLOCKDIM shared-memory
 * tiles over the outputDim reduction axis.  Result is accumulated via atomicAdd.
 *
 * Grid:
 *   gridDim.x = (inputDim  + blockDim.x - 1) / blockDim.x  — input-column tiles
 *   gridDim.y = (batchSize + blockDim.y - 1) / blockDim.y  — row tiles
 * Block: (blockDim.x, blockDim.y)  —  typically (BLOCKDIM, BLOCKDIM)
 *
 * Shared memory layout (contiguous extern char[]):
 *   sharedOutputGrad [blockDim.y * blockDim.x]  — tile of outputGrad columns
 *   sharedWeights    [blockDim.x * blockDim.y]  — tile of weight rows (transposed load)
 * Total shared: 2 * blockDim.x * blockDim.y * sizeof(DType)
 *
 * @tparam DType Floating-point data type.
 *
 * @param inputGrad   Device pointer (accumulate via atomicAdd); shape [batchSize, inputDim].
 * @param outputGrad  Device pointer (read);  shape [batchSize, outputDim].
 *                    Upstream gradient dL/d(output).
 * @param weights     Device pointer (read);  shape [outputDim, inputDim].
 *                    Same weight matrix used in the forward pass.
 * @param inputDim    Number of input features.
 * @param outputDim   Number of output features.
 * @param batchSize   Number of rows.
 */
template <typename DType = float> __global__ void linearBackward(
    DType *inputGrad,
    const DType *outputGrad,
    const DType *weights,
    int inputDim,
    int outputDim,
    int batchSize
) {
    extern __shared__ char block_shared_mem[];
    DType *sharedOutputGrad = reinterpret_cast<DType*>(block_shared_mem);
    DType *sharedWeights = sharedOutputGrad + blockDim.y * blockDim.x;

    int colIdx = blockIdx.x * blockDim.x + threadIdx.x; // inputDim index
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y; // batchSize index

    DType accum = static_cast<DType>(0);

    for (int tile = 0; tile < outputDim; tile += blockDim.x) {
        if (rowIdx < batchSize && (tile + threadIdx.x) < outputDim) {
            sharedOutputGrad[threadIdx.y * blockDim.x + threadIdx.x] = 
                outputGrad[rowIdx * outputDim + tile + threadIdx.x];
        } else {
            sharedOutputGrad[threadIdx.y * blockDim.x + threadIdx.x] = static_cast<DType>(0);
        }

        if (colIdx < inputDim && (tile + threadIdx.y) < outputDim) {
            sharedWeights[threadIdx.x * blockDim.y + threadIdx.y] = 
                weights[(tile + threadIdx.y) * inputDim + colIdx];
        } else {
            sharedWeights[threadIdx.x * blockDim.y + threadIdx.y] = static_cast<DType>(0);
        }
        __syncthreads();

        for (int k = 0; k < blockDim.x; k++) {
            accum += sharedOutputGrad[threadIdx.y * blockDim.x + k] * sharedWeights[threadIdx.x * blockDim.x + k];
        }
        __syncthreads();
    }

    if (rowIdx < batchSize && colIdx < inputDim) {
        atomicAdd(&inputGrad[rowIdx * inputDim + colIdx], accum);
    }
}

#endif // LINEAR_KERNELS