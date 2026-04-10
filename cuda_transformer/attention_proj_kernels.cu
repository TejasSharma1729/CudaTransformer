#include "headers.cu"

#ifndef ATTENTION_PROJECTION_KERNELS
#define ATTENTION_PROJECTION_KERNELS


/**
 * @brief Projects multi-head attention output to the model dimension using a tiled matrix multiply.
 *
 * Computes output = input_flat × W^T + bias, where input_flat is the attention output
 * rearranged from [batchSize, numHeads, seqLen, headDim] into a logical
 * [batchSize*seqLen, numHeads*headDim] matrix, and W has shape [outputDim, numHeads*headDim].
 *
 * The kernel uses BLOCKDIM×BLOCKDIM shared-memory tiles to amortise global memory
 * bandwidth.  Each output element is accumulated across tiles of width blockDim.x
 * along the totalInputDim (= numHeads*headDim) reduction axis.
 *
 * Grid:
 *   gridDim.x = (outputDim       + blockDim.x - 1) / blockDim.x   — output-column tiles
 *   gridDim.y = (batchSize*seqLen + blockDim.y - 1) / blockDim.y   — output-row tiles
 * Block: (blockDim.x, blockDim.y)  —  typically (BLOCKDIM, BLOCKDIM)
 *
 * Shared memory layout (contiguous extern char[]):
 *   sharedInput   [blockDim.y * blockDim.x]  — tile of input_flat rows
 *   sharedWeights [blockDim.x * blockDim.y]  — tile of weight columns (transposed)
 * Total shared: 2 * blockDim.x * blockDim.y * sizeof(DType)
 *
 * Input rearrangement (in-kernel, no extra copy):
 *   flat row    r  →  batch b = r / seqLen,  seq s = r % seqLen
 *   flat column c  →  head  h = c / headDim, dim  d = c % headDim
 *   memory offset: ((b * numHeads + h) * seqLen + s) * headDim + d
 *
 * @tparam DType  Floating-point data type (float, double, __half, __nv_bfloat16).
 *
 * @param input    Device pointer (read);  shape [batchSize, numHeads, seqLen, headDim].
 * @param output   Device pointer (write); shape [batchSize * seqLen, outputDim]
 *                 (equivalently [batchSize, seqLen, outputDim]).
 * @param weights  Device pointer (read);  shape [outputDim, numHeads * headDim].
 *                 Row-major; weight row `o` corresponds to output dimension `o`.
 * @param biases   Device pointer (read);  shape [outputDim].
 *                 Added to every row of the output.
 * @param outputDim      Number of output features (model dimension after projection).
 * @param headDim        Dimension of each individual attention head.
 * @param sequenceLength Number of tokens in the sequence (seqLen).
 * @param numHeads       Number of attention heads.
 * @param batchSize      Number of sequences in the batch.
 */
template <typename DType = float> __global__ void attentionProj(
    const DType *input,
    DType *output,
    const DType *weights,
    const DType *biases,
    int outputDim,
    int headDim,
    int sequenceLength,
    int numHeads,
    int batchSize
) {
    extern __shared__ char block_shared_mem[];
    int totalInputDim = numHeads * headDim;
    int totalBatchSize = batchSize * sequenceLength;

    DType *sharedInput = reinterpret_cast<DType*>(block_shared_mem);
    DType *sharedWeights = sharedInput + blockDim.y * blockDim.x;

    int columnIdx = blockIdx.x * blockDim.x + threadIdx.x; // outputDim index
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;    // batch/sequence index

    DType accumulator = static_cast<DType>(0);
    for (int tileIdx = 0; tileIdx < totalInputDim; tileIdx += blockDim.x) {
        if (rowIdx < totalBatchSize && (tileIdx + threadIdx.x) < totalInputDim) {
            int curInIdx = tileIdx + threadIdx.x;
            int b = rowIdx / sequenceLength;
            int s = rowIdx % sequenceLength;
            int h = curInIdx / headDim;
            int d = curInIdx % headDim;
            sharedInput[threadIdx.y * blockDim.x + threadIdx.x] = input[((b * numHeads + h) * sequenceLength + s) * headDim + d];
        } else {
            sharedInput[threadIdx.y * blockDim.x + threadIdx.x] = static_cast<DType>(0);
        }

        if (columnIdx < outputDim && (tileIdx + threadIdx.y) < totalInputDim) {
            sharedWeights[threadIdx.x * blockDim.y + threadIdx.y] = weights[columnIdx * totalInputDim + tileIdx + threadIdx.y];
        } else {
            sharedWeights[threadIdx.x * blockDim.y + threadIdx.y] = static_cast<DType>(0);
        }
        __syncthreads();
        
        for (int k = 0; k < blockDim.x; k++) {
            accumulator += sharedInput[threadIdx.y * blockDim.x + k] * sharedWeights[threadIdx.x * blockDim.x + k];
        }
        __syncthreads();
    }

    if (rowIdx < totalBatchSize && columnIdx < outputDim) {
        output[rowIdx * outputDim + columnIdx] = accumulator + biases[columnIdx];
    }
}


/**
 * @brief Backward pass for the attention projection: accumulates weight and bias gradients.
 *
 * Computes dL/dW and dL/db for the projection layer, treating the operation as
 * output = input_flat × W^T + b.  Specifically:
 *   dW[outputDimIdx, inputDimIdx] += Σ_{batch} outputGrad[batch, outputDimIdx] * input_flat[batch, inputDimIdx]
 *   db[outputDimIdx]              += Σ_{batch} outputGrad[batch, outputDimIdx]
 *
 * The kernel iterates over the totalBatchSize (batchSize*seqLen) axis in tiles of
 * width blockDim.x to amortise memory traffic.  Both output-gradient rows and flattened
 * input rows are loaded into shared memory per tile.
 *
 * Grid:
 *   gridDim.x = (totalInputDim + blockDim.x - 1) / blockDim.x   — input-column tiles  (totalInputDim = numHeads*headDim)
 *   gridDim.y = (outputDim     + blockDim.y - 1) / blockDim.y   — output-dimension tiles
 * Block: (blockDim.x, blockDim.y)  —  typically (BLOCKDIM, BLOCKDIM)
 *
 * Shared memory layout (contiguous extern char[]):
 *   sharedOutputGrad [blockDim.y * blockDim.x]  — tile of outputGrad rows
 *   sharedInput      [blockDim.x * blockDim.y]  — tile of input_flat rows (transposed load)
 * Total shared: 2 * blockDim.x * blockDim.y * sizeof(DType)
 *
 * Gradient accumulation uses atomicAdd so that multiple blocks can safely update the
 * same weightGrad and biasGrad element when tiles overlap.
 *
 * @tparam DType  Floating-point data type (float, double, __half, __nv_bfloat16).
 *
 * @param input       Device pointer (read);  shape [batchSize, numHeads, seqLen, headDim].
 *                    Rearranged inside the kernel to [batchSize*seqLen, numHeads*headDim].
 * @param outputGrad  Device pointer (read);  shape [batchSize * seqLen, outputDim].
 *                    Upstream gradient dL/d(output).
 * @param weightGrad  Device pointer (accumulate via atomicAdd); shape [outputDim, numHeads*headDim].
 * @param biasGrad    Device pointer (accumulate via atomicAdd); shape [outputDim].
 *                    Only accumulated by threads where inputDimIdx == 0 to avoid
 *                    multi-counting along the input-column dimension.
 * @param outputDim      Number of output features.
 * @param headDim        Dimension of each attention head.
 * @param sequenceLength Number of tokens in the sequence (seqLen).
 * @param numHeads       Number of attention heads.
 * @param batchSize      Number of sequences in the batch.
 */
template <typename DType = float> __global__ void projBackwardWB(
    const DType *input,
    const DType *outputGrad,
    DType *weightGrad,
    DType *biasGrad,
    int outputDim,
    int headDim,
    int sequenceLength,
    int numHeads,
    int batchSize
) {
    extern __shared__ char block_shared_mem[];
    DType *sharedOutputGrad = reinterpret_cast<DType*>(block_shared_mem);
    DType *sharedInput = sharedOutputGrad + blockDim.y * blockDim.x;

    int inputDimIdx = blockIdx.x * blockDim.x + threadIdx.x;  // totalInputDim index
    int outputDimIdx = blockIdx.y * blockDim.y + threadIdx.y; // outputDim index

    DType weightGradAccumulator = static_cast<DType>(0);
    DType biasGradAccumulator = static_cast<DType>(0);

    int totalBatchSize = batchSize * sequenceLength;
    int totalInputDim = numHeads * headDim;

    for (int tileIdx = 0; tileIdx < totalBatchSize; tileIdx += blockDim.x) {
        if (outputDimIdx < outputDim && (tileIdx + threadIdx.x) < totalBatchSize) {
            sharedOutputGrad[threadIdx.y * blockDim.x + threadIdx.x] = outputGrad[(tileIdx + threadIdx.x) * outputDim + outputDimIdx];
        } else {
            sharedOutputGrad[threadIdx.y * blockDim.x + threadIdx.x] = static_cast<DType>(0);
        }

        if (inputDimIdx < totalInputDim && (tileIdx + threadIdx.y) < totalBatchSize) {
            int b = (tileIdx + threadIdx.y) / sequenceLength;
            int s = (tileIdx + threadIdx.y) % sequenceLength;
            int h = inputDimIdx / headDim;
            int d = inputDimIdx % headDim;
            sharedInput[threadIdx.x * blockDim.y + threadIdx.y] = input[((b * numHeads + h) * sequenceLength + s) * headDim + d];
        } else {
            sharedInput[threadIdx.x * blockDim.y + threadIdx.y] = static_cast<DType>(0);
        }

        __syncthreads();

        for (int k = 0; k < blockDim.x; k++) {
            weightGradAccumulator += sharedOutputGrad[threadIdx.y * blockDim.x + k] * sharedInput[threadIdx.x * blockDim.x + k];
        }
        biasGradAccumulator += sharedOutputGrad[threadIdx.y * blockDim.x + threadIdx.x];
        __syncthreads();
    }

    if (outputDimIdx < outputDim && inputDimIdx < totalInputDim) {
        atomicAdd(&weightGrad[outputDimIdx * totalInputDim + inputDimIdx], weightGradAccumulator);
        if (inputDimIdx == 0) {
            atomicAdd(&biasGrad[outputDimIdx], biasGradAccumulator);
        }
    }
}

/**
 * @brief Backward pass for the attention projection: computes input gradient.
 *
 * Computes dL/d(input_flat) = outputGrad × W, then scatter-writes the result
 * back into the [batchSize, numHeads, seqLen, headDim] layout.
 *   inputGrad_flat[row, col] = Σ_k outputGrad[row, k] * W[k, col]
 *
 * Uses BLOCKDIM×BLOCKDIM shared-memory tiles iterating over the outputDim axis.
 *
 * Grid:
 *   gridDim.x = (totalInputDim   + blockDim.x - 1) / blockDim.x   — input-column tiles  (totalInputDim = numHeads*headDim)
 *   gridDim.y = (batchSize*seqLen + blockDim.y - 1) / blockDim.y   — row tiles
 * Block: (blockDim.x, blockDim.y)  —  typically (BLOCKDIM, BLOCKDIM)
 *
 * Shared memory layout (contiguous extern char[]):
 *   sharedOutputGrad [blockDim.y * blockDim.x]  — tile of outputGrad columns
 *   sharedWeights    [blockDim.x * blockDim.y]  — tile of weight rows (transposed load)
 * Total shared: 2 * blockDim.x * blockDim.y * sizeof(DType)
 *
 * Output scatter (in-kernel, no extra copy):
 *   flat row    r  →  batch b = r / seqLen,  seq s = r % seqLen
 *   flat column c  →  head  h = c / headDim, dim  d = c % headDim
 *   write offset: ((b * numHeads + h) * seqLen + s) * headDim + d
 *
 * Note: this kernel overwrites (not accumulates) inputGrad — zero it before calling
 * if multiple gradient contributions are expected.
 *
 * @tparam DType  Floating-point data type (float, double, __half, __nv_bfloat16).
 *
 * @param inputGrad   Device pointer (write); shape [batchSize, numHeads, seqLen, headDim].
 * @param outputGrad  Device pointer (read);  shape [batchSize * seqLen, outputDim].
 *                    Upstream gradient dL/d(output).
 * @param weights     Device pointer (read);  shape [outputDim, numHeads * headDim].
 *                    Same weight matrix used in the forward pass.
 * @param outputDim      Number of output features.
 * @param headDim        Dimension of each attention head.
 * @param sequenceLength Number of tokens in the sequence (seqLen).
 * @param numHeads       Number of attention heads.
 * @param batchSize      Number of sequences in the batch.
 */
template <typename DType = float> __global__ void projBackward(
    DType *inputGrad,
    const DType *outputGrad,
    const DType *weights,
    int outputDim,
    int headDim,
    int sequenceLength,
    int numHeads,
    int batchSize
) {
    extern __shared__ char block_shared_mem[];
    DType *sharedOutputGrad = reinterpret_cast<DType*>(block_shared_mem);
    DType *sharedWeights = sharedOutputGrad + blockDim.y * blockDim.x;

    int totalBatchSize = batchSize * sequenceLength;
    int inputDim = numHeads * headDim;

    int colIdx = blockIdx.x * blockDim.x + threadIdx.x; // inputDim index
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y; // batch/sequence index

    DType accumulator = static_cast<DType>(0);

    for (int tileIdx = 0; tileIdx < outputDim; tileIdx += blockDim.x) {
        if (rowIdx < totalBatchSize && (tileIdx + threadIdx.x) < outputDim) {
            sharedOutputGrad[threadIdx.y * blockDim.x + threadIdx.x] = outputGrad[rowIdx * outputDim + tileIdx + threadIdx.x];
        } else {
            sharedOutputGrad[threadIdx.y * blockDim.x + threadIdx.x] = static_cast<DType>(0);
        }

        if (colIdx < inputDim && (tileIdx + threadIdx.y) < outputDim) {
            sharedWeights[threadIdx.x * blockDim.y + threadIdx.y] = weights[(tileIdx + threadIdx.y) * inputDim + colIdx];
        } else {
            sharedWeights[threadIdx.x * blockDim.y + threadIdx.y] = static_cast<DType>(0);
        }
        __syncthreads();

        for (int k = 0; k < blockDim.x; k++) {
            accumulator += sharedOutputGrad[threadIdx.y * blockDim.x + k] * sharedWeights[threadIdx.x * blockDim.x + k];
        }
        __syncthreads();
    }

    if (rowIdx < totalBatchSize && colIdx < inputDim) {
        int b = rowIdx / sequenceLength;
        int s = rowIdx % sequenceLength;
        int h = colIdx / headDim;
        int d = colIdx % headDim;
        int flatIdx = ((b * numHeads + h) * sequenceLength + s) * headDim + d;
        inputGrad[flatIdx] = accumulator;
    }
}

#endif // ATTENTION_PROJECTION_KERNELS