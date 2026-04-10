#include "headers.cu"

#ifndef ATTENTION_QKV_KERNELS
#define ATTENTION_QKV_KERNELS

/**
 * @brief Fused tiled projection of the input to Query, Key, and Value matrices.
 *
 * Computes three linear projections in a single kernel pass:
 *   Q[b, h, s, d] = Σ_k input[b*S+s, k] * WQ[h*headDim+d, k] + bQ[h*headDim+d]
 *   K[b, h, s, d] = Σ_k input[b*S+s, k] * WK[h*headDim+d, k] + bK[h*headDim+d]
 *   V[b, h, s, d] = Σ_k input[b*S+s, k] * WV[h*headDim+d, k] + bV[h*headDim+d]
 *
 * Uses BLOCKDIM×BLOCKDIM shared-memory tiles over the inputDim reduction axis.
 * All three weight tiles are loaded simultaneously per tile iteration to maximise
 * L1 reuse of the shared input tile.
 *
 * Output is scatter-written from the logical flat (batchSize*seqLen, numHeads*headDim)
 * layout into the [batchSize, numHeads, seqLen, headDim] 4-D layout.
 *
 * Grid:
 *   gridDim.x = (numHeads * headDim    + blockDim.x - 1) / blockDim.x  — projection-column tiles
 *   gridDim.y = (batchSize * seqLen    + blockDim.y - 1) / blockDim.y  — row tiles
 * Block: (blockDim.x, blockDim.y)  —  typically (BLOCKDIM, BLOCKDIM)
 *
 * Shared memory layout (contiguous extern char[]):
 *   sharedInput        [blockDim.y * blockDim.x]  — tile of input rows
 *   sharedWeightsQuery [blockDim.x * blockDim.y]  — tile of WQ columns (transposed load)
 *   sharedWeightsKey   [blockDim.x * blockDim.y]  — tile of WK columns (transposed load)
 *   sharedWeightsValue [blockDim.x * blockDim.y]  — tile of WV columns (transposed load)
 * Total shared: 4 * blockDim.x * blockDim.y * sizeof(DType)
 *
 * @tparam DType Floating-point data type (float, double, __half, __nv_bfloat16).
 *
 * @param input         Device pointer (read);  shape [batchSize * seqLen, inputDim].
 * @param queries       Device pointer (write); shape [batchSize, numHeads, seqLen, headDim].
 * @param weightsQuery  Device pointer (read);  shape [numHeads * headDim, inputDim].
 * @param biasesQuery   Device pointer (read);  shape [numHeads * headDim].
 * @param keys          Device pointer (write); shape [batchSize, numHeads, seqLen, headDim].
 * @param weightsKey    Device pointer (read);  shape [numHeads * headDim, inputDim].
 * @param biasesKey     Device pointer (read);  shape [numHeads * headDim].
 * @param values        Device pointer (write); shape [batchSize, numHeads, seqLen, headDim].
 * @param weightsValue  Device pointer (read);  shape [numHeads * headDim, inputDim].
 * @param biasesValue   Device pointer (read);  shape [numHeads * headDim].
 * @param inputDim      Number of input features per token.
 * @param headDim       Dimension of each individual attention head.
 * @param sequenceLength Number of tokens per sequence (seqLen).
 * @param numHeads      Number of attention heads.
 * @param batchSize     Number of sequences in the batch.
 */
template <typename DType = float> __global__ void getQKVmatrices(
    const DType *input,
    DType *queries,
    const DType *weightsQuery,
    const DType *biasesQuery,
    DType *keys,
    const DType *weightsKey,
    const DType *biasesKey,
    DType *values,
    const DType *weightsValue,
    const DType *biasesValue,
    int inputDim,
    int headDim,
    int sequenceLength,
    int numHeads,
    int batchSize
) {
    extern __shared__ char block_shared_mem[];
    int totalOutputDim = numHeads * headDim;
    int totalBatchSize = batchSize * sequenceLength;

    DType *sharedInput = reinterpret_cast<DType*>(block_shared_mem);
    DType *sharedWeightsQuery = sharedInput + blockDim.y * blockDim.x;
    DType *sharedWeightsKey = sharedWeightsQuery + blockDim.x * blockDim.y;
    DType *sharedWeightsValue = sharedWeightsKey + blockDim.x * blockDim.y;

    int columnIdx = blockIdx.x * blockDim.x + threadIdx.x; // proj index
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;    // combined batch/sequence index

    DType queryAccumulator = static_cast<DType>(0);
    DType keyAccumulator = static_cast<DType>(0);
    DType valueAccumulator = static_cast<DType>(0);

    for (int tileIdx = 0; tileIdx < inputDim; tileIdx += blockDim.x) {
        if (rowIdx < totalBatchSize && (tileIdx + threadIdx.x) < inputDim) {
            sharedInput[threadIdx.y * blockDim.x + threadIdx.x] = input[rowIdx * inputDim + tileIdx + threadIdx.x];
        } else {
            sharedInput[threadIdx.y * blockDim.x + threadIdx.x] = static_cast<DType>(0);
        }

        if (columnIdx < totalOutputDim && (tileIdx + threadIdx.y) < inputDim) {
            int weightIdx = columnIdx * inputDim + (tileIdx + threadIdx.y);
            sharedWeightsQuery[threadIdx.x * blockDim.y + threadIdx.y] = weightsQuery[weightIdx];
            sharedWeightsKey[threadIdx.x * blockDim.y + threadIdx.y] = weightsKey[weightIdx];
            sharedWeightsValue[threadIdx.x * blockDim.y + threadIdx.y] = weightsValue[weightIdx];
        } else {
            sharedWeightsQuery[threadIdx.x * blockDim.y + threadIdx.y] = static_cast<DType>(0);
            sharedWeightsKey[threadIdx.x * blockDim.y + threadIdx.y] = static_cast<DType>(0);
            sharedWeightsValue[threadIdx.x * blockDim.y + threadIdx.y] = static_cast<DType>(0);
        }
        __syncthreads();

        for (int k = 0; k < blockDim.x; k++) {
            DType inputValue = sharedInput[threadIdx.y * blockDim.x + k];
            queryAccumulator += inputValue * sharedWeightsQuery[threadIdx.x * blockDim.x + k];
            keyAccumulator += inputValue * sharedWeightsKey[threadIdx.x * blockDim.x + k];
            valueAccumulator += inputValue * sharedWeightsValue[threadIdx.x * blockDim.x + k];
        }
        __syncthreads();
    }

    if (rowIdx < totalBatchSize && columnIdx < totalOutputDim) {
        int bIdx = rowIdx / sequenceLength;
        int sIdx = rowIdx % sequenceLength;
        int hIdx = columnIdx / headDim;
        int dIdx = columnIdx % headDim;
        int outputIdx = ((bIdx * numHeads + hIdx) * sequenceLength + sIdx) * headDim + dIdx;
        queries[outputIdx] = queryAccumulator + biasesQuery[columnIdx];
        keys[outputIdx] = keyAccumulator + biasesKey[columnIdx];
        values[outputIdx] = valueAccumulator + biasesValue[columnIdx];
    }
}


/**
 * @brief Backward pass for the fused QKV projection: accumulates weight and bias gradients.
 *
 * Computes dL/dWQ, dL/dWK, dL/dWV and dL/dbQ, dL/dbK, dL/dbV from the upstream
 * gradients of Q, K, V and the original input:
 *   dWQ[outputIdx, inputIdx] += Σ_{batch} queryGrad_flat[batch, outputIdx] * input[batch, inputIdx]
 *   (identical formula for K and V)
 *   dbQ[outputIdx]           += Σ_{batch} queryGrad_flat[batch, outputIdx]
 *
 * Iterates over the totalBatchSize (batchSize*seqLen) axis in tiles of blockDim.x.
 * All three upstream gradient tiles (Q, K, V) are loaded simultaneously per tile
 * to share the input tile load.  Gradient accumulation uses atomicAdd.
 * Bias gradients are accumulated per tile (inputIdx == 0 guard to avoid counting
 * each input-dim column multiple times) and atomicAdded at the end.
 *
 * Grid:
 *   gridDim.x = (inputDim          + blockDim.x - 1) / blockDim.x  — input-dim tiles
 *   gridDim.y = (numHeads * headDim + blockDim.y - 1) / blockDim.y  — projection-dim tiles
 * Block: (blockDim.x, blockDim.y)  —  typically (BLOCKDIM, BLOCKDIM)
 *
 * Shared memory layout (contiguous extern char[]):
 *   sharedInput [blockDim.y * blockDim.x]  — tile of input rows (transposed load)
 *   sharedQGrad [blockDim.y * blockDim.x]  — tile of queryGrad_flat rows
 *   sharedKGrad [blockDim.y * blockDim.x]  — tile of keyGrad_flat rows
 *   sharedVGrad [blockDim.y * blockDim.x]  — tile of valueGrad_flat rows
 * Total shared: 4 * blockDim.x * blockDim.y * sizeof(DType)
 *
 * @tparam DType Floating-point data type.
 *
 * @param input           Device pointer (read); shape [batchSize * seqLen, inputDim].
 * @param queryGrad       Device pointer (read); shape [batchSize, numHeads, seqLen, headDim].
 *                        Upstream gradient dL/dQ; gathered into flat layout in-kernel.
 * @param weightsQueryGrad Device pointer (accumulate via atomicAdd); shape [numHeads*headDim, inputDim].
 * @param biasesQueryGrad  Device pointer (accumulate via atomicAdd); shape [numHeads*headDim].
 * @param keyGrad         Device pointer (read); shape [batchSize, numHeads, seqLen, headDim].
 * @param weightsKeyGrad  Device pointer (accumulate via atomicAdd); shape [numHeads*headDim, inputDim].
 * @param biasesKeyGrad   Device pointer (accumulate via atomicAdd); shape [numHeads*headDim].
 * @param valueGrad       Device pointer (read); shape [batchSize, numHeads, seqLen, headDim].
 * @param weightsValueGrad Device pointer (accumulate via atomicAdd); shape [numHeads*headDim, inputDim].
 * @param biasesValueGrad  Device pointer (accumulate via atomicAdd); shape [numHeads*headDim].
 * @param inputDim        Number of input features per token.
 * @param headDim         Dimension of each attention head.
 * @param sequenceLength  Number of tokens per sequence.
 * @param numHeads        Number of attention heads.
 * @param batchSize       Number of sequences in the batch.
 */
template <typename DType = float> __global__ void qkvBackwardWB(
    const DType *input,
    const DType *queryGrad,
    DType *weightsQueryGrad,
    DType *biasesQueryGrad,
    const DType *keyGrad,
    DType *weightsKeyGrad,
    DType *biasesKeyGrad,
    const DType *valueGrad,
    DType *weightsValueGrad,
    DType *biasesValueGrad,
    int inputDim,
    int headDim,
    int sequenceLength,
    int numHeads,
    int batchSize
) {
    extern __shared__ char block_shared_mem[];
    int projDim = numHeads * headDim;
    int totalBatchSize = batchSize * sequenceLength;

    DType *sharedInput = reinterpret_cast<DType*>(block_shared_mem);
    DType *sharedQGrad = sharedInput + blockDim.y * blockDim.x;
    DType *sharedKGrad = sharedQGrad + blockDim.y * blockDim.x;
    DType *sharedVGrad = sharedKGrad + blockDim.y * blockDim.x;

    int inputIdx = blockIdx.x * blockDim.x + threadIdx.x;  // modelDim index
    int outputIdx = blockIdx.y * blockDim.y + threadIdx.y; // projDim index
    
    DType wGradQ = 0, wGradK = 0, wGradV = 0;
    DType bGradQ = 0, bGradK = 0, bGradV = 0;

    for (int tile = 0; tile < totalBatchSize; tile += blockDim.x) {
        if (inputIdx < inputDim && (tile + threadIdx.y) < totalBatchSize) {
            sharedInput[threadIdx.x * blockDim.y + threadIdx.y] = input[(tile + threadIdx.y) * inputDim + inputIdx];
        } else {
            sharedInput[threadIdx.x * blockDim.y + threadIdx.y] = 0;
        }

        if (outputIdx < projDim && (tile + threadIdx.x) < totalBatchSize) {
            int b = (tile + threadIdx.x) / sequenceLength;
            int s = (tile + threadIdx.x) % sequenceLength;
            int h = outputIdx / headDim;
            int d = outputIdx % headDim;
            int flatIdx = ((b * numHeads + h) * sequenceLength + s) * headDim + d;
            sharedQGrad[threadIdx.y * blockDim.x + threadIdx.x] = queryGrad[flatIdx];
            sharedKGrad[threadIdx.y * blockDim.x + threadIdx.x] = keyGrad[flatIdx];
            sharedVGrad[threadIdx.y * blockDim.x + threadIdx.x] = valueGrad[flatIdx];
        } else {
            sharedQGrad[threadIdx.y * blockDim.x + threadIdx.x] = 0;
            sharedKGrad[threadIdx.y * blockDim.x + threadIdx.x] = 0;
            sharedVGrad[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }
        __syncthreads();

        for (int k = 0; k < blockDim.x; k++) {
            DType inVal = sharedInput[threadIdx.x * blockDim.x + k];
            wGradQ += sharedQGrad[threadIdx.y * blockDim.x + k] * inVal;
            wGradK += sharedKGrad[threadIdx.y * blockDim.x + k] * inVal;
            wGradV += sharedVGrad[threadIdx.y * blockDim.x + k] * inVal;
        }
        
        if (inputIdx == 0) {
            bGradQ += sharedQGrad[threadIdx.y * blockDim.x + threadIdx.x];
            bGradK += sharedKGrad[threadIdx.y * blockDim.x + threadIdx.x];
            bGradV += sharedVGrad[threadIdx.y * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }

    if (inputIdx < inputDim && outputIdx < projDim) {
        atomicAdd(&weightsQueryGrad[outputIdx * inputDim + inputIdx], wGradQ);
        atomicAdd(&weightsKeyGrad[outputIdx * inputDim + inputIdx], wGradK);
        atomicAdd(&weightsValueGrad[outputIdx * inputDim + inputIdx], wGradV);
        if (inputIdx == 0) {
            atomicAdd(&biasesQueryGrad[outputIdx], bGradQ);
            atomicAdd(&biasesKeyGrad[outputIdx], bGradK);
            atomicAdd(&biasesValueGrad[outputIdx], bGradV);
        }
    }
}

/**
 * @brief Backward pass for the fused QKV projection: computes input gradient.
 *
 * Accumulates the contribution to dL/dinput from all three projections:
 *   dInput[row, col] += Σ_k queryGrad_flat[row, k] * WQ[k, col]
 *                     + Σ_k keyGrad_flat  [row, k] * WK[k, col]
 *                     + Σ_k valueGrad_flat[row, k] * WV[k, col]
 *
 * Uses BLOCKDIM×BLOCKDIM shared-memory tiles iterating over the projDim
 * (= numHeads*headDim) reduction axis.  All three gradient tiles and all three
 * weight tiles are loaded simultaneously per iteration to amortise memory traffic.
 * Gradient accumulation uses atomicAdd.
 *
 * Grid:
 *   gridDim.x = (inputDim          + blockDim.x - 1) / blockDim.x  — input-column tiles
 *   gridDim.y = (batchSize * seqLen + blockDim.y - 1) / blockDim.y  — row tiles
 * Block: (blockDim.x, blockDim.y)  —  typically (BLOCKDIM, BLOCKDIM)
 *
 * Shared memory layout (contiguous extern char[]):
 *   sQGrad [blockDim.y * blockDim.x]  — tile of queryGrad_flat columns
 *   sKGrad [blockDim.y * blockDim.x]  — tile of keyGrad_flat columns
 *   sVGrad [blockDim.y * blockDim.x]  — tile of valueGrad_flat columns
 *   sQW    [blockDim.y * blockDim.x]  — tile of WQ rows (transposed load)
 *   sKW    [blockDim.y * blockDim.x]  — tile of WK rows (transposed load)
 *   sVW    [blockDim.y * blockDim.x]  — tile of WV rows (transposed load)
 * Total shared: 6 * blockDim.x * blockDim.y * sizeof(DType)
 *
 * @tparam DType Floating-point data type.
 *
 * @param inputGrad    Device pointer (accumulate via atomicAdd); shape [batchSize*seqLen, inputDim].
 * @param queryGrad    Device pointer (read); shape [batchSize, numHeads, seqLen, headDim].
 * @param keyGrad      Device pointer (read); shape [batchSize, numHeads, seqLen, headDim].
 * @param valueGrad    Device pointer (read); shape [batchSize, numHeads, seqLen, headDim].
 * @param weightsQuery Device pointer (read); shape [numHeads * headDim, inputDim].
 * @param weightsKey   Device pointer (read); shape [numHeads * headDim, inputDim].
 * @param weightsValue Device pointer (read); shape [numHeads * headDim, inputDim].
 * @param inputDim     Number of input features per token.
 * @param headDim      Dimension of each attention head.
 * @param sequenceLength Number of tokens per sequence.
 * @param numHeads     Number of attention heads.
 * @param batchSize    Number of sequences in the batch.
 */
template <typename DType = float> __global__ void qkvBackward(
    DType *inputGrad,
    const DType *queryGrad,
    const DType *keyGrad,
    const DType *valueGrad,
    const DType *weightsQuery,
    const DType *weightsKey,
    const DType *weightsValue,
    int inputDim,
    int headDim,
    int sequenceLength,
    int numHeads,
    int batchSize
) {
    extern __shared__ char block_shared_mem[];
    int totalBatchSize = batchSize * sequenceLength;
    int projDim = numHeads * headDim;

    DType *sQGrad = reinterpret_cast<DType*>(block_shared_mem);
    DType *sKGrad = sQGrad + blockDim.y * blockDim.x;
    DType *sVGrad = sKGrad + blockDim.y * blockDim.x;
    DType *sQW = sVGrad + blockDim.y * blockDim.x;
    DType *sKW = sQW + blockDim.y * blockDim.x;
    DType *sVW = sKW + blockDim.y * blockDim.x;

    int colIdx = blockIdx.x * blockDim.x + threadIdx.x; // modelDim index
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y; // totalBatchSize index

    DType accum = 0;

    for (int tile = 0; tile < projDim; tile += blockDim.x) {
        if (rowIdx < totalBatchSize && (tile + threadIdx.x) < projDim) {
            int b = rowIdx / sequenceLength;
            int s = rowIdx % sequenceLength;
            int h = (tile + threadIdx.x) / headDim;
            int d = (tile + threadIdx.x) % headDim;
            int idx = ((b * numHeads + h) * sequenceLength + s) * headDim + d;
            sQGrad[threadIdx.y * blockDim.x + threadIdx.x] = queryGrad[idx];
            sKGrad[threadIdx.y * blockDim.x + threadIdx.x] = keyGrad[idx];
            sVGrad[threadIdx.y * blockDim.x + threadIdx.x] = valueGrad[idx];
        } else {
            sQGrad[threadIdx.y * blockDim.x + threadIdx.x] = 0;
            sKGrad[threadIdx.y * blockDim.x + threadIdx.x] = 0;
            sVGrad[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }

        if (colIdx < inputDim && (tile + threadIdx.y) < projDim) {
            int wRow = tile + threadIdx.y;
            sQW[threadIdx.x * blockDim.y + threadIdx.y] = weightsQuery[wRow * inputDim + colIdx];
            sKW[threadIdx.x * blockDim.y + threadIdx.y] = weightsKey[wRow * inputDim + colIdx];
            sVW[threadIdx.x * blockDim.y + threadIdx.y] = weightsValue[wRow * inputDim + colIdx];
        } else {
            sQW[threadIdx.x * blockDim.y + threadIdx.y] = 0;
            sKW[threadIdx.x * blockDim.y + threadIdx.y] = 0;
            sVW[threadIdx.x * blockDim.y + threadIdx.y] = 0;
        }
        __syncthreads();

        for (int k = 0; k < blockDim.x; k++) {
            accum += sQGrad[threadIdx.y * blockDim.x + k] * sQW[threadIdx.x * blockDim.x + k];
            accum += sKGrad[threadIdx.y * blockDim.x + k] * sKW[threadIdx.x * blockDim.x + k];
            accum += sVGrad[threadIdx.y * blockDim.x + k] * sVW[threadIdx.x * blockDim.x + k];
        }
        __syncthreads();
    }

    if (rowIdx < totalBatchSize && colIdx < inputDim) {
        atomicAdd(&inputGrad[rowIdx * inputDim + colIdx], accum);
    }
}

#endif // ATTENTION_QKV_KERNELS