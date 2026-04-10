#include "headers.cu"

#ifndef FLASH_ATTENTION_KERNELS
#define FLASH_ATTENTION_KERNELS

/**
 * @brief Tiled Flash Attention forward pass using shared memory and the online-softmax algorithm.
 *
 * Implements the Flash Attention forward pass with O(seqLen) memory complexity by
 * processing keys and values in tiles and maintaining running max and denominator
 * accumulators per query row (the "online softmax" trick).  No full seqLen×seqLen
 * attention score matrix is ever materialised.
 *
 * Each thread block is responsible for a contiguous tile of query rows (`blockDim.y`
 * rows) for one (batch, head) pair.  Within the tile, `threadIdx.y` selects the query
 * row and `threadIdx.x` selects the key/value column within the tile.
 *
 * Grid:
 *   gridDim.x = (sequenceLength + blockDim.y - 1) / blockDim.y   — tiles along seq dim
 *   gridDim.y = numHeads
 *   gridDim.z = batchSize
 * Block: (blockDim.x, blockDim.y)  —  typically (BLOCKDIM, BLOCKDIM)
 *
 * Shared memory layout (contiguous, allocated as a single extern char[]):
 *   tileQuery    [blockDim.y * headDim]    — query tile for this block's rows
 *   tileKey      [blockDim.y * headDim]    — key tile loaded each outer iteration
 *   tileValue    [blockDim.y * headDim]    — value tile loaded each outer iteration
 *   tileScores   [blockDim.y * blockDim.x] — raw or exp-scaled dot-product scores
 * Total shared memory: (3 * blockDim.y * headDim + blockDim.y * blockDim.x) * sizeof(DType)
 *
 * Algorithm per query row `sequenceIdx`:
 *   1. Load query[sequenceIdx] into tileQuery.
 *   2. For each tile of keys/values:
 *      a. Compute scaled dot-product scores: score = Q·K / sqrt(headDim).
 *      b. Find the tile row-max; update running max with rescale.
 *      c. Compute exp(score − newMax) and accumulate into runningDenominator.
 *      d. Accumulate exp-weighted values into runningWeightedSum[].
 *   3. Write output[sequenceIdx] = runningWeightedSum / runningDenominator.
 *   4. Optionally cache maxScores and denominators for the backward pass.
 *
 * @tparam DType  Floating-point data type (float, double, __half, __nv_bfloat16).
 *
 * @param queries      Device pointer; shape [batchSize, numHeads, sequenceLength, headDim].
 * @param keys         Device pointer; shape [batchSize, numHeads, sequenceLength, headDim].
 * @param values       Device pointer; shape [batchSize, numHeads, sequenceLength, headDim].
 * @param output       Device pointer (write); shape [batchSize, numHeads, sequenceLength, headDim].
 * @param maxScores    Device pointer (write, nullable); shape [batchSize, numHeads, sequenceLength].
 *                     Stores the final running max per query row for the backward pass.
 *                     Written only by threadIdx.x == 0.  Pass nullptr to skip.
 * @param denominators Device pointer (write, nullable); shape [batchSize, numHeads, sequenceLength].
 *                     Stores the final softmax denominator per query row.  Pass nullptr to skip.
 * @param headDim        Dimension of each attention head.  Must be ≤ 8 * blockDim.x (≤ 256 typically).
 * @param sequenceLength Number of tokens in the sequence.
 * @param numHeads       Number of attention heads.
 * @param batchSize      Number of sequences in the batch.
 */
template <typename DType = float> __global__ void flashAttention(
    const DType *queries,
    const DType *keys,
    const DType *values,
    DType *output,
    DType *maxScores,
    DType *denominators,
    int headDim,
    int sequenceLength,
    int numHeads,
    int batchSize
) {
    extern __shared__ char attn_shared_all[];
    DType *tileQuery = reinterpret_cast<DType*>(attn_shared_all);
    DType *tileKey = tileQuery + blockDim.y * headDim;
    DType *tileValue = tileKey + blockDim.y * headDim;
    DType *tileScores = tileValue + blockDim.y * headDim;

    int headIdx = blockIdx.y;
    int batchIdx = blockIdx.z;
    int globalRowOffset = (batchIdx * numHeads + headIdx) * sequenceLength;
    int globalOffset = globalRowOffset * headDim;
    int sequenceIdx = blockIdx.x * blockDim.y + threadIdx.y;

    if (sequenceIdx < sequenceLength) {
        for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
            tileQuery[threadIdx.y * headDim + i] = queries[globalOffset + sequenceIdx * headDim + i];
        }
    }
    __syncthreads();

    DType maxScore; // Large negative, but supports half.
    if constexpr (std::is_same_v<DType, half>) {
        maxScore = (DType)(-6e4f);
    } else {
        maxScore = (DType)(-1e20f);  
    }
    DType runningDenominator = (DType)0.0f;
    DType runningWeightedSum[8] = {0}; // Support headDim up to 256

    for (int tileIdx = 0; tileIdx < sequenceLength; tileIdx += blockDim.y) {
        int currentSequenceIdx = tileIdx + threadIdx.y;
        if (currentSequenceIdx < sequenceLength) {
            for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
                tileKey[threadIdx.y * headDim + i] = keys[globalOffset + currentSequenceIdx * headDim + i];
                tileValue[threadIdx.y * headDim + i] = values[globalOffset + currentSequenceIdx * headDim + i];
            }
        } else {
            for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
                tileKey[threadIdx.y * headDim + i] = (DType)0.0f;
                tileValue[threadIdx.y * headDim + i] = (DType)0.0f;
            }
        }
        __syncthreads();
        
        DType score = 0.0f;
        for (int i = 0; i < headDim; i++) {
            score += tileQuery[threadIdx.y * headDim + i] * tileKey[threadIdx.x * headDim + i];
        }
        score /= (DType)sqrt((ComputeType<DType>)(headDim));
        tileScores[threadIdx.y * blockDim.x + threadIdx.x] = score;
        __syncthreads();

        DType rowMax;
        if constexpr (std::is_same_v<DType, half>) {
            rowMax = static_cast<half>(-6e4f);
        } else {
            rowMax = static_cast<DType>(-1e20f);
        }
        for (int i = 0; i < blockDim.x; i++) {
            rowMax = (DType)fmaxf((float)rowMax, (float)tileScores[threadIdx.y * blockDim.x + i]);
        }
        
        DType newMax = (DType)fmaxf((float)maxScore, (float)rowMax);
        DType rescale = (DType)exp((ComputeType<DType>)maxScore - (ComputeType<DType>)newMax);
        DType currentExp = (DType)exp((ComputeType<DType>)score - (ComputeType<DType>)newMax);
        tileScores[threadIdx.y * blockDim.x + threadIdx.x] = currentExp;
        __syncthreads();

        DType rowExpSum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            rowExpSum += tileScores[threadIdx.y * blockDim.x + i];
        }

        runningDenominator = runningDenominator * rescale + rowExpSum;
        for (int k = 0; k < (headDim + blockDim.x - 1) / blockDim.x; k++) {
            int dIdx = k * blockDim.x + threadIdx.x;
            runningWeightedSum[k] *= rescale;
            if (dIdx < headDim) {
                DType localWeightedSum = 0.0f;
                for (int j = 0; j < blockDim.y; j++) {
                    localWeightedSum += tileScores[threadIdx.y * blockDim.x + j] * tileValue[j * headDim + dIdx];
                }
                runningWeightedSum[k] += localWeightedSum;
            }
        }
        maxScore = newMax;
        __syncthreads();
    }

    if (sequenceIdx < sequenceLength) {
        for (int k = 0; k < (headDim + blockDim.x - 1) / blockDim.x; k++) {
            int dIdx = k * blockDim.x + threadIdx.x;
            if (dIdx < headDim) {
                output[globalOffset + sequenceIdx * headDim + dIdx] = runningWeightedSum[k] / runningDenominator;
            }
        }
        if (threadIdx.x == 0 && maxScores != nullptr && denominators != nullptr) {
            maxScores[globalRowOffset + sequenceIdx] = maxScore;
            denominators[globalRowOffset + sequenceIdx] = runningDenominator;
        }
    }
}


/**
 * @brief Tiled Flash Attention backward pass.
 *
 * Recomputes attention weights on-the-fly from the cached maxScores and denominators
 * (written by the forward pass) and accumulates gradients for queries, keys, and values.
 * Like the forward pass, no full seqLen×seqLen matrix is ever materialised.
 *
 * Each thread block handles a tile of query rows (`blockDim.y` rows) for one
 * (batch, head) pair.  For each tile of key/value rows the block computes:
 *   - softmax weight:  s_{ij} = exp(Q_i·K_j / √d − maxScore_i) / denominator_i
 *   - "delta" scalar:  δ_i = Σ_d output_i[d] * gradOutput_i[d]
 *   - attention-score gradient:  dProd_{ij} = s_{ij} * (gradOutput_i · V_j − δ_i) / √d
 *   - dQ_i  += Σ_j dProd_{ij} * K_j
 *   - dK_j  += Σ_i dProd_{ij} * Q_i
 *   - dV_j  += Σ_i s_{ij} * gradOutput_i
 * All gradient writes are done with atomicAdd to handle overlapping tiles safely.
 *
 * Grid (same as forward):
 *   gridDim.x = (sequenceLength + blockDim.y - 1) / blockDim.y
 *   gridDim.y = numHeads
 *   gridDim.z = batchSize
 * Block: (blockDim.x, blockDim.y)  —  typically (BLOCKDIM, BLOCKDIM)
 *
 * Shared memory layout (contiguous, allocated as a single extern char[]):
 *   tileQuery      [blockDim.y * headDim]    — query rows for this block
 *   tileKey        [blockDim.y * headDim]    — key tile (each outer iteration)
 *   tileValue      [blockDim.y * headDim]    — value tile (each outer iteration)
 *   tileOutput     [blockDim.y * headDim]    — output rows for this block
 *   tileOutputGrad [blockDim.y * headDim]    — upstream gradient rows for this block
 *   tileScores     [blockDim.y * blockDim.x] — softmax attention weights s_{ij}
 *   tileProdGrad   [blockDim.y * blockDim.x] — dProd_{ij} (attention-score gradients)
 * Total shared: (5 * blockDim.y * headDim + 2 * blockDim.y * blockDim.x) * sizeof(DType)
 *
 * @tparam DType  Floating-point data type (float, double, __half, __nv_bfloat16).
 *
 * @param queries      Device pointer (read);  shape [batchSize, numHeads, sequenceLength, headDim].
 * @param queriesGrad  Device pointer (accumulate via atomicAdd); shape [batchSize, numHeads, sequenceLength, headDim].
 * @param keys         Device pointer (read);  shape [batchSize, numHeads, sequenceLength, headDim].
 * @param keysGrad     Device pointer (accumulate via atomicAdd); shape [batchSize, numHeads, sequenceLength, headDim].
 * @param values       Device pointer (read);  shape [batchSize, numHeads, sequenceLength, headDim].
 * @param valuesGrad   Device pointer (accumulate via atomicAdd); shape [batchSize, numHeads, sequenceLength, headDim].
 * @param output       Device pointer (read);  shape [batchSize, numHeads, sequenceLength, headDim].
 *                     Must be the output stored during the forward pass.
 * @param outputGrad   Device pointer (read);  shape [batchSize, numHeads, sequenceLength, headDim].
 *                     Upstream gradient dL/d(output).
 * @param maxScores    Device pointer (read);  shape [batchSize, numHeads, sequenceLength].
 *                     Per-row running max from the forward pass.
 * @param denominators Device pointer (read);  shape [batchSize, numHeads, sequenceLength].
 *                     Per-row softmax denominator from the forward pass.
 * @param headDim        Dimension of each attention head.
 * @param sequenceLength Number of tokens in the sequence.
 * @param numHeads       Number of attention heads.
 * @param batchSize      Number of sequences in the batch.
 */
template <typename DType = float> __global__ void flashAttentionBackward(
    const DType *queries,
    DType *queriesGrad,
    const DType *keys,
    DType *keysGrad,
    const DType *values,
    DType *valuesGrad,
    const DType *output,
    const DType *outputGrad,
    const DType *maxScores,
    const DType *denominators,
    int headDim,
    int sequenceLength,
    int numHeads,
    int batchSize
) {
    extern __shared__ char attn_shared_all[];
    DType *tileQuery = reinterpret_cast<DType*>(attn_shared_all);
    DType *tileKey = tileQuery + blockDim.y * headDim;
    DType *tileValue = tileKey + blockDim.y * headDim;
    DType *tileOutput = tileValue + blockDim.y * headDim;
    DType *tileOutputGrad = tileOutput + blockDim.y * headDim;
    DType *tileScores = tileOutputGrad + blockDim.y * headDim;
    DType *tileProdGrad = tileScores + blockDim.y * blockDim.x;

    int headIdx = blockIdx.y;
    int batchIdx = blockIdx.z;
    int globalRowOffset = (batchIdx * numHeads + headIdx) * sequenceLength;
    int globalOffset = globalRowOffset * headDim;
    int sequenceIdx = blockIdx.x * blockDim.y + threadIdx.y;

    if (sequenceIdx < sequenceLength) {
        for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
            tileQuery[threadIdx.y * headDim + i] = queries[globalOffset + sequenceIdx * headDim + i];
            tileOutput[threadIdx.y * headDim + i] = output[globalOffset + sequenceIdx * headDim + i];
            tileOutputGrad[threadIdx.y * headDim + i] = outputGrad[globalOffset + sequenceIdx * headDim + i];
        }
    }

    DType maxScore = (sequenceIdx < sequenceLength) ? maxScores[globalRowOffset + sequenceIdx] : (DType)0.0f;
    DType denominator = (sequenceIdx < sequenceLength) ? denominators[globalRowOffset + sequenceIdx] : (DType)1.0f;
    DType delta = 0.0f;
    
    if (sequenceIdx < sequenceLength) {
        for (int i = 0; i < headDim; i++) {
            delta += tileOutput[threadIdx.y * headDim + i] * tileOutputGrad[threadIdx.y * headDim + i];
        }
    }
    __syncthreads();

    for (int tileIdx = 0; tileIdx < sequenceLength; tileIdx += blockDim.y) {
        int currentSequenceIdx = tileIdx + threadIdx.y;
        if (currentSequenceIdx < sequenceLength) {
            for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
                tileKey[threadIdx.y * headDim + i] = keys[globalOffset + currentSequenceIdx * headDim + i];
                tileValue[threadIdx.y * headDim + i] = values[globalOffset + currentSequenceIdx * headDim + i];
            }
        } else {
            for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
                tileKey[threadIdx.y * headDim + i] = (DType)(0.0f);
                tileValue[threadIdx.y * headDim + i] = (DType)(0.0f);
            }
        }
        __syncthreads();
        
        DType score = 0.0f;
        DType dProd = 0.0f;
        for (int i = 0; i < headDim; i++) {
            score += tileQuery[threadIdx.y * headDim + i] * tileKey[threadIdx.x * headDim + i];
            dProd += tileOutputGrad[threadIdx.y * headDim + i] * tileValue[threadIdx.x * headDim + i];
        }
        score /= (DType)sqrt((double)(headDim));
        score = (DType)exp((ComputeType<DType>)score - (ComputeType<DType>)maxScore) / denominator;
        dProd = score * (dProd - delta) / (DType)sqrt((ComputeType<DType>)(headDim));
        
        tileScores[threadIdx.y * blockDim.x + threadIdx.x] = score;
        tileProdGrad[threadIdx.y * blockDim.x + threadIdx.x] = dProd;
        __syncthreads();

        for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
            DType dQuery = 0.0f;
            DType dKey = 0.0f;
            DType dValue = 0.0f;
            
            for (int k = 0; k < blockDim.y; k++) {
                dQuery += tileProdGrad[threadIdx.y * blockDim.x + k] * tileKey[k * headDim + i];
                dKey += tileProdGrad[threadIdx.y * blockDim.x + k] * tileQuery[threadIdx.y * headDim + i];
                dValue += tileScores[threadIdx.y * blockDim.x + k] * tileOutputGrad[threadIdx.y * headDim + i];
            }
            
            if (sequenceIdx < sequenceLength) {
                atomicAdd(&queriesGrad[globalOffset + sequenceIdx * headDim + i], dQuery);
            }
            if (currentSequenceIdx < sequenceLength) {
                atomicAdd(&keysGrad[globalOffset + currentSequenceIdx * headDim + i], dKey);
                atomicAdd(&valuesGrad[globalOffset + currentSequenceIdx * headDim + i], dValue);
            }
        }
        __syncthreads();
    }
}

#endif // FLASH_ATTENTION_KERNELS