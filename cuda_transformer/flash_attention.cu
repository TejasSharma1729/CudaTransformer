#include "headers.cu"

#ifndef FLASH_ATTENTION_KERNELS
#define FLASH_ATTENTION_KERNELS

/**
 * @brief Tiled Flash Attention forward pass using shared memory tiles.
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
        score /= (DType)sqrt((double)(headDim));
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
        DType rescale = (DType)exp((double)maxScore - (double)newMax);
        DType currentExp = (DType)exp((double)score - (double)newMax);
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
 * @brief Backward pass for Flash Attention.
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
        score = (DType)exp((double)score - (double)maxScore) / denominator;
        dProd = score * (dProd - delta) / (DType)sqrt((double)(headDim));
        
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