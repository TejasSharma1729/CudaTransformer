#include "headers.cu"

#ifndef ATTENTION_QKV_KERNELS
#define ATTENTION_QKV_KERNELS

/**
 * @brief Tiled proj from input to Query, Key, and Value matrices.
 * 
 * Grid: ((numHeads * headDim) / blockDim.x, (batchSize * sequenceLength) / blockDim.y)
 * Block: (blockDim.x, blockDim.y)
 * 
 * @tparam DType The data type for computations.
 * @param input Pointer to input data [batchSize * sequenceLength, inputDim].
 * @param queries Pointer to output Query matrix [batchSize, numHeads, sequenceLength, headDim].
 * @param weightsQuery Pointer to Query weights [numHeads * headDim, inputDim].
 * @param biasesQuery Pointer to Query biases [numHeads * headDim].
 * @param keys Pointer to output Key matrix [batchSize, numHeads, sequenceLength, headDim].
 * @param weightsKey Pointer to Key weights [numHeads * headDim, inputDim].
 * @param biasesKey Pointer to Key biases [numHeads * headDim].
 * @param values Pointer to output Value matrix [batchSize, numHeads, sequenceLength, headDim].
 * @param weightsValue Pointer to Value weights [numHeads * headDim, inputDim].
 * @param biasesValue Pointer to Value biases [numHeads * headDim].
 * @param inputDim Dim of the input features.
 * @param headDim Dim of each attention head.
 * @param sequenceLength Length of the input sequence.
 * @param numHeads Number of attention heads.
 * @param batchSize Number of sequences in the batch.
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
 * @brief Backward pass for Q, K, V projs to compute weight and bias grad.
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
 * @brief Backward pass for Q, K, V projs to compute input grad.
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