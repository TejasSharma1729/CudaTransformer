#include "headers.cu"

#ifndef EMBEDDING_KERNELS
#define EMBEDDING_KERNELS


template <typename DType = float, typename IdType = int> __global__ void tokenEmbeddingForwardKernel(
    const IdType* input,
    DType* output,
    const DType* embeddingMatrix,
    int vocabSize,
    int embeddingDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIdx = blockIdx.y;
    if (featureIdx < embeddingDim && batchIdx < totalBatchSize) {
        int tokenId = (int)input[batchIdx];
        if (tokenId >= 0 && tokenId < vocabSize) {
            output[batchIdx * embeddingDim + featureIdx] = embeddingMatrix[tokenId * embeddingDim + featureIdx];
        } else {
            output[batchIdx * embeddingDim + featureIdx] = static_cast<DType>(0);
        }
    }
}

template <typename DType = float, typename IdType = int> __global__ void positionEmbeddingForwardKernel(
    const IdType* input,
    DType* output,
    const DType* embeddingMatrix,
    int embeddingDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIdx = blockIdx.y;
    if (featureIdx < embeddingDim && batchIdx < totalBatchSize) {
        output[batchIdx * embeddingDim + featureIdx] = embeddingMatrix[batchIdx * embeddingDim + featureIdx];
    }
}

template <typename DType = float, typename IdType = int> __global__ void tokenEmbeddingBackwardKernel(
    const IdType* input,
    const DType* gradOutput,
    DType* gradEmbeddingMatrix,
    int vocabSize,
    int embeddingDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIdx = blockIdx.y;
    if (featureIdx < embeddingDim && batchIdx < totalBatchSize) {
        int tokenId = (int)input[batchIdx];
        if (tokenId >= 0 && tokenId < vocabSize) {
            atomicAdd(&gradEmbeddingMatrix[tokenId * embeddingDim + featureIdx], gradOutput[batchIdx * embeddingDim + featureIdx]);
        }
    }
}

template <typename DType = float, typename IdType = int> __global__ void positionEmbeddingBackwardKernel(
    const IdType* input,
    const DType* gradOutput,
    DType* gradEmbeddingMatrix,
    int embeddingDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIdx = blockIdx.y;
    if (featureIdx < embeddingDim && batchIdx < totalBatchSize) {
        atomicAdd(&gradEmbeddingMatrix[batchIdx * embeddingDim + featureIdx], gradOutput[batchIdx * embeddingDim + featureIdx]);
    }
}

// unembedding == just a linear layer.

template <typename DType = float> __global__ void softmaxKernel(
    const DType* input,
    DType* output,
    int embeddingDim,
    int totalBatchSize,
    DType temperature = (DType)1.0,
    bool logSoftmax = false
) {
    int batchIdx = blockIdx.x * blockDim.x + threadIdx.x; // one thread per feature
    if (batchIdx >= totalBatchSize) {
        return;
    }
    DType maxVal;
    if constexpr (std::is_same_v<DType, half>) {
        maxVal = -1e4;
    } else if constexpr (std::is_same_v<DType, double>) {
        maxVal = -1e10;
    }
    for (int i = 0; i < embeddingDim; i++) {
        DType val = input[batchIdx * embeddingDim + i];
        if (val > maxVal) maxVal = val;
    }
    DType sumExp = 0;
    for (int i = 0; i < embeddingDim; i++) {
        DType expVal = (DType)exp((double)((input[batchIdx * embeddingDim + i] - maxVal) / temperature));
        sumExp += expVal;
    }

    for (int i = 0; i < embeddingDim; i++) {
        DType expVal = (DType)exp((double)((input[batchIdx * embeddingDim + i] - maxVal) / temperature));
        if (logSoftmax) {
            output[batchIdx * embeddingDim + i] = input[batchIdx * embeddingDim + i] - maxVal - (DType)log((double)sumExp);
        } else {
            output[batchIdx * embeddingDim + i] = expVal / sumExp;
        }
    }
}

template <typename DType = float, typename IdType = int> __global__ void crossEntropyLossKernel(
    const DType* input,
    const IdType* target,
    DType* lossOutput,
    DType* inputGrad,
    int embeddingDim,
    int totalBatchSize,
    DType temperature = (DType)1.0,
    bool logSoftmax = false
) {
    int batchIdx = blockIdx.x * blockDim.x + threadIdx.x; // one thread per feature
    if (batchIdx >= totalBatchSize) {
        return;
    }
    int targetId = (int)target[batchIdx];
    if (targetId < 0 || targetId >= embeddingDim) {
        if (lossOutput != nullptr) {
            lossOutput[batchIdx] = (DType)0;
        }
        return;
    }
    int fullIdx = batchIdx * embeddingDim + targetId;
    double val = (double)input[fullIdx];
    if (lossOutput != nullptr) {
        if (logSoftmax) {
            lossOutput[batchIdx] = (DType)(-val);
            val = (DType)exp((double)val);
        } else {
            lossOutput[batchIdx] = (DType)(-log((double)val));
        }
    }
    
    if (inputGrad == nullptr) {
        return;
    }
    for (int i = 0; i < targetId; i++) {
        inputGrad[batchIdx * embeddingDim + i] = input[batchIdx * embeddingDim + i] / temperature;
    }
    inputGrad[fullIdx] = (DType)((val - 1.0) / (double)temperature);
    for (int i = targetId + 1; i < embeddingDim; i++) {
        inputGrad[batchIdx * embeddingDim + i] = input[batchIdx * embeddingDim + i] / temperature;
    }
}


#endif // EMBEDDING_KERNELS