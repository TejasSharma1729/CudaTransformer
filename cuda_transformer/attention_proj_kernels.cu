#include "headers.cu"

#ifndef ATTENTION_PROJECTION_KERNELS
#define ATTENTION_PROJECTION_KERNELS


/**
 * @brief Projects multi-head attention output back to the model dim using tiling.
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
 * @brief Backward pass for attention proj to compute weight and bias grad.
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
 * @brief Backward pass for attention proj to compute input grad.
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