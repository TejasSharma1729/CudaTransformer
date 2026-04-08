#include "headers.cu"

#ifndef LINEAR_KERNELS
#define LINEAR_KERNELS

/**
 * @brief Forward pass for a Linear layer using 2D matrix tiling.
 * 
 * Grid: (outputDim / blockDim.x, batchSize / blockDim.y)
 * Block: (blockDim.x, blockDim.y)
 * 
 * @tparam DType The data type for computations.
 * @param input Pointer to the input matrix [batchSize, inputDim].
 * @param output Pointer to the output matrix [batchSize, outputDim].
 * @param weights Pointer to the weights matrix [outputDim, inputDim].
 * @param biases Pointer to the bias vector [outputDim].
 * @param inputDim Number of input features.
 * @param outputDim Number of output features.
 * @param batchSize Number of examples in the batch.
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
 * @brief Backward pass for Linear layer weights and biases.
 * 
 * Tiles over the batch dim to accumulate grad for weights and biases.
 * 
 * @tparam DType The data type for computations.
 * @param input Pointer to the original input [batchSize, inputDim].
 * @param outputGrad Pointer to the grad from the next layer [batchSize, outputDim].
 * @param weightsGrad Pointer to accumulate weight grad [outputDim, inputDim].
 * @param biasesGrad Pointer to accumulate bias grad [outputDim].
 * @param inputDim Number of input features.
 * @param outputDim Number of output features.
 * @param batchSize Number of examples in the batch.
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
 * @brief Backward pass for Linear layer input grad.
 * 
 * Computes grad with respect to input by multiplying output grad with transposed weights.
 * 
 * @tparam DType The data type for computations.
 * @param inputGrad Pointer to accumulate input grad [batchSize, inputDim].
 * @param outputGrad Pointer to the grad from the next layer [batchSize, outputDim].
 * @param weights Pointer to the layer weights [outputDim, inputDim].
 * @param inputDim Number of input features.
 * @param outputDim Number of output features.
 * @param batchSize Number of examples in the batch.
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