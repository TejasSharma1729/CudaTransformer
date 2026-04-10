#include "ModuleLayer.cu"
#include "attention_qkv_kernels.cu"

#include <array>

#ifndef QKV_PROJECTION_LAYER
#define QKV_PROJECTION_LAYER

/**
 * @brief QKV projection layer for multi-head attention.
 * NOT a Layer - internal utility used by AttentionLayer.
 * Projects input to Q, K, V and reshapes to multi-head format.
 * @tparam DType The data type used for computations.
 */
template <typename DType = float> struct QKVProjectionLayer {
    std::shared_ptr<DType[]> weightsQuery = nullptr; /// Weight matrix for query projection [inputDim, headDim * numHeads].
    std::shared_ptr<DType[]> weightsQueryGrad = nullptr; /// Accumulated gradient for query weights.
    std::shared_ptr<DType[]> biasesQuery = nullptr; /// Bias vector for query projection [headDim * numHeads].
    std::shared_ptr<DType[]> biasesQueryGrad = nullptr; /// Accumulated gradient for query biases.

    std::shared_ptr<DType[]> weightsKey = nullptr; /// Weight matrix for key projection [inputDim, headDim * numHeads].
    std::shared_ptr<DType[]> weightsKeyGrad = nullptr; /// Accumulated gradient for key weights.
    std::shared_ptr<DType[]> biasesKey = nullptr; /// Bias vector for key projection [headDim * numHeads].
    std::shared_ptr<DType[]> biasesKeyGrad = nullptr; /// Accumulated gradient for key biases.

    std::shared_ptr<DType[]> weightsValue = nullptr; /// Weight matrix for value projection [inputDim, headDim * numHeads].
    std::shared_ptr<DType[]> weightsValueGrad = nullptr; /// Accumulated gradient for value weights.
    std::shared_ptr<DType[]> biasesValue = nullptr; /// Bias vector for value projection [headDim * numHeads].
    std::shared_ptr<DType[]> biasesValueGrad = nullptr; /// Accumulated gradient for value biases.

    int inputDim = 1; /// Size of the input feature space.
    int numHeads = 1; /// Number of attention heads.
    int headDim = 1; /// Dimension of each attention head (inputDim / numHeads).

    /**
     * @brief Constructs a QKVProjectionLayer.
     * @param inputDim Size of input features.
     * @param numHeads Number of attention heads.
     * @param headDim Dimension per head.
     */
    QKVProjectionLayer(int inputDim, int numHeads, int headDim)
        : inputDim(inputDim), numHeads(numHeads), headDim(headDim)
    {
        int projSize = inputDim * headDim * numHeads;
        int biasSize = headDim * numHeads;
        weightsQuery = cudaMakeShared<DType>(projSize);
        biasesQuery = cudaMakeShared<DType>(biasSize);
        weightsKey = cudaMakeShared<DType>(projSize);
        biasesKey = cudaMakeShared<DType>(biasSize);
        weightsValue = cudaMakeShared<DType>(projSize);
        biasesValue = cudaMakeShared<DType>(biasSize);
    }

    /** @brief Clones the QKV projection layer. */
    std::shared_ptr<QKVProjectionLayer<DType>> clone() const {
        auto clonedLayer = std::make_shared<QKVProjectionLayer<DType>>(inputDim, numHeads, headDim);
        int projSize = inputDim * headDim * numHeads;
        int biasSize = headDim * numHeads;
        cudaMemcpy(clonedLayer->weightsQuery.get(), this->weightsQuery.get(), projSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->biasesQuery.get(), this->biasesQuery.get(), biasSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->weightsKey.get(), this->weightsKey.get(), projSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->biasesKey.get(), this->biasesKey.get(), biasSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->weightsValue.get(), this->weightsValue.get(), projSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->biasesValue.get(), this->biasesValue.get(), biasSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        return clonedLayer;
    }

    /**
     * @brief Computes Q, K, and V matrices from input.
     * @param input The input tensor [batch, seq_len, inputDim].
     * @return Array containing Q, K, V tensors.
     */
    std::array<Tensor<DType>, 3> computeQKV(const Tensor<DType>& input) {
        int sequenceLength = input.shape()[input.nDim() - 2];
        int batchSize = input.size() / (sequenceLength * inputDim);
        std::vector<size_t> qkvShape = {(size_t)batchSize, (size_t)numHeads, (size_t)sequenceLength, (size_t)headDim};

        Tensor<DType> queries(qkvShape), keys(qkvShape), values(qkvShape);
        dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
        dim3 gridQKV((numHeads * headDim + BLOCKDIM - 1) / BLOCKDIM, (batchSize * sequenceLength + BLOCKDIM - 1) / BLOCKDIM);
        
        getQKVmatrices<DType><<<gridQKV, threadsPerBlock, 4 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            input.get(), queries.get(), weightsQuery.get(), biasesQuery.get(),
            keys.get(), weightsKey.get(), biasesKey.get(),
            values.get(), weightsValue.get(), biasesValue.get(),
            inputDim, headDim, sequenceLength, numHeads, batchSize
        );
        
        return {queries, keys, values};
    }

    /**
     * @brief Backward pass for QKV projection layer.
     * @param input The original input tensor.
     * @param queryGrad Gradient w.r.t queries.
     * @param keyGrad Gradient w.r.t keys.
     * @param valueGrad Gradient w.r.t values.
     * @return Gradient w.r.t input tensor.
     */
    Tensor<DType> computeBackward(
        const Tensor<DType>& input, 
        const Tensor<DType>& queryGrad, 
        const Tensor<DType>& keyGrad, 
        const Tensor<DType>& valueGrad
    ) {
        int sequenceLength = input.shape()[input.nDim() - 2];
        int batchSize = input.size() / (sequenceLength * inputDim);

        if (weightsQueryGrad == nullptr) {
            int projSize = inputDim * headDim * numHeads;
            int biasSize = headDim * numHeads;
            weightsQueryGrad = cudaMakeShared<DType>(projSize);
            biasesQueryGrad = cudaMakeShared<DType>(biasSize);
            weightsKeyGrad = cudaMakeShared<DType>(projSize);
            biasesKeyGrad = cudaMakeShared<DType>(biasSize);
            weightsValueGrad = cudaMakeShared<DType>(projSize);
            biasesValueGrad = cudaMakeShared<DType>(biasSize);
        }

        dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
        int totalInputDim = numHeads * headDim;
        dim3 gridQKVWB((inputDim + BLOCKDIM - 1) / BLOCKDIM, (totalInputDim + BLOCKDIM - 1) / BLOCKDIM);

        qkvBackwardWB<DType><<<gridQKVWB, threadsPerBlock, 4 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            input.get(), queryGrad.get(), weightsQueryGrad.get(), biasesQueryGrad.get(),
            keyGrad.get(), weightsKeyGrad.get(), biasesKeyGrad.get(),
            valueGrad.get(), weightsValueGrad.get(), biasesValueGrad.get(),
            inputDim, headDim, sequenceLength, numHeads, batchSize
        );

        Tensor<DType> inputGrad(input.shape().toVector());
        dim3 gridQKVInput((inputDim + BLOCKDIM - 1) / BLOCKDIM, (batchSize * sequenceLength + BLOCKDIM - 1) / BLOCKDIM);
        
        qkvBackward<DType><<<gridQKVInput, threadsPerBlock, 6 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            inputGrad.get(), queryGrad.get(), keyGrad.get(), valueGrad.get(),
            weightsQuery.get(), weightsKey.get(), weightsValue.get(),
            inputDim, headDim, sequenceLength, numHeads, batchSize
        );

        return inputGrad;
    }
};

#endif // QKV_PROJECTION_LAYER
