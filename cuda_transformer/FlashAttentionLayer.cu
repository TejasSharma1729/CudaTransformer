#include "ModuleLayer.cu"
#include "flash_attention.cu"

#ifndef FLASH_ATTENTION_LAYER
#define FLASH_ATTENTION_LAYER

/**
 * @brief Flash attention core utility with cached max scores and denominators.
 * NOT a Layer - just a computation utility for AttentionLayer to use.
 * Caches values for efficient backward pass without recomputing forward.
 * @tparam DType The data type used for computations.
 */
template <typename DType = float> struct FlashAttentionLayer {
    int headDim = 1;
    int numHeads = 1;

    // Cached values from forward pass for backward
    std::shared_ptr<DType[]> cachedMaxScores = nullptr;
    std::shared_ptr<DType[]> cachedDenominators = nullptr;
    std::shared_ptr<DType[]> cachedOutput = nullptr;
    int cachedBatchSize = 0;
    int cachedSequenceLength = 0;

    /**
     * @brief Constructs a FlashAttentionLayer.
     * @param headDim Dimension per attention head.
     * @param numHeads Number of attention heads.
     */
    FlashAttentionLayer(int headDim, int numHeads)
        : headDim(headDim), numHeads(numHeads)
    {
    }

    /** @brief Clone the flash attention layer. */
    std::shared_ptr<FlashAttentionLayer<DType>> clone() const {
        return std::make_shared<FlashAttentionLayer<DType>>(headDim, numHeads);
    }

    /**
     * @brief Core forward: compute attention with caching.
     * @param queries [batch, num_heads, seq_len, head_dim]
     * @param keys [batch, num_heads, seq_len, head_dim]
     * @param values [batch, num_heads, seq_len, head_dim]
     * @return Attention output [batch, num_heads, seq_len, head_dim]
     */
    Tensor<DType> computeAttention(const Tensor<DType> &queries, const Tensor<DType> &keys, const Tensor<DType> &values) {
        int batchSize = queries.shape()[0];
        int sequenceLength = queries.shape()[2];
        std::vector<size_t> outputShape = queries.shape().toVector();
        Tensor<DType> output(outputShape);

        // Cache max scores and denominators for backward
        cachedMaxScores = cudaMakeShared<DType>(batchSize * numHeads * sequenceLength);
        cachedDenominators = cudaMakeShared<DType>(batchSize * numHeads * sequenceLength);
        cachedOutput = cudaMakeShared<DType>(output.size());
        cachedBatchSize = batchSize;
        cachedSequenceLength = sequenceLength;

        dim3 gridAtt((sequenceLength + BLOCKDIM - 1) / BLOCKDIM, numHeads, batchSize);
        dim3 blockAtt(BLOCKDIM, BLOCKDIM);
        size_t sharedMemSize = BLOCKDIM * (BLOCKDIM + 3 * headDim) * sizeof(DType);
        flashAttention<DType><<<gridAtt, blockAtt, sharedMemSize>>>(
            queries.get(), keys.get(), values.get(), output.get(),
            cachedMaxScores.get(), cachedDenominators.get(), headDim, sequenceLength, numHeads, batchSize
        );

        // Also cache output for backward
        cudaMemcpy(cachedOutput.get(), output.get(), output.size() * sizeof(DType), cudaMemcpyDeviceToDevice);

        return output;
    }

    /**
     * @brief Backward: compute gradients using cached values.
     * @return Tuple of gradient tensors: [queryGrad, keyGrad, valueGrad]
     */
    void computeBackward(
        const Tensor<DType> &queries, Tensor<DType> &queryGrad,
        const Tensor<DType> &keys, Tensor<DType> &keyGrad,
        const Tensor<DType> &values, Tensor<DType> &valueGrad,
        const Tensor<DType> &outputGrad
    ) {
        if (cachedMaxScores == nullptr || cachedDenominators == nullptr) {
            // Forward was not called or cache was cleared
            return;
        }

        int batchSize = cachedBatchSize;
        int sequenceLength = cachedSequenceLength;

        dim3 gridAtt((sequenceLength + BLOCKDIM - 1) / BLOCKDIM, numHeads, batchSize);
        dim3 blockAtt(BLOCKDIM, BLOCKDIM);
        size_t sharedMemSize = (5 * headDim * BLOCKDIM + 2 * BLOCKDIM * BLOCKDIM) * sizeof(DType);

        flashAttentionBackward<DType><<<gridAtt, blockAtt, sharedMemSize>>>(
            queries.get(), queryGrad.get(),
            keys.get(), keyGrad.get(),
            values.get(), valueGrad.get(),
            cachedOutput.get(), outputGrad.get(),
            cachedMaxScores.get(), cachedDenominators.get(),
            headDim, sequenceLength, numHeads, batchSize
        );
    }

    /** @brief Clear cached values to free memory. */
    void clearCache() {
        cachedMaxScores = nullptr;
        cachedDenominators = nullptr;
        cachedOutput = nullptr;
    }
};

#endif // FLASH_ATTENTION_LAYER

