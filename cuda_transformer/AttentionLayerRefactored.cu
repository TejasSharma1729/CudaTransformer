#include "QKVProjectionLayer.cu"
#include "FlashAttentionLayer.cu"
#include "OutputProjectionLayer.cu"

#ifndef ATTENTION_LAYER_REFACTORED
#define ATTENTION_LAYER_REFACTORED

/**
 * @brief Master attention layer composing QKV projection, flash attention, and output projection.
 * @tparam DType The data type used for computations.
 */
template <typename DType = float> struct AttentionLayer : public Layer<DType> {
    std::shared_ptr<QKVProjectionLayer<DType>> qkvProj;
    std::shared_ptr<FlashAttentionLayer<DType>> flashAttn;
    std::shared_ptr<OutputProjectionLayer<DType>> outProj;

    int inputDim = 1;
    int numHeads = 1;
    int headDim = 1;

    /**
     * @brief Constructs an AttentionLayer.
     * @param inputDim Size of input/output features.
     * @param numHeads Number of attention heads.
     * @param headDim Dimension per head.
     */
    AttentionLayer(int inputDim, int numHeads, int headDim)
        : Layer<DType>(), inputDim(inputDim), numHeads(numHeads), headDim(headDim)
    {
        qkvProj = std::make_shared<QKVProjectionLayer<DType>>(inputDim, numHeads, headDim);
        flashAttn = std::make_shared<FlashAttentionLayer<DType>>(headDim, numHeads);
        outProj = std::make_shared<OutputProjectionLayer<DType>>(inputDim, headDim, numHeads);
    }

    /** @brief Clones the attention layer. */
    std::shared_ptr<Layer<DType>> clone() override {
        auto clonedLayer = std::make_shared<AttentionLayer<DType>>(inputDim, numHeads, headDim);
        clonedLayer->qkvProj = std::dynamic_pointer_cast<QKVProjectionLayer<DType>>(qkvProj->clone());
        clonedLayer->flashAttn = std::dynamic_pointer_cast<FlashAttentionLayer<DType>>(flashAttn->clone());
        clonedLayer->outProj = std::dynamic_pointer_cast<OutputProjectionLayer<DType>>(outProj->clone());
        return clonedLayer;
    }

    /** @brief Forward pass through all three sub-layers. */
    Tensor<DType> forward(Tensor<DType> input) override {
        int sequenceLength = input.shape()[input.nDim() - 2];
        int batchSize = input.size() / (sequenceLength * inputDim);
        std::vector<size_t> qkvShape = {(size_t)batchSize, (size_t)numHeads, (size_t)sequenceLength, (size_t)headDim};

        // Step 1: Project to Q, K, V
        Tensor<DType> queries(qkvShape), keys(qkvShape), values(qkvShape);
        dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
        dim3 gridQKV((numHeads * headDim + BLOCKDIM - 1) / BLOCKDIM, (batchSize * sequenceLength + BLOCKDIM - 1) / BLOCKDIM);
        getQKVmatrices<DType><<<gridQKV, threadsPerBlock, 4 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            input.get(), queries.get(), qkvProj->weightsQuery.get(), qkvProj->biasesQuery.get(),
            keys.get(), qkvProj->weightsKey.get(), qkvProj->biasesKey.get(),
            values.get(), qkvProj->weightsValue.get(), qkvProj->biasesValue.get(),
            inputDim, headDim, sequenceLength, numHeads, batchSize
        );

        // Step 2: Compute flash attention (with caching for backward)
        Tensor<DType> attentionOutput = flashAttn->computeAttention(queries, keys, values);

        // Step 3: Project output
        // Reshape attention output for projection: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads*head_dim]
        std::vector<size_t> reshapedShape = {(size_t)batchSize, (size_t)sequenceLength, (size_t)(numHeads * headDim)};
        Tensor<DType> reshapedAttn(reshapedShape);
        // Simple reshape by copying (could be optimized with view)
        cudaMemcpy(reshapedAttn.get(), attentionOutput.get(), attentionOutput.size() * sizeof(DType), cudaMemcpyDeviceToDevice);

        Tensor<DType> finalOutput = outProj->forward(reshapedAttn);

        return finalOutput;
    }

    /** @brief Backward pass through all three sub-layers. */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) override {
        int sequenceLength = input.shape()[input.nDim() - 2];
        int batchSize = input.size() / (sequenceLength * inputDim);
        std::vector<size_t> qkvShape = {(size_t)batchSize, (size_t)numHeads, (size_t)sequenceLength, (size_t)headDim};

        // Recompute forward pass for activations
        Tensor<DType> queries(qkvShape), keys(qkvShape), values(qkvShape);
        dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
        dim3 gridQKV((numHeads * headDim + BLOCKDIM - 1) / BLOCKDIM, (batchSize * sequenceLength + BLOCKDIM - 1) / BLOCKDIM);
        getQKVmatrices<DType><<<gridQKV, threadsPerBlock, 4 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            input.get(), queries.get(), qkvProj->weightsQuery.get(), qkvProj->biasesQuery.get(),
            keys.get(), qkvProj->weightsKey.get(), qkvProj->biasesKey.get(),
            values.get(), qkvProj->weightsValue.get(), qkvProj->biasesValue.get(),
            inputDim, headDim, sequenceLength, numHeads, batchSize
        );

        Tensor<DType> attentionOutput = flashAttn->computeAttention(queries, keys, values);
        std::vector<size_t> reshapedShape = {(size_t)batchSize, (size_t)sequenceLength, (size_t)(numHeads * headDim)};
        Tensor<DType> reshapedAttn(reshapedShape);
        cudaMemcpy(reshapedAttn.get(), attentionOutput.get(), attentionOutput.size() * sizeof(DType), cudaMemcpyDeviceToDevice);

        // Backward through output projection
        Tensor<DType> attnGrad = outProj->backward(reshapedAttn, gradOutput);

        // Reshape gradient back: [batch, seq_len, num_heads*head_dim] -> [batch, num_heads, seq_len, head_dim]
        Tensor<DType> reshapedAttnGrad(qkvShape);
        cudaMemcpy(reshapedAttnGrad.get(), attnGrad.get(), attnGrad.size() * sizeof(DType), cudaMemcpyDeviceToDevice);

        // Backward through flash attention using cached values
        Tensor<DType> queryGrad(qkvShape), keyGrad(qkvShape), valueGrad(qkvShape);
        flashAttn->computeBackward(queries, queryGrad, keys, keyGrad, values, valueGrad, reshapedAttnGrad);

        // Initialize gradient buffers if needed
        if (qkvProj->weightsQueryGrad == nullptr) {
            int projSize = inputDim * headDim * numHeads;
            int biasSize = headDim * numHeads;
            qkvProj->weightsQueryGrad = cudaMakeShared<DType>(projSize);
            qkvProj->biasesQueryGrad = cudaMakeShared<DType>(biasSize);
            qkvProj->weightsKeyGrad = cudaMakeShared<DType>(projSize);
            qkvProj->biasesKeyGrad = cudaMakeShared<DType>(biasSize);
            qkvProj->weightsValueGrad = cudaMakeShared<DType>(projSize);
            qkvProj->biasesValueGrad = cudaMakeShared<DType>(biasSize);
        }

        // Backward through QKV projection
        int totalInputDim = numHeads * headDim;
        dim3 gridQKVWB((inputDim + BLOCKDIM - 1) / BLOCKDIM, (totalInputDim + BLOCKDIM - 1) / BLOCKDIM);
        qkvBackwardWB<DType><<<gridQKVWB, threadsPerBlock, 4 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            input.get(), queryGrad.get(), qkvProj->weightsQueryGrad.get(), qkvProj->biasesQueryGrad.get(),
            keyGrad.get(), qkvProj->weightsKeyGrad.get(), qkvProj->biasesKeyGrad.get(),
            valueGrad.get(), qkvProj->weightsValueGrad.get(), qkvProj->biasesValueGrad.get(),
            inputDim, headDim, sequenceLength, numHeads, batchSize
        );

        Tensor<DType> inputGrad(input.shape().toVector());
        dim3 gridQKVInput((inputDim + BLOCKDIM - 1) / BLOCKDIM, (batchSize * sequenceLength + BLOCKDIM - 1) / BLOCKDIM);
        qkvBackward<DType><<<gridQKVInput, threadsPerBlock, 6 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            inputGrad.get(), queryGrad.get(), keyGrad.get(), valueGrad.get(),
            qkvProj->weightsQuery.get(), qkvProj->weightsKey.get(), qkvProj->weightsValue.get(),
            inputDim, headDim, sequenceLength, numHeads, batchSize
        );

        flashAttn->clearCache();
        return inputGrad;
    }
};

/**
 * @brief Factory function for Attention layer.
 * @param inputDim Size of input features.
 * @param numHeads Number of attention heads.
 * @param headDim Dimension per head.
 * @return Module<DType> Shared pointer to Attention module.
 */
template <typename DType = float> Module<DType> Attention(int inputDim, int numHeads, int headDim) {
    return std::make_shared<AttentionLayer<DType>>(inputDim, numHeads, headDim);
}

#endif // ATTENTION_LAYER_REFACTORED
