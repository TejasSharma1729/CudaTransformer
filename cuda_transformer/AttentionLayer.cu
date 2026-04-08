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
    QKVProjectionLayer<DType> qkvProj;
    FlashAttentionLayer<DType> flashAttn;
    OutputProjectionLayer<DType> outProj;

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
        : Layer<DType>(),
          qkvProj(inputDim, numHeads, headDim),
          flashAttn(headDim, numHeads),
          outProj(inputDim, headDim, numHeads),
          inputDim(inputDim), numHeads(numHeads), headDim(headDim)
    {
    }

    /** @brief Clones the attention layer. */
    std::shared_ptr<Layer<DType>> clone() override {
        auto clonedLayer = std::make_shared<AttentionLayer<DType>>(inputDim, numHeads, headDim);

        // Clone QKV projection weights
        int projSize = inputDim * headDim * numHeads;
        int biasSize = headDim * numHeads;
        cudaMemcpy(clonedLayer->qkvProj.weightsQuery.get(), this->qkvProj.weightsQuery.get(), projSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->qkvProj.biasesQuery.get(), this->qkvProj.biasesQuery.get(), biasSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->qkvProj.weightsKey.get(), this->qkvProj.weightsKey.get(), projSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->qkvProj.biasesKey.get(), this->qkvProj.biasesKey.get(), biasSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->qkvProj.weightsValue.get(), this->qkvProj.weightsValue.get(), projSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->qkvProj.biasesValue.get(), this->qkvProj.biasesValue.get(), biasSize * sizeof(DType), cudaMemcpyDeviceToDevice);

        // Clone output projection weights
        int totalHeadDim = headDim * numHeads;
        cudaMemcpy(clonedLayer->outProj.weightsProj.get(), this->outProj.weightsProj.get(), totalHeadDim * inputDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->outProj.biasesProj.get(), this->outProj.biasesProj.get(), inputDim * sizeof(DType), cudaMemcpyDeviceToDevice);

        return clonedLayer;
    }

    /** @brief Forward pass through all three sub-components. */
    Tensor<DType> forward(Tensor<DType> input) override {
        int sequenceLength = input.shape()[input.nDim() - 2];
        int batchSize = input.size() / (sequenceLength * inputDim);
        std::vector<size_t> qkvShape = {(size_t)batchSize, (size_t)numHeads, (size_t)sequenceLength, (size_t)headDim};

        // Step 1: Project to Q, K, V
        Tensor<DType> queries(qkvShape), keys(qkvShape), values(qkvShape);
        dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
        dim3 gridQKV((numHeads * headDim + BLOCKDIM - 1) / BLOCKDIM, (batchSize * sequenceLength + BLOCKDIM - 1) / BLOCKDIM);
        getQKVmatrices<DType><<<gridQKV, threadsPerBlock, 4 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            input.get(), queries.get(), qkvProj.weightsQuery.get(), qkvProj.biasesQuery.get(),
            keys.get(), qkvProj.weightsKey.get(), qkvProj.biasesKey.get(),
            values.get(), qkvProj.weightsValue.get(), qkvProj.biasesValue.get(),
            inputDim, headDim, sequenceLength, numHeads, batchSize
        );

        // Step 2: Compute flash attention (with caching for backward)
        Tensor<DType> attentionOutput = flashAttn.computeAttention(queries, keys, values);

        // Step 3: Project output
        // Reshape attention output for projection: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads*head_dim]
        std::vector<size_t> reshapedShape = {(size_t)batchSize, (size_t)sequenceLength, (size_t)(numHeads * headDim)};
        Tensor<DType> reshapedAttn(reshapedShape);
        cudaMemcpy(reshapedAttn.get(), attentionOutput.get(), attentionOutput.size() * sizeof(DType), cudaMemcpyDeviceToDevice);

        Tensor<DType> finalOutput = outProj.forward(reshapedAttn);

        return finalOutput;
    }

    /** @brief Backward pass through all three sub-components. */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) override {
        int sequenceLength = input.shape()[input.nDim() - 2];
        int batchSize = input.size() / (sequenceLength * inputDim);
        std::vector<size_t> qkvShape = {(size_t)batchSize, (size_t)numHeads, (size_t)sequenceLength, (size_t)headDim};

        // Recompute forward pass for activations
        Tensor<DType> queries(qkvShape), keys(qkvShape), values(qkvShape);
        dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
        dim3 gridQKV((numHeads * headDim + BLOCKDIM - 1) / BLOCKDIM, (batchSize * sequenceLength + BLOCKDIM - 1) / BLOCKDIM);
        getQKVmatrices<DType><<<gridQKV, threadsPerBlock, 4 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            input.get(), queries.get(), qkvProj.weightsQuery.get(), qkvProj.biasesQuery.get(),
            keys.get(), qkvProj.weightsKey.get(), qkvProj.biasesKey.get(),
            values.get(), qkvProj.weightsValue.get(), qkvProj.biasesValue.get(),
            inputDim, headDim, sequenceLength, numHeads, batchSize
        );

        Tensor<DType> attentionOutput = flashAttn.computeAttention(queries, keys, values);
        std::vector<size_t> reshapedShape = {(size_t)batchSize, (size_t)sequenceLength, (size_t)(numHeads * headDim)};

        Tensor<DType> reshapedAttn(reshapedShape);
        cudaMemcpy(reshapedAttn.get(), attentionOutput.get(), attentionOutput.size() * sizeof(DType), cudaMemcpyDeviceToDevice);

        // Backward through output projection
        Tensor<DType> attnGrad = outProj.backward(reshapedAttn, gradOutput);

        // Reshape gradient back: [batch, seq_len, num_heads*head_dim] -> [batch, num_heads, seq_len, head_dim]
        Tensor<DType> reshapedAttnGrad(qkvShape);
        cudaMemcpy(reshapedAttnGrad.get(), attnGrad.get(), attnGrad.size() * sizeof(DType), cudaMemcpyDeviceToDevice);

        // Backward through flash attention using cached values
        Tensor<DType> queryGrad(qkvShape), keyGrad(qkvShape), valueGrad(qkvShape);
        flashAttn.computeBackward(queries, queryGrad, keys, keyGrad, values, valueGrad, reshapedAttnGrad);

        // Initialize gradient buffers if needed
        if (qkvProj.weightsQueryGrad == nullptr) {
            int projSize = inputDim * headDim * numHeads;
            int biasSize = headDim * numHeads;
            qkvProj.weightsQueryGrad = cudaMakeShared<DType>(projSize);
            qkvProj.biasesQueryGrad = cudaMakeShared<DType>(biasSize);
            qkvProj.weightsKeyGrad = cudaMakeShared<DType>(projSize);
            qkvProj.biasesKeyGrad = cudaMakeShared<DType>(biasSize);
            qkvProj.weightsValueGrad = cudaMakeShared<DType>(projSize);
            qkvProj.biasesValueGrad = cudaMakeShared<DType>(biasSize);
        }

        // Backward through QKV projection
        int totalInputDim = numHeads * headDim;
        dim3 gridQKVWB((inputDim + BLOCKDIM - 1) / BLOCKDIM, (totalInputDim + BLOCKDIM - 1) / BLOCKDIM);
        qkvBackwardWB<DType><<<gridQKVWB, threadsPerBlock, 4 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            input.get(), queryGrad.get(), qkvProj.weightsQueryGrad.get(), qkvProj.biasesQueryGrad.get(),
            keyGrad.get(), qkvProj.weightsKeyGrad.get(), qkvProj.biasesKeyGrad.get(),
            valueGrad.get(), qkvProj.weightsValueGrad.get(), qkvProj.biasesValueGrad.get(),
            inputDim, headDim, sequenceLength, numHeads, batchSize
        );

        Tensor<DType> inputGrad(input.shape().toVector());
        dim3 gridQKVInput((inputDim + BLOCKDIM - 1) / BLOCKDIM, (batchSize * sequenceLength + BLOCKDIM - 1) / BLOCKDIM);
        qkvBackward<DType><<<gridQKVInput, threadsPerBlock, 6 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            inputGrad.get(), queryGrad.get(), keyGrad.get(), valueGrad.get(),
            qkvProj.weightsQuery.get(), qkvProj.weightsKey.get(), qkvProj.weightsValue.get(),
            inputDim, headDim, sequenceLength, numHeads, batchSize
        );

        flashAttn.clearCache();
        return inputGrad;
    }

    // -------------------------------------------------------------------------
    // QKV projection weight / bias get & set (shallow copy)
    // All getters return a Tensor that shares the underlying device buffer — no copy.
    // All setters redirect the layer's shared_ptr to the caller's buffer — no copy.
    // -------------------------------------------------------------------------

    /**
     * @brief Returns a Tensor view of the Query projection weight matrix [inputDim, headDim*numHeads].
     * The returned Tensor shares the device buffer; modifying it modifies the layer's weights.
     */
    Tensor<DType> getQueryWeights() {
        return Tensor<DType>(qkvProj.weightsQuery, {(size_t)inputDim, (size_t)(headDim * numHeads)});
    }
    /**
     * @brief Replaces the Query projection weight buffer with the buffer owned by `t` (shallow).
     * @param t Tensor whose device buffer this layer will adopt as its Query weights.
     */
    void setQueryWeights(Tensor<DType> t) { qkvProj.weightsQuery = t.dataPtr(); }

    /**
     * @brief Returns a Tensor view of the Query projection bias vector [headDim*numHeads].
     * The returned Tensor shares the device buffer; modifying it modifies the layer's biases.
     */
    Tensor<DType> getQueryBiases() {
        return Tensor<DType>(qkvProj.biasesQuery, {(size_t)(headDim * numHeads)});
    }
    /**
     * @brief Replaces the Query projection bias buffer with the buffer owned by `t` (shallow).
     * @param t Tensor whose device buffer this layer will adopt as its Query biases.
     */
    void setQueryBiases(Tensor<DType> t) { qkvProj.biasesQuery = t.dataPtr(); }

    /**
     * @brief Returns a Tensor view of the Key projection weight matrix [inputDim, headDim*numHeads].
     * The returned Tensor shares the device buffer; modifying it modifies the layer's weights.
     */
    Tensor<DType> getKeyWeights() {
        return Tensor<DType>(qkvProj.weightsKey, {(size_t)inputDim, (size_t)(headDim * numHeads)});
    }
    /**
     * @brief Replaces the Key projection weight buffer with the buffer owned by `t` (shallow).
     * @param t Tensor whose device buffer this layer will adopt as its Key weights.
     */
    void setKeyWeights(Tensor<DType> t) { qkvProj.weightsKey = t.dataPtr(); }

    /**
     * @brief Returns a Tensor view of the Key projection bias vector [headDim*numHeads].
     * The returned Tensor shares the device buffer; modifying it modifies the layer's biases.
     */
    Tensor<DType> getKeyBiases() {
        return Tensor<DType>(qkvProj.biasesKey, {(size_t)(headDim * numHeads)});
    }
    /**
     * @brief Replaces the Key projection bias buffer with the buffer owned by `t` (shallow).
     * @param t Tensor whose device buffer this layer will adopt as its Key biases.
     */
    void setKeyBiases(Tensor<DType> t) { qkvProj.biasesKey = t.dataPtr(); }

    /**
     * @brief Returns a Tensor view of the Value projection weight matrix [inputDim, headDim*numHeads].
     * The returned Tensor shares the device buffer; modifying it modifies the layer's weights.
     */
    Tensor<DType> getValueWeights() {
        return Tensor<DType>(qkvProj.weightsValue, {(size_t)inputDim, (size_t)(headDim * numHeads)});
    }
    /**
     * @brief Replaces the Value projection weight buffer with the buffer owned by `t` (shallow).
     * @param t Tensor whose device buffer this layer will adopt as its Value weights.
     */
    void setValueWeights(Tensor<DType> t) { qkvProj.weightsValue = t.dataPtr(); }

    /**
     * @brief Returns a Tensor view of the Value projection bias vector [headDim*numHeads].
     * The returned Tensor shares the device buffer; modifying it modifies the layer's biases.
     */
    Tensor<DType> getValueBiases() {
        return Tensor<DType>(qkvProj.biasesValue, {(size_t)(headDim * numHeads)});
    }
    /**
     * @brief Replaces the Value projection bias buffer with the buffer owned by `t` (shallow).
     * @param t Tensor whose device buffer this layer will adopt as its Value biases.
     */
    void setValueBiases(Tensor<DType> t) { qkvProj.biasesValue = t.dataPtr(); }

    // -------------------------------------------------------------------------
    // Output projection weight / bias get & set (shallow copy)
    // -------------------------------------------------------------------------

    /**
     * @brief Returns a Tensor view of the output projection weight matrix [headDim*numHeads, inputDim].
     * The returned Tensor shares the device buffer; modifying it modifies the layer's weights.
     */
    Tensor<DType> getOutputWeights() {
        return Tensor<DType>(outProj.weightsProj, {(size_t)(headDim * numHeads), (size_t)inputDim});
    }
    /**
     * @brief Replaces the output projection weight buffer with the buffer owned by `t` (shallow).
     * @param t Tensor whose device buffer this layer will adopt as its output projection weights.
     */
    void setOutputWeights(Tensor<DType> t) { outProj.weightsProj = t.dataPtr(); }

    /**
     * @brief Returns a Tensor view of the output projection bias vector [inputDim].
     * The returned Tensor shares the device buffer; modifying it modifies the layer's biases.
     */
    Tensor<DType> getOutputBiases() {
        return Tensor<DType>(outProj.biasesProj, {(size_t)inputDim});
    }
    /**
     * @brief Replaces the output projection bias buffer with the buffer owned by `t` (shallow).
     * @param t Tensor whose device buffer this layer will adopt as its output projection biases.
     */
    void setOutputBiases(Tensor<DType> t) { outProj.biasesProj = t.dataPtr(); }

    /**
     * @brief Resets all parameter gradients to zero.
     * Gradient buffers are lazily allocated if backward has not been called.
     */
    void zeroGrad() override {
        int projSize = inputDim * headDim * numHeads;
        int biasSize = headDim * numHeads;
        int totalHeadDim = headDim * numHeads;

        if (qkvProj.weightsQueryGrad != nullptr) 
            cudaMemset(qkvProj.weightsQueryGrad.get(), 0, projSize * sizeof(DType));
        if (qkvProj.weightsKeyGrad != nullptr) 
            cudaMemset(qkvProj.weightsKeyGrad.get(), 0, projSize * sizeof(DType));
        if (qkvProj.weightsValueGrad != nullptr) 
            cudaMemset(qkvProj.weightsValueGrad.get(), 0, projSize * sizeof(DType));
        if (qkvProj.biasesQueryGrad != nullptr) 
            cudaMemset(qkvProj.biasesQueryGrad.get(), 0, biasSize * sizeof(DType));
        if (qkvProj.biasesKeyGrad != nullptr) 
            cudaMemset(qkvProj.biasesKeyGrad.get(), 0, biasSize * sizeof(DType));
        if (qkvProj.biasesValueGrad != nullptr) 
            cudaMemset(qkvProj.biasesValueGrad.get(), 0, biasSize * sizeof(DType));
        if (outProj.weightsProjGrad != nullptr) 
            cudaMemset(outProj.weightsProjGrad.get(), 0, totalHeadDim * inputDim * sizeof(DType));
        if (outProj.biasesProjGrad != nullptr) 
            cudaMemset(outProj.biasesProjGrad.get(), 0, inputDim * sizeof(DType));
    }

    /**
     * @brief In-place SGD step for all QKV and output projection parameters.
     * @param lr Learning rate.
     */
    void sgdUpdate(DType lr) override {
        int projSize = inputDim * headDim * numHeads;
        int biasSize = headDim * numHeads;

        // Lazily allocate grad buffers if backward was never called
        if (qkvProj.weightsQueryGrad == nullptr) 
            qkvProj.weightsQueryGrad = cudaMakeShared<DType>(projSize);
        if (qkvProj.biasesQueryGrad  == nullptr) 
            qkvProj.biasesQueryGrad  = cudaMakeShared<DType>(biasSize);
        if (qkvProj.weightsKeyGrad   == nullptr) 
            qkvProj.weightsKeyGrad   = cudaMakeShared<DType>(projSize);
        if (qkvProj.biasesKeyGrad    == nullptr) 
            qkvProj.biasesKeyGrad    = cudaMakeShared<DType>(biasSize);
        if (qkvProj.weightsValueGrad == nullptr) 
            qkvProj.weightsValueGrad = cudaMakeShared<DType>(projSize);
        if (qkvProj.biasesValueGrad  == nullptr) 
            qkvProj.biasesValueGrad  = cudaMakeShared<DType>(biasSize);

        int totalHeadDim = headDim * numHeads;
        if (outProj.weightsProjGrad == nullptr) 
            outProj.weightsProjGrad = cudaMakeShared<DType>(totalHeadDim * inputDim);
        if (outProj.biasesProjGrad  == nullptr) 
            outProj.biasesProjGrad  = cudaMakeShared<DType>(inputDim);

        runSgdUpdate<DType>(qkvProj.weightsQuery, qkvProj.weightsQueryGrad, lr, projSize);
        runSgdUpdate<DType>(qkvProj.biasesQuery,  qkvProj.biasesQueryGrad,  lr, biasSize);
        runSgdUpdate<DType>(qkvProj.weightsKey,   qkvProj.weightsKeyGrad,   lr, projSize);
        runSgdUpdate<DType>(qkvProj.biasesKey,    qkvProj.biasesKeyGrad,    lr, biasSize);
        runSgdUpdate<DType>(qkvProj.weightsValue, qkvProj.weightsValueGrad, lr, projSize);
        runSgdUpdate<DType>(qkvProj.biasesValue,  qkvProj.biasesValueGrad,  lr, biasSize);
        runSgdUpdate<DType>(outProj.weightsProj,  outProj.weightsProjGrad,  lr, totalHeadDim * inputDim);
        runSgdUpdate<DType>(outProj.biasesProj,   outProj.biasesProjGrad,   lr, inputDim);
    }

    /**
     * @brief Returns all QKV and output projection parameters.
     * @return Map of parameter names to Tensor views for all 8 weight/bias buffers.
     */
    std::map<std::string, Tensor<DType>> getParameters() override {
        return {
            {"query.weights",  Tensor<DType>(qkvProj.weightsQuery, {(size_t)inputDim, (size_t)(headDim * numHeads)})},
            {"query.biases",   Tensor<DType>(qkvProj.biasesQuery, {(size_t)(headDim * numHeads)})},
            {"key.weights",    Tensor<DType>(qkvProj.weightsKey, {(size_t)inputDim, (size_t)(headDim * numHeads)})},
            {"key.biases",     Tensor<DType>(qkvProj.biasesKey, {(size_t)(headDim * numHeads)})},
            {"value.weights",  Tensor<DType>(qkvProj.weightsValue, {(size_t)inputDim, (size_t)(headDim * numHeads)})},
            {"value.biases",   Tensor<DType>(qkvProj.biasesValue, {(size_t)(headDim * numHeads)})},
            {"output.weights", Tensor<DType>(outProj.weightsProj, {(size_t)(headDim * numHeads), (size_t)inputDim})},
            {"output.biases",  Tensor<DType>(outProj.biasesProj, {(size_t)inputDim})}
        };
    }

    /**
     * @brief Sets parameters from a dictionary. Only keys matching the 8 expected parameter names are used; others are ignored.
     * @param params Map of parameter names to Tensors. 
     *      The layer will adopt the device buffers of the provided Tensors for any matching keys.
     */
    void setParameters(const std::map<std::string, Tensor<DType>>& params) override {
        if (params.count("query.weights")) setQueryWeights(params.at("query.weights"));
        if (params.count("query.biases")) setQueryBiases(params.at("query.biases"));
        if (params.count("key.weights")) setKeyWeights(params.at("key.weights"));
        if (params.count("key.biases")) setKeyBiases(params.at("key.biases"));
        if (params.count("value.weights")) setValueWeights(params.at("value.weights"));
        if (params.count("value.biases")) setValueBiases(params.at("value.biases"));
        if (params.count("output.weights")) setOutputWeights(params.at("output.weights"));
        if (params.count("output.biases")) setOutputBiases(params.at("output.biases"));
    }

    /**
     * @brief Returns all QKV and output projection gradients (lazy-allocated).
     * @return Map of gradient names to Tensor views for all 8 weight/bias gradients.
     */
    std::map<std::string, Tensor<DType>> getGradients() override {
        int projSize = inputDim * headDim * numHeads;
        int biasSize = headDim * numHeads;
        int totalHeadDim = headDim * numHeads;

        // Lazy allocation
        if (qkvProj.weightsQueryGrad == nullptr) 
            qkvProj.weightsQueryGrad = cudaMakeShared<DType>(projSize);
        if (qkvProj.biasesQueryGrad  == nullptr) 
            qkvProj.biasesQueryGrad  = cudaMakeShared<DType>(biasSize);
        if (qkvProj.weightsKeyGrad   == nullptr) 
            qkvProj.weightsKeyGrad   = cudaMakeShared<DType>(projSize);
        if (qkvProj.biasesKeyGrad    == nullptr) 
            qkvProj.biasesKeyGrad    = cudaMakeShared<DType>(biasSize);
        if (qkvProj.weightsValueGrad == nullptr) 
            qkvProj.weightsValueGrad = cudaMakeShared<DType>(projSize);
        if (qkvProj.biasesValueGrad  == nullptr) 
            qkvProj.biasesValueGrad  = cudaMakeShared<DType>(biasSize);
        if (outProj.weightsProjGrad  == nullptr) 
            outProj.weightsProjGrad  = cudaMakeShared<DType>(totalHeadDim * inputDim);
        if (outProj.biasesProjGrad   == nullptr) 
            outProj.biasesProjGrad   = cudaMakeShared<DType>(inputDim);

        return {
            {"query.weights",  Tensor<DType>(qkvProj.weightsQueryGrad, {(size_t)inputDim, (size_t)(headDim * numHeads)})},
            {"query.biases",   Tensor<DType>(qkvProj.biasesQueryGrad, {(size_t)(headDim * numHeads)})},
            {"key.weights",    Tensor<DType>(qkvProj.weightsKeyGrad, {(size_t)inputDim, (size_t)(headDim * numHeads)})},
            {"key.biases",     Tensor<DType>(qkvProj.biasesKeyGrad, {(size_t)(headDim * numHeads)})},
            {"value.weights",  Tensor<DType>(qkvProj.weightsValueGrad, {(size_t)inputDim, (size_t)(headDim * numHeads)})},
            {"value.biases",   Tensor<DType>(qkvProj.biasesValueGrad, {(size_t)(headDim * numHeads)})},
            {"output.weights", Tensor<DType>(outProj.weightsProjGrad, {(size_t)(headDim * numHeads), (size_t)inputDim})},
            {"output.biases",  Tensor<DType>(outProj.biasesProjGrad, {(size_t)inputDim})}
        };
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
