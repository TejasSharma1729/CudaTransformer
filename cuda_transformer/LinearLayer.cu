#include "ModuleLayer.cu"
#include "linear_kernels.cu"

#ifndef LINEAR_LAYER
#define LINEAR_LAYER

/**
 * @brief Represents a fully connected linear layer.
 * 
 * @tparam DType The data type used for parameters and computations.
 */
template <typename DType = float> struct LinearLayer : public Layer<DType> {
    std::shared_ptr<DType[]> weights = nullptr;      /// Weight matrix [outputDim, inputDim].
    std::shared_ptr<DType[]> weightGrad = nullptr; ///< Accumulated grad for weights.
    std::shared_ptr<DType[]> biases = nullptr;       /// Bias vector [outputDim].
    std::shared_ptr<DType[]> biasGrad = nullptr;  /// Accumulated grad for biases.
    int inputDim = 1;  /// Size of the input feature space.
    int outputDim = 1; /// Size of the output feature space.
    
    /**
     * @brief Constructs a LinearLayer with specified dims.
     * @param inputDim Size of input features.
     * @param outputDim Size of output features.
     */
    LinearLayer(int inputDim, int outputDim) 
        : Layer<DType>(), inputDim(inputDim), outputDim(outputDim) 
    {
        this->weights = cudaMakeShared<DType>(inputDim * outputDim);
        this->biases = cudaMakeShared<DType>(outputDim);
    }

    /** @brief Clones the layer and its parameters but not the grad. */
    std::shared_ptr<Layer<DType>> clone() override {
        auto clonedLayer = std::make_shared<LinearLayer<DType>>(inputDim, outputDim);
        cudaMemcpy(clonedLayer->weights.get(), this->weights.get(), inputDim * outputDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->biases.get(), this->biases.get(), outputDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        return clonedLayer;
    }

    /**
     * @brief Performs forward linear proj: output = input * weights^T + biases.
     * @param input Input tensor.
     * @return Tensor<DType> Projected output tensor.
     */
    Tensor<DType> forward(Tensor<DType> input) override {
        assert(input.shape()[input.nDim() - 1] == inputDim);
        std::vector<size_t> os = input.shape().toVector();
        os[os.size() - 1] = outputDim;
        Tensor<DType> output(os);
        int totalBatchSize = output.size() / outputDim;

        dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
        dim3 blocksPerGrid((outputDim + BLOCKDIM - 1) / BLOCKDIM, (totalBatchSize + BLOCKDIM - 1) / BLOCKDIM);
        size_t Size = 2 * BLOCKDIM * BLOCKDIM * sizeof(DType);
        linearForward<DType><<<blocksPerGrid, threadsPerBlock, Size>>>(
            input.get(),
            output.get(),
            weights.get(),
            biases.get(),
            inputDim,
            outputDim,
            totalBatchSize
        );
        return output;
    }

    /**
     * @brief Performs backward pass to compute grad for weights, biases, and input.
     * @param input Original input tensor.
     * @param gradOutput Grad of the loss with respect to the output.
     * @return Tensor<DType> Grad of the loss with respect to the input.
     */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) override {
        assert(gradOutput.shape()[gradOutput.nDim() - 1] == outputDim);
        assert(input.shape()[input.nDim() - 1] == inputDim);
        
        int totalBatchSize = gradOutput.size() / outputDim;
        if (weightGrad == nullptr) weightGrad = cudaMakeShared<DType>(inputDim * outputDim);
        if (biasGrad == nullptr) biasGrad = cudaMakeShared<DType>(outputDim);
        
        dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
        dim3 weightGrid((inputDim + BLOCKDIM - 1) / BLOCKDIM, (outputDim + BLOCKDIM - 1) / BLOCKDIM);
        linearBackwardWB<DType><<<weightGrid, threadsPerBlock, 2 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            input.get(),
            gradOutput.get(),
            weightGrad.get(),
            biasGrad.get(),
            inputDim,
            outputDim,
            totalBatchSize
        );

        Tensor<DType> gradInput(input.shape().toVector());
        dim3 inputGrid((inputDim + BLOCKDIM - 1) / BLOCKDIM, (totalBatchSize + BLOCKDIM - 1) / BLOCKDIM);
        linearBackward<DType><<<inputGrid, threadsPerBlock, 2 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            gradInput.get(),
            gradOutput.get(), 
            weights.get(), 
            inputDim, 
            outputDim, 
            totalBatchSize
        );
        return gradInput;
    }
    /**
     * @brief Get all parameters as a map of name to Tensor.
     * @return Map of parameter names to Tensor views (shared device buffers).
     */
    std::map<std::string, Tensor<DType>> getParameters() override {
        return {
            {"weights", Tensor<DType>(weights, {(size_t)outputDim, (size_t)inputDim})},
            {"biases", Tensor<DType>(biases, {(size_t)outputDim})}
        };
    }

    /**
     * @brief Set parameters from a dictionary. Only keys matching the 2 expected parameter names are used; others are ignored.
     * @param params Map of parameter names to Tensors.
     *      The layer will adopt the device buffers of the provided Tensors for any matching keys.
     */
    void setParameters(const std::map<std::string, Tensor<DType>>& params) override {
        if (params.count("weights")) setWeights(params.at("weights"));
        if (params.count("biases")) setBiases(params.at("biases"));
    }

    /**
     * @brief Get all gradients as a map of name to Tensor.
     * Returns lazily-allocated zero tensors if backward not yet called.
     * @return Map of gradient names to Tensor views (shared device buffers).
     */
    std::map<std::string, Tensor<DType>> getGradients() override {
        if (weightGrad == nullptr) 
            weightGrad = cudaMakeShared<DType>(inputDim * outputDim);
        if (biasGrad == nullptr) 
            biasGrad = cudaMakeShared<DType>(outputDim);
        return {
            {"weights", Tensor<DType>(weightGrad, {(size_t)outputDim, (size_t)inputDim})},
            {"biases", Tensor<DType>(biasGrad, {(size_t)outputDim})}
        };
    }

    // -------------------------------------------------------------------------
    // Weight / bias shallow-copy get & set
    // -------------------------------------------------------------------------

    /** @brief Returns a Tensor that shares the underlying weight buffer (no copy). */
    Tensor<DType> getWeights() {
        return Tensor<DType>(weights, {(size_t)outputDim, (size_t)inputDim});
    }

    /** @brief Makes this layer share the buffer of `t` as its weights (no copy). */
    void setWeights(Tensor<DType> t) {
        weights = t.dataPtr();
    }

    /** @brief Returns a Tensor that shares the underlying bias buffer (no copy). */
    Tensor<DType> getBiases() {
        return Tensor<DType>(biases, {(size_t)outputDim});
    }

    /** @brief Makes this layer share the buffer of `t` as its biases (no copy). */
    void setBiases(Tensor<DType> t) {
        biases = t.dataPtr();
    }

    /** @brief Returns a Tensor that shares the weight-gradient buffer (lazy-allocated). */
    Tensor<DType> getWeightGrad() {
        if (weightGrad == nullptr) weightGrad = cudaMakeShared<DType>(inputDim * outputDim);
        return Tensor<DType>(weightGrad, {(size_t)outputDim, (size_t)inputDim});
    }

    /** @brief Makes this layer share the buffer of `t` as its weight gradient. */
    void setWeightGrad(Tensor<DType> t) {
        weightGrad = t.dataPtr();
    }

    /** @brief Returns a Tensor that shares the bias-gradient buffer (lazy-allocated). */
    Tensor<DType> getBiasGrad() {
        if (biasGrad == nullptr) biasGrad = cudaMakeShared<DType>(outputDim);
        return Tensor<DType>(biasGrad, {(size_t)outputDim});
    }

    /** @brief Makes this layer share the buffer of `t` as its bias gradient. */
    void setBiasGrad(Tensor<DType> t) {
        biasGrad = t.dataPtr();
    }

    /**
     * @brief Resets all parameter gradients to zero.
     * Gradient buffers are lazily allocated if backward has not been called.
     */
    void zeroGrad() override {
        if (weightGrad != nullptr) 
            cudaMemset(weightGrad.get(), 0, inputDim * outputDim * sizeof(DType));
        if (biasGrad != nullptr) 
            cudaMemset(biasGrad.get(), 0, outputDim * sizeof(DType));
    }

    /**
     * @brief In-place SGD step: weights -= lr * weightGrad, biases -= lr * biasGrad.
     * Gradient buffers are lazily allocated (zeros) if backward has not been called.
     * @param lr Learning rate.
     */
    void sgdUpdate(DType lr) override {
        if (weightGrad == nullptr) 
            weightGrad = cudaMakeShared<DType>(inputDim * outputDim);
        if (biasGrad == nullptr) 
            biasGrad = cudaMakeShared<DType>(outputDim);
        runSgdUpdate<DType>(weights, weightGrad, lr, inputDim * outputDim);
        runSgdUpdate<DType>(biases, biasGrad, lr, outputDim);
    }
};

/**
 * @brief Factory function to create a Linear layer module.
 * @param inputDim Size of input features.
 * @param outputDim Size of output features.
 * @return Module<DType> Shared pointer to the created layer.
 */
template <typename DType = float> Module<DType> Linear(int inputDim, int outputDim) {
    return std::make_shared<LinearLayer<DType>>(inputDim, outputDim);
}

#endif // LINEAR_LAYER