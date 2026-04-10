#include "ModuleLayer.cu"
#include "layer_norm_kernels.cu"

#ifndef LAYERNORM_LAYER
#define LAYERNORM_LAYER


/**
 * @brief Layer normalization layer.
 * Normalizes input across the last dimension and applies learnable scale and shift.
 * @tparam DType The data type used for parameters and computations.
 */
template <typename DType = float> struct LayerNormLayer : public Layer<DType> {
    int inputDim; /// Size of the last dimension to normalize over.
    int cachedN; /// Cached batch size from the last forward pass to manage dynamic allocation of mean and inv_std buffers.
    DType epsilon; /// Small constant for numerical stability in variance calculation.
    
    std::shared_ptr<DType[]> weights; /// Learnable scale parameters (gamma) for layer normalization.
    std::shared_ptr<DType[]> biases; /// Learnable shift parameters (beta) for layer normalization.
    std::shared_ptr<DType[]> weightsGrad; /// Gradient of the loss with respect to the weights (gamma).
    std::shared_ptr<DType[]> biasesGrad; /// Gradient of the loss with respect to the biases (beta).

    std::shared_ptr<DType[]> cache_mean; /// Cached mean values from the forward pass for use in backward pass.
    std::shared_ptr<DType[]> cache_inv_std; /// Cached inverse standard deviation values for use in backward pass.
    
    /** 
     * @brief Constructor for LayerNormLayer.
     * @param inputDim The size of the last dimension to normalize over (feature dimension).
     * @param epsilon A small constant for numerical stability in variance calculation.
     */
    LayerNormLayer(int inputDim, DType epsilon = 1e-5) 
        : inputDim(inputDim), epsilon(epsilon), cachedN(0) {
        weights = cudaMakeShared<DType>(inputDim, 1.0f);
        biases = cudaMakeShared<DType>(inputDim, 0.0f);
        weightsGrad = cudaMakeShared<DType>(inputDim, 0.0f);
        biasesGrad = cudaMakeShared<DType>(inputDim, 0.0f);
    }

    /**
     * @brief Returns a map of parameter names to their corresponding tensors for this layer.
     * @return std::map<std::string, Tensor<DType>>
     */
    std::map<std::string, Tensor<DType>> getParameters() override {
        return {
            {"weights", Tensor<DType>(weights, {(size_t)inputDim})},
            {"biases", Tensor<DType>(biases, {(size_t)inputDim})}
        };
    }

    /**
     * @brief Sets the parameters of the layer from a given map. 
     * This allows external code to update the weights and biases of the layer, 
     * such as during loading from a checkpoint or applying updates from an optimizer.
     * @param params A map where keys are parameter names ("weights", "biases") and values are the corresponding tensors.
     */
    void setParameters(const std::map<std::string, Tensor<DType>>& params) override {
        if (params.count("weights")) {
            cudaMemcpy(weights.get(), params.at("weights").get(), inputDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        }
        if (params.count("biases")) {
            cudaMemcpy(biases.get(), params.at("biases").get(), inputDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        }
    }

    /**
     * @brief Creates a deep copy of the layer and its parameters. 
     * This is essential for operations like cloning layers in a transformer block.
     */
    std::shared_ptr<Layer<DType>> clone() override {
        auto cloned = std::make_shared<LayerNormLayer<DType>>(inputDim, epsilon);
        cudaMemcpy(cloned->weights.get(), this->weights.get(), inputDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(cloned->biases.get(), this->biases.get(), inputDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        return cloned;
    }

    /**
     * @brief Resets the gradients for weights and biases to zero. T
     * This should called before a new backward pass to clear out old gradient values.
     */
    void zeroGrad() override {
        cudaMemset(weightsGrad.get(), 0, inputDim * sizeof(DType));
        cudaMemset(biasesGrad.get(), 0, inputDim * sizeof(DType));
    }

    /**
     * @brief Performs an in-place SGD update on the weights and biases using their gradients 
     * and the specified learning rate.
     * @param lr The learning rate to use for the SGD update.
     */
    void sgdUpdate(DType lr) override {
        layerNormUpdateKernel<<< (inputDim + 255) / 256, 256 >>>(
            weights.get(), biases.get(), weightsGrad.get(), biasesGrad.get(), inputDim, lr
        );
        cudaDeviceSynchronize();
    }

    /**
     * @brief Forward pass through the layer normalization layer.
     * This normalizes the input across the last dimension and applies the learnable scale and shift.
     * It also caches the mean and inverse standard deviation for use in the backward pass.
     * @param input The input tensor to the layer normalization layer. Shape: [batch_size, seq_len, inputDim].
     * @return The output tensor after applying layer normalization. Shape: [batch_size, seq_len, inputDim].
     */
    Tensor<DType> forward(Tensor<DType> input) override {
        int D = input.shape()[input.nDim() - 1];
        assert(D == inputDim);
        int N = input.size() / D;
        
        Tensor<DType> output(input.shape().toVector());

        if (cachedN < N) {
            cache_mean = cudaMakeShared<DType>(N);
            cache_inv_std = cudaMakeShared<DType>(N);
            cachedN = N;
        }

        int threads = 256;
        size_t smem = threads * sizeof(DType);
        layerNormForwardKernel<<<N, threads, smem>>>(
            input.get(), output.get(), weights.get(), biases.get(),
            D, N, epsilon, cache_mean.get(), cache_inv_std.get()
        );

        return output;
    }

    /**
     * @brief Executes the backward pass of the layer normalization layer to 
     * compute gradients with respect to the input, weights, and biases.
     * @param input The input tensor to the layer normalization layer. Shape: [batch_size, seq_len, inputDim].
     * @param gradOutput The gradient of the loss with respect to the output of the layer normalization layer. 
     *      Shape: [batch_size, seq_len, inputDim].
     * @return The gradient of the loss with respect to the input of the layer normalization layer. Shape: [batch_size, seq_len, inputDim].
     */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) override {
        int D = input.shape()[input.nDim() - 1];
        int N = input.size() / D;
        
        Tensor<DType> gradInput(input.shape().toVector());

        int threads = 256;
        size_t smem = 2 * threads * sizeof(DType);
        layerNormBackwardKernel<<<N, threads, smem>>>(
            gradOutput.get(), input.get(), gradInput.get(),
            weights.get(), weightsGrad.get(), biasesGrad.get(),
            cache_mean.get(), cache_inv_std.get(), D, N
        );

        return gradInput;
    }
};

/**
 * @brief Factory function to create a LayerNormLayer instance.
 * This provides a convenient way to create a layer normalization layer without directly dealing with the class.
 * @tparam DType The data type for the layer parameters and computations.
 * @param inputDim The size of the last dimension to normalize over (feature dimension).
 * @param epsilon A small constant for numerical stability in variance calculation.
 * @return Module<DType> A shared pointer to the created LayerNormLayer instance.
 */
template <typename DType> Module<DType> LayerNorm(int inputDim, DType epsilon = (DType)1e-5) {
    return std::make_shared<LayerNormLayer<DType>>(inputDim, epsilon);
}

#endif
