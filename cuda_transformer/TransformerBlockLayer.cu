#include "AttentionLayer.cu"
#include "MLPLayer.cu"

#ifndef TRANSFORMER_BLOCK_LAYER
#define TRANSFORMER_BLOCK_LAYER


/**
 * @brief Represents a single Transformer block (Attention + MLP).
 */
template <typename DType = float> struct TransformerBlockLayer : public Layer<DType> {
    Module<DType> attention = nullptr; ///< Self-attention module.
    Module<DType> mlp = nullptr;       ///< Feed-forward MLP module.
    int inputDim = 1;
    int numHeads = 1;
    int headDim = 1;
    int mlpDim = 1;
    ActivationType activationType = ActivationType::ReLU; ///< Activation function for the MLP sub-layer.

    /**
     * @brief Constructs a TransformerBlockLayer.
     * @param inputDim Size of input features.
     * @param numHeads Number of attention heads.
     * @param headDim Dim per head.
     * @param mlpDim Hidden dim of the MLP.
     * @param activationType Activation function for the MLP (ReLU, GELU, Sigmoid, or Tanh).
     */
    TransformerBlockLayer(
        int inputDim, int numHeads, int headDim, int mlpDim, 
        ActivationType activationType = ActivationType::ReLU
    ) : Layer<DType>(), inputDim(inputDim), numHeads(numHeads), headDim(headDim),
        mlpDim(mlpDim), activationType(activationType)
    {
        this->attention = Attention<DType>(inputDim, numHeads, headDim);
        this->mlp = MLP<DType>({
            Linear<DType>(inputDim, mlpDim),
            Activation<DType>(mlpDim, activationType),
            Linear<DType>(mlpDim, inputDim)
        });
    }

    /** @brief Clones the block. */
    std::shared_ptr<Layer<DType>> clone() override {
        auto clonedLayer = std::make_shared<TransformerBlockLayer<DType>>(
            inputDim, numHeads, headDim, mlpDim, activationType
        );
        clonedLayer->attention = attention->clone();
        clonedLayer->mlp = mlp->clone();
        return clonedLayer;
    }

    /** @brief Forward pass. */
    Tensor<DType> forward(Tensor<DType> input) override {
        input = attention->forward(input);
        input = mlp->forward(input);
        return input;
    }

    /** @brief Backward pass. */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) override {
        Tensor<DType> intermediateActivation = attention->forward(input);
        Tensor<DType> propagatedGrad = mlp->backward(intermediateActivation, gradOutput);
        propagatedGrad = attention->backward(input, propagatedGrad);
        return propagatedGrad;
    }

    /**
     * @brief Propagates SGD update to the attention and MLP sub-modules.
     * @param lr Learning rate.
     */
    void sgdUpdate(DType lr) override {
        attention->sgdUpdate(lr);
        mlp->sgdUpdate(lr);
    }

    /**
     * @brief Aggregates parameters from attention and MLP sub-modules.
     * Prefixes each parameter name with "attention." or "mlp." to avoid collisions.
     * @return Map of all parameters from sub-modules with prefixed names.
     */
    std::map<std::string, Tensor<DType>> getParameters() override {
        std::map<std::string, Tensor<DType>> params;

        auto attentionParams = attention->getParameters();
        for (const auto& [name, tensor] : attentionParams) {
            params["attention." + name] = tensor;
        }

        auto mlpParams = mlp->getParameters();
        for (const auto& [name, tensor] : mlpParams) {
            params["mlp." + name] = tensor;
        }

        return params;
    }

    /**
     * @brief Aggregates gradients from attention and MLP sub-modules.
     * Prefixes each gradient name with "attention." or "mlp." to avoid collisions.
     * @return Map of all gradients from sub-modules with prefixed names.
     */
    std::map<std::string, Tensor<DType>> getGradients() override {
        std::map<std::string, Tensor<DType>> grads;

        auto attentionGrads = attention->getGradients();
        for (const auto& [name, tensor] : attentionGrads) {
            grads["attention." + name] = tensor;
        }

        auto mlpGrads = mlp->getGradients();
        for (const auto& [name, tensor] : mlpGrads) {
            grads["mlp." + name] = tensor;
        }

        return grads;
    }
};

/**
 * @brief Factory function for Transformer Block.
 * @param inputDim Size of input features.
 * @param numHeads Number of attention heads.
 * @param headDim Dim per head.
 * @param mlpDim Hidden dim of the MLP.
 * @param activationType Activation function for the MLP (ReLU, GELU, Sigmoid, or Tanh).
 * @return Module<DType> Shared pointer to TransformerBlock module.
 */
template <typename DType = float> Module<DType> TransformerBlock(
    int inputDim, int numHeads, int headDim, int mlpDim, 
    ActivationType activationType = ActivationType::ReLU
) {
    return std::make_shared<TransformerBlockLayer<DType>>(
        inputDim, numHeads, headDim, mlpDim, activationType
    );
}

#endif // TRANSFORMER_BLOCK_LAYER