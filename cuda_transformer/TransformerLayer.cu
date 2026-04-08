#include "TransformerBlockLayer.cu"

#ifndef TRANSFORMER_LAYER
#define TRANSFORMER_LAYER

/**
 * @brief Full Transformer model consisting of multiple blocks.
 */
template <typename DType = float> struct TransformerLayer : public Layer<DType> {
    std::vector<Module<DType>> modelLayers; ///< List of transformer blocks and checkpoints.
    int inputDim = 1;
    int numHeads = 1;
    int headDim = 1;
    int mlpDim = 1;
    int numLayers = 1;
    int checkpointGap = 1;
    ActivationType activationType = ActivationType::ReLU; ///< Activation function for MLP sub-layers.

    /**
     * @brief Constructs a TransformerLayer.
     * @param inputDim Size of input features.
     * @param numHeads Number of attention heads.
     * @param headDim Dim per head.
     * @param mlpDim Hidden dim of the MLP.
     * @param numLayers Number of transformer blocks.
     * @param checkpointGap Number of layers between grad checkpoints.
     * @param activationType Activation function for the MLP (ReLU, GELU, Sigmoid, or Tanh).
     */
    TransformerLayer(
        int inputDim, int numHeads, int headDim, int mlpDim, int numLayers, 
        int checkpointGap = 0, ActivationType activationType = ActivationType::ReLU
    ) : Layer<DType>(), inputDim(inputDim), numHeads(numHeads), headDim(headDim), mlpDim(mlpDim),
        numLayers(numLayers), checkpointGap(checkpointGap), activationType(activationType)
    {
        for (int i = 0; i < numLayers; i++) {
            if (checkpointGap > 0 && i > 0 && i % checkpointGap == 0) {
                modelLayers.push_back(Checkpoint<DType>());
            }
            modelLayers.push_back(TransformerBlock<DType>(inputDim, numHeads, headDim, mlpDim, activationType));
        }
    }

    /** @brief Clones the transformer. */
    std::shared_ptr<Layer<DType>> clone() override {
        auto clonedLayer = std::make_shared<TransformerLayer<DType>>(
            inputDim, numHeads, headDim, mlpDim, numLayers, checkpointGap, activationType
        );
        clonedLayer->modelLayers.clear();
        for (auto layer : modelLayers) {
            clonedLayer->modelLayers.push_back(layer->clone());
        }
        return clonedLayer;
    }

    /** @brief Forward pass. */
    Tensor<DType> forward(Tensor<DType> input) override {
        Tensor<DType> currentOutput = input;
        for (auto layer : modelLayers) {
            currentOutput = layer->forward(currentOutput);
        }
        return currentOutput;
    }

    /** @brief Backward pass. */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) override {
        Tensor<DType> propagatedGrad = gradOutput;
        int lastLayerIdx = modelLayers.size();

        for (int l = modelLayers.size() - 1; l >= 0; l--) {
            auto checkpointPtr = std::dynamic_pointer_cast<CheckpointLayer<DType>>(modelLayers[l]);
            if (checkpointPtr == nullptr) continue;

            /**
             * @brief Execute activations operation.
             */
            std::vector<Tensor<DType>> activations(1, checkpointPtr->activationStorage);
            for (int layerIdx = l + 1; layerIdx < lastLayerIdx; layerIdx++) {
                auto modelLayer = std::dynamic_pointer_cast<TransformerBlockLayer<DType>>(modelLayers[layerIdx]);
                activations.push_back(modelLayer->firstNorm->forward(activations.back()));
                activations.push_back(modelLayer->attention->forward(activations.back()));
                activations.push_back(modelLayer->secondNorm->forward(activations.back()));
                activations.push_back(modelLayer->mlp->forward(activations.back()));
            }
            for (int layerIdx = lastLayerIdx - 1; layerIdx >= l; layerIdx--) {
                int baseActivation = 4 * (layerIdx - l);
                auto modelLayer = std::dynamic_pointer_cast<TransformerBlockLayer<DType>>(modelLayers[layerIdx]);
                propagatedGrad = modelLayer->mlp->backward(activations[baseActivation + 1], propagatedGrad);
                propagatedGrad = modelLayer->secondNorm->backward(activations[baseActivation + 2], propagatedGrad);
                propagatedGrad = modelLayer->attention->backward(activations[baseActivation], propagatedGrad);
                propagatedGrad = modelLayer->firstNorm->backward(activations[baseActivation + 3], propagatedGrad);
            }
            lastLayerIdx = l;
        }

        /**
         * @brief Execute activations operation.
         */
        std::vector<Tensor<DType>> activations(1, input);
        for (int layerIdx = 0; layerIdx < lastLayerIdx; layerIdx++) {
            auto modelLayer = std::dynamic_pointer_cast<TransformerBlockLayer<DType>>(modelLayers[layerIdx]);
            activations.push_back(modelLayer->firstNorm->forward(activations.back()));
            activations.push_back(modelLayer->attention->forward(activations.back()));
            activations.push_back(modelLayer->secondNorm->forward(activations.back()));
            activations.push_back(modelLayer->mlp->forward(activations.back()));
        }
        for (int layerIdx = lastLayerIdx - 1; layerIdx >= 0; layerIdx--) {
            int baseActivation = 2 * layerIdx;
            auto modelLayer = std::dynamic_pointer_cast<TransformerBlockLayer<DType>>(modelLayers[layerIdx]);
            propagatedGrad = modelLayer->mlp->backward(activations[baseActivation + 1], propagatedGrad);
            propagatedGrad = modelLayer->secondNorm->backward(activations[baseActivation + 2], propagatedGrad);
            propagatedGrad = modelLayer->attention->backward(activations[baseActivation], propagatedGrad);
            propagatedGrad = modelLayer->firstNorm->backward(activations[baseActivation + 3], propagatedGrad);
        }
        return propagatedGrad;
    }

    /**
     * @brief Resets all parameter gradients in every sub-layer (transformer blocks and checkpoints).
     */
    void zeroGrad() override {
        for (auto& layer : modelLayers) 
            layer->zeroGrad();
    }

    /**
     * @brief Propagates SGD update through all transformer blocks.
     * CheckpointLayer entries have no parameters and use the base no-op.
     * @param lr Learning rate.
     */
    void sgdUpdate(DType lr) override {
        for (auto &layer : modelLayers) 
            layer->sgdUpdate(lr);
    }

    /**
     * @brief Aggregates parameters from all sub-layers (transformer blocks and checkpoints).
     * Each parameter is prefixed with "layer_N." where N is the layer index.
     * Checkpoint layers have no parameters and contribute nothing.
     * @return Map of all parameters from sub-modules with indexed prefix names.
     */
    std::map<std::string, Tensor<DType>> getParameters() override {
        std::map<std::string, Tensor<DType>> params;
        for (size_t i = 0; i < modelLayers.size(); i++) {
            auto layerParams = modelLayers[i]->getParameters();
            for (const auto& [name, tensor] : layerParams) {
                params["layer_" + std::to_string(i) + "." + name] = tensor;
            }
        }
        return params;
    }

    /**
     * @brief Execute setParameters operation.
     */
    void setParameters(const std::map<std::string, Tensor<DType>>& params) override {
        for (size_t i = 0; i < modelLayers.size(); i++) {
            std::string prefix = "layer_" + std::to_string(i) + ".";
            std::map<std::string, Tensor<DType>> layerParams;
            for (const auto& [name, tensor] : params) {
                if (name.find(prefix) == 0) layerParams[name.substr(prefix.size())] = tensor;
            }
            if (!layerParams.empty()) modelLayers[i]->setParameters(layerParams);
        }
    }

    /**
     * @brief Aggregates gradients from all sub-layers (transformer blocks and checkpoints).
     * Each gradient is prefixed with "layer_N." where N is the layer index.
     * Checkpoint layers have no parameters and contribute nothing.
     * @return Map of all gradients from sub-modules with indexed prefix names.
     */
    std::map<std::string, Tensor<DType>> getGradients() override {
        std::map<std::string, Tensor<DType>> grads;
        for (size_t i = 0; i < modelLayers.size(); i++) {
            auto layerGrads = modelLayers[i]->getGradients();
            for (const auto& [name, tensor] : layerGrads) {
                grads["layer_" + std::to_string(i) + "." + name] = tensor;
            }
        }
        return grads;
    }
};

/**
 * @brief Factory function for full Transformer model.
 * @param inputDim Size of input features.
 * @param numHeads Number of attention heads.
 * @param headDim Dim per head.
 * @param mlpDim Hidden dim of the MLP.
 * @param numLayers Number of transformer blocks.
 * @param checkpointGap Gap between checkpoints.
 * @param activationType ReLU or GELU activation for the MLP.
 * @return Module<DType> Shared pointer to Transformer module.
 */
template <typename DType = float> Module<DType> Transformer(
    int inputDim, int numHeads, int headDim, int mlpDim, int numLayers, 
    int checkpointGap = 0, ActivationType activationType = ActivationType::ReLU
) {
    return std::make_shared<TransformerLayer<DType>>(
        inputDim, numHeads, headDim, mlpDim, numLayers, checkpointGap, activationType
    );
}

#endif // TRANSFORMER_LAYER