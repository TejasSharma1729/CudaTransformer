#include "LinearLayer.cu"
#include "ActivationLayer.cu"
#include "CheckpointLayer.cu"
#include "LayerNorm.cu"

#ifndef MLP_LAYER
#define MLP_LAYER


/**
 * @brief Multi-Layer Perceptron (MLP) container.
 */
template <typename DType = float> struct MLPLayer : public Layer<DType> {
    std::vector<Module<DType>> modelLayers; ///< Sequential list of sub-modules.

    /**
     * @brief Constructs an MLP with a sequence of layers.
     * @param layers Vector of shared pointers to layers.
     */
    MLPLayer(std::vector<Module<DType>> layers) : modelLayers(layers) { }

    /** @brief Clones the MLP and all its sub-layers. */
    std::shared_ptr<Layer<DType>> clone() override {
        std::vector<Module<DType>> clonedLayers;
        for (auto layer : modelLayers) {
            clonedLayers.push_back(layer->clone());
        }
        return std::make_shared<MLPLayer<DType>>(clonedLayers);
    }

    /** @brief Sequential forward pass through all sub-layers. */
    Tensor<DType> forward(Tensor<DType> input) override {
        Tensor<DType> currentOutput = input;
        for (auto layer : modelLayers) currentOutput = layer->forward(currentOutput);
        return currentOutput;
    }

    /**
     * @brief Sequential backward pass, handling grad checkpointing if present.
     * @param input original input to the MLP.
     * @param gradOutput grad of the loss with respect to the final output.
     * @return Tensor<DType> grad of the loss with respect to the MLP input.
     */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) override {
        Tensor<DType> propagatedGrad = gradOutput;
        int lastLayerIdx = modelLayers.size();

        // Handle checkpointed segments in reverse
        for (int l = modelLayers.size() - 1; l >= 0; l--) {
            auto checkpointPtr = std::dynamic_pointer_cast<CheckpointLayer<DType>>(modelLayers[l]);
            if (checkpointPtr == nullptr) continue;

            /**
             * @brief Execute activations operation.
             */
            std::vector<Tensor<DType>> activations(1, checkpointPtr->activationStorage);
            for (int layerIdx = l + 1; layerIdx < lastLayerIdx; layerIdx++) {
                activations.push_back(modelLayers[layerIdx]->forward(activations.back()));
            }
            for (int layerIdx = lastLayerIdx - 1; layerIdx >= l; layerIdx--) {
                propagatedGrad = modelLayers[layerIdx]->backward(activations[layerIdx - l], propagatedGrad);
            }
            lastLayerIdx = l;
        }

        // Handle the remaining (non-checkpointed) prefix
        std::vector<Tensor<DType>> activations(1, input);
        for (int layerIdx = 0; layerIdx < lastLayerIdx; layerIdx++) {
            activations.push_back(modelLayers[layerIdx]->forward(activations.back()));
        }
        for (int layerIdx = lastLayerIdx - 1; layerIdx >= 0; layerIdx--) {
            propagatedGrad = modelLayers[layerIdx]->backward(activations[layerIdx], propagatedGrad);
        }
        return propagatedGrad;
    }

    /**
     * @brief Resets all parameter gradients in every sub-layer to zero.
     */
    void zeroGrad() override {
        for (auto& layer : modelLayers) {
            layer->zeroGrad();
        }
    }

    /**
     * @brief Propagates SGD update to every sub-layer.
     * Each layer updates its parameters in-place using its own gradients.
     * @param lr Learning rate.
     */
    void sgdUpdate(DType lr) override {
        for (auto &layer : modelLayers) {
            layer->sgdUpdate(lr);
        }
    }

    /**
     * @brief Aggregates parameters from all sub-layers.
     * Each parameter is prefixed with "layer_N." where N is the layer index.
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
     * @brief Sets parameters for all sub-layers from a dictionary. 
     * Only keys matching the expected "layer_N.param" format are used; others are ignored.
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
     * @brief Aggregates gradients from all sub-layers.
     * Each gradient is prefixed with "layer_N." where N is the layer index.
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
 * @brief Factory function for MLP layer.
 * @param layers Vector of sub-modules.
 * @return Module<DType> Shared pointer to MLP module.
 */
template <typename DType = float> Module<DType> MLP(std::vector<Module<DType>> layers) {
    return std::make_shared<MLPLayer<DType>>(layers);
}

#endif // MLP_LAYER