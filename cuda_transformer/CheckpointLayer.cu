#include "ModuleLayer.cu"

#ifndef CHECKPOINT_LAYER
#define CHECKPOINT_LAYER

/**
 * @brief Layer used for grad checkpointing by storing input activations.
 * This just stores activation and it does nothing otherwise.
 */
template <typename DType = float> struct CheckpointLayer : public Layer<DType> {
    Tensor<DType> activationStorage; ///< Stored input activations for re-computation.

    /** @brief Constructs a Checkpoint layer. */
    CheckpointLayer() : Layer<DType>() { }

    /** @brief Clones the checkpoint layer and its stored data. */
    std::shared_ptr<Layer<DType>> clone() override { 
        auto clonedLayer = std::make_shared<CheckpointLayer<DType>>();
        if (this->activationStorage.size() > 0) {
            clonedLayer->activationStorage = activationStorage.clone();
        }
        return clonedLayer;
    }

    /** 
     * @brief Stores input and returns it. Does not do anything.
     * @param input The input tensor to be stored and returned.
     */
    Tensor<DType> forward(Tensor<DType> input) override {
        activationStorage = input;
        return input;
    }

    /**
     * @brief Identity backward pass, does nothing and returns the gradients.
     * @param input The input tensor (unused, but present for interface consistency).
     * @param gradOutput The gradient tensor (passed through unchanged).
     * @return The gradient tensor.
     */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) override {
        return gradOutput;
    }

    /**
     * @brief Clears the stored activations.
     */
    void clear() {
        activationStorage = Tensor<DType>(); // Reset to an empty tensor
    }
};

/**
 * @brief Factory function for Checkpoint layer.
 * @return Module<DType> Shared pointer to Checkpoint module.
 */
template <typename DType = float> Module<DType> Checkpoint() {
    return std::make_shared<CheckpointLayer<DType>>();
}

#endif // CHECKPOINT_LAYER