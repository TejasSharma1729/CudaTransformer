#include "ModuleLayer.cu"

#ifndef CHECKPOINT_LAYER
#define CHECKPOINT_LAYER

/**
 * @brief Layer used for grad checkpointing by storing input activations.
 */
template <typename DType = float> struct CheckpointLayer : public Layer<DType> {
    Tensor<DType> activationStorage; ///< Stored input activations for re-computation.

    CheckpointLayer() : Layer<DType>() { }

    /** @brief Clones the checkpoint layer and its stored data. */
    std::shared_ptr<Layer<DType>> clone() override { 
        auto clonedLayer = std::make_shared<CheckpointLayer<DType>>();
        if (this->activationStorage.size() > 0) {
            clonedLayer->activationStorage = activationStorage.clone();
        }
        return clonedLayer;
    }

    /** @brief Stores input and returns it. */
    Tensor<DType> forward(Tensor<DType> input) override {
        activationStorage = input;
        return input;
    }

    /** @brief Identity backward pass. */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) override {
        return gradOutput;
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