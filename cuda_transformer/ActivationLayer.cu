#include "ModuleLayer.cu"
#include "sigmoid_kernels.cu"
#include "tanh_kernels.cu"
#include "relu_kernels.cu"
#include "gelu_kernels.cu"

#ifndef ACTIVATION_LAYER
#define ACTIVATION_LAYER

/**
 * @brief Scoped enumeration of supported activation functions.
 * Using enum class avoids injecting ReLU/GELU/Sigmoid/Tanh into the global
 * namespace, which would clash with the factory function templates of the same name.
 */
enum class ActivationType { ReLU, GELU, Sigmoid, Tanh };


/**
 * @brief Single activation layer that dispatches to ReLU, GELU, Sigmoid, or Tanh
 * based on the ActivationType enum.  Has no learnable parameters; sgdUpdate is a no-op
 * (inherited from Layer).
 *
 * @tparam DType The data type used for computations.
 */
template <typename DType = float> struct ActivationLayer : public Layer<DType> {
    int inputDim;
    ActivationType activationType;

    /**
     * @brief Constructs an ActivationLayer.
     * @param inputDim Feature dimension of the input.
     * @param act      Which activation function to apply.
     */
    ActivationLayer(int inputDim, ActivationType act)
        : Layer<DType>(), inputDim(inputDim), activationType(act) {}

    /** @brief Clones the layer (trivial — no parameters). */
    std::shared_ptr<Layer<DType>> clone() override {
        return std::make_shared<ActivationLayer<DType>>(inputDim, activationType);
    }

    /** @brief Forward pass; dispatches to the appropriate kernel. */
    Tensor<DType> forward(Tensor<DType> input) override {
        assert(input.shape()[input.nDim() - 1] == (size_t)inputDim);
        Tensor<DType> output(input.shape().toVector());
        int totalBatch = output.size() / inputDim;
        dim3 grid((inputDim + 255) / 256, totalBatch);

        switch (activationType) {
            case ActivationType::ReLU:
                reluForward<DType><<<grid, 256>>>(input.get(), output.get(), inputDim, totalBatch);
                break;
            case ActivationType::GELU:
                geluForward<DType><<<grid, 256>>>(input.get(), output.get(), inputDim, totalBatch);
                break;
            case ActivationType::Sigmoid:
                sigmoidForward<DType><<<grid, 256>>>(input.get(), output.get(), inputDim, totalBatch);
                break;
            case ActivationType::Tanh:
                tanhForward<DType><<<grid, 256>>>(input.get(), output.get(), inputDim, totalBatch);
                break;
        }
        return output;
    }

    /** @brief Backward pass; dispatches to the appropriate gradient kernel. */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) override {
        Tensor<DType> gradInput(gradOutput.shape().toVector());
        int totalBatch = gradOutput.size() / inputDim;
        dim3 grid((inputDim + 255) / 256, totalBatch);

        switch (activationType) {
            case ActivationType::ReLU:
                reluBackward<DType><<<grid, 256>>>(input.get(), gradInput.get(), gradOutput.get(), inputDim, totalBatch);
                break;
            case ActivationType::GELU:
                geluBackward<DType><<<grid, 256>>>(input.get(), gradInput.get(), gradOutput.get(), inputDim, totalBatch);
                break;
            case ActivationType::Sigmoid:
                sigmoidBackward<DType><<<grid, 256>>>(input.get(), gradInput.get(), gradOutput.get(), inputDim, totalBatch);
                break;
            case ActivationType::Tanh:
                tanhBackward<DType><<<grid, 256>>>(input.get(), gradInput.get(), gradOutput.get(), inputDim, totalBatch);
                break;
        }
        return gradInput;
    }

    /** @brief Activation functions have no learnable parameters. */
    std::map<std::string, Tensor<DType>> getParameters() override {
        return {};
    }

    /** @brief Activation functions have no learnable parameters. */
    std::map<std::string, Tensor<DType>> getGradients() override {
        return {};
    }
};

/**
 * @brief Factory function for ActivationLayer.
 * @param inputDim Feature dimension.
 * @param act      Activation type (ReLU, GELU, Sigmoid, Tanh).
 * @return Module<DType> Shared pointer to the activation module.
 */
template <typename DType = float>
/**
 * @brief Execute Activation operation.
 */
Module<DType> Activation(int inputDim, ActivationType act) {
    return std::make_shared<ActivationLayer<DType>>(inputDim, act);
}

template <typename DType = float>
/**
 * @brief Execute SigmoidActivation operation.
 */
Module<DType> SigmoidActivation(int inputDim) {
    return Activation<DType>(inputDim, ActivationType::Sigmoid);
}

template <typename DType = float>
/**
 * @brief Execute TanhActivation operation.
 */
Module<DType> TanhActivation(int inputDim) {
    return Activation<DType>(inputDim, ActivationType::Tanh);
}

template <typename DType = float>
/**
 * @brief Execute ReLUActivation operation.
 */
Module<DType> ReLUActivation(int inputDim) {
    return Activation<DType>(inputDim, ActivationType::ReLU);
}

template <typename DType = float>
/**
 * @brief Execute GELUActivation operation.
 */
Module<DType> GELUActivation(int inputDim) {
    return Activation<DType>(inputDim, ActivationType::GELU);
}


#endif // ACTIVATION_LAYER
