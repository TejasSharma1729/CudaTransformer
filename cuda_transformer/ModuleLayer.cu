#include "Tensor.cu"

/**
 * @brief Default block dim for CUDA kernels.
 */
#define BLOCKDIM 32
#ifndef MODULE_LAYER
#define MODULE_LAYER

/**
 * @brief Base class for neural network layers.
 * 
 * @tparam DType The data type used for computations.
 */
template <typename DType = float> struct Layer {
    Layer() {}
    virtual ~Layer() = default;

    /**
     * @brief Creates a deep copy of the layer and its parameters.
     * @return std::shared_ptr<Layer<DType>> A deep-cloned layer module.
     */
    virtual std::shared_ptr<Layer<DType>> clone() = 0;

    /**
     * @brief Executes the forward pass of the layer.
     * @param input The input tensor.
     * @return Tensor<DType> The output tensor.
     */
    virtual Tensor<DType> forward(Tensor<DType> input) = 0;

    /**
     * @brief Executes the backward pass of the layer to compute grad.
     * @param input The original input tensor used in the forward pass.
     * @param gradOutput The grad of the loss with respect to the output.
     * @return Tensor<DType> The grad of the loss with respect to the input.
     */
    virtual Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) = 0;

    /**
     * @brief Function call operator shorthand for the forward pass.
     * @param input The input tensor.
     * @return Tensor<DType> The output tensor.
     */
    Tensor<DType> operator()(Tensor <DType>input) { return this->forward(input); }

    /**
     * @brief Obtains all the parameters of the layer or module.
     * The default implementation returns an empty map.
     * The keys refer to the name of the parameter (weights, biases, etc.)
     * and the values in the map are Tensors.
     * @return map<string, Tensor> A map of parameter names to their tensors.
     */
    virtual std::map<std::string, Tensor<DType>> getParameters() { return {}; }

    /**
     * @brief Obtains all the gradients for all parameters of the layer or module.
     * The default implementation returns an empty map.
     * The keys refer to the name of the parameter (weights, biases, etc.)
     * and the values in the map are Tensors.
     * @return map<string, Tensor> A map of parameter names to their tensors.
     */
    virtual std::map<std::string, Tensor<DType>> getGradients() { return {}; }

    /**
     * @brief Performs a single in-place SGD update: param -= lr * grad.
     * Layers without learnable parameters (ReLU, GELU, Checkpoint) use this no-op default.
     * Layers with parameters override this to update their weights and biases.
     * Container layers (MLP, TransformerBlock, Transformer) override this to propagate to sub-layers.
     * @param lr Learning rate.
     */
    virtual void sgdUpdate(DType lr) {}
};

/**
 * @brief Shorthand for a shared pointer to a Layer.
 */
template <typename DType = float> using Module = std::shared_ptr<Layer<DType>>;

#endif // MODULE_LAYER