#include "Tensor.cu"

#ifndef SGD_OPTIMIZER
#define SGD_OPTIMIZER

/**
 * @brief Standalone SGD optimizer that operates on explicit parameter/gradient pointer maps.
 * Useful when layers expose `getParameters()` / `getGradients()` returning raw pointer maps
 * (e.g. LinearLayer).  For the common case of updating a whole layer tree, prefer calling
 * `layer->sgdUpdate(lr)` directly, which is implemented on every layer type.
 *
 * The underlying CUDA kernel is `sgdUpdateKernel` declared in headers.cu and launched
 * via `runSgdUpdate`.  This class holds no duplicate kernel — it delegates to those helpers.
 *
 * @tparam DType The data type for computations (float, double, __half, __nv_bfloat16).
 */
template <typename DType = float> class SGDOptimizer {
private:
    DType learningRate; /// The learning rate for SGD updates.
    std::map<std::string, Tensor<DType>> parameters; /// Map of parameter names to their corresponding Tensors.

public:
    /**
     * @brief Constructs an SGD optimizer.
     * @param lr Learning rate.
     */
    SGDOptimizer(std::map<std::string, Tensor<DType>> params, DType lr) 
        : parameters(params), learningRate(lr)
    { }

    /**
     * @brief Performs one SGD update step.
     * Updates all parameters: param -= lr * grad
     * @param gradients Map of parameter names to their corresponding gradient Tensors. 
     * Must match keys and sizes in `parameters`.
     */
    void step(std::map<std::string, Tensor<DType>> gradients) {
        for (const auto& [name, param] : parameters) {
            if (gradients.find(name) == gradients.end()) {
                throw std::invalid_argument("Gradient for parameter '" + name + "' not found in gradients map.");
            }
            if (param.size() != gradients.at(name).size()) {
                throw std::invalid_argument("Size mismatch between parameter '" + name + "' and its gradient.");
            }
        }
        for (auto& [name, param] : parameters) {
            Tensor<DType> grad = gradients[name]; // Get corresponding gradient tensor by name
            
            // Get raw pointers from Tensors for the kernel
            runSgdUpdate<DType>(
                param.dataPtr(),
                grad.dataPtr(),
                learningRate,
                param.size()
            );
        }
    }

    /**
     * @brief Set learning rate.
     * @param lr New learning rate.
     */
    void setLearningRate(DType lr) {
        learningRate = lr;
    }

    /**
     * @brief Get current learning rate.
     * @return Current learning rate.
     */
    DType getLearningRate() const {
        return learningRate;
    }
};

#endif // SGD_OPTIMIZER
