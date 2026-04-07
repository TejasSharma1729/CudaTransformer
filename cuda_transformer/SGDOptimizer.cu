#include "headers.cu"

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
    DType learningRate;

public:
    /**
     * @brief Constructs an SGD optimizer.
     * @param lr Learning rate.
     */
    SGDOptimizer(DType lr) : learningRate(lr) {}

    /**
     * @brief Performs one SGD update step.
     * Updates all parameters: param -= lr * grad
     *
     * @param params Map of parameter names to the parameters Tensor.
     * @param grads Map of gradient names to the gradient Tensor.
     *              Keys should match parameter names.
     */
    void step(
        const std::map<std::string, Tensor<DType>>& params,
        const std::map<std::string, Tensor<DType>>& grads
    ) {
        for (auto& [name, param] : params) {
            auto it = grads.find(name);
            if (it == grads.end()) continue;
            
            // Get raw pointers from Tensors for the kernel
            runSgdUpdate<DType>(
                param.dataPtr(),
                it->second.dataPtr(), 
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
