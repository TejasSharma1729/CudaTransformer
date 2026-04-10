#include "adam_kernels.cu"
#include "Tensor.cu"

#ifndef ADAM_OPTIMIZER
#define ADAM_OPTIMIZER


/**
 * @brief Standalone Adam optimizer that operates on explicit parameter/gradient pointer maps.
 * Useful when layers expose `getParameters()` / `getGradients()` returning raw pointer maps
 * (e.g. LinearLayer).  For the common case of updating a whole layer tree, prefer calling
 * `layer->sgdUpdate(lr)` directly, which is implemented on every layer type.
 *
 * The underlying CUDA kernel is `adamUpdateKernel` declared in headers.cu and launched
 * via `runAdamUpdate`.  This class holds no duplicate kernel — it delegates to those helpers.
 *
 * @tparam DType The data type for computations (float, double, __half, __nv_bfloat16). 
 * The optimizer maintains internal state (first and second moments) for each parameter, which are updated at each step.
 * The `step()` method performs the Adam update for all parameters based on their gradients and the internal moment estimates.
 */
template <typename DType = float> class AdamOptimizer {
private:
    double learningRate; /// Learning rate for the Adam optimizer.
    double beta1; /// Exponential decay rate for the first moment estimates.
    double beta2; /// Exponential decay rate for the second moment estimates.
    double epsilon; /// Small constant for numerical stability to prevent division by zero.
    double decay; /// Weight decay (L2 regularization) coefficient.
    int timeStep; /// Time step counter for bias correction in Adam.

    std::map<std::string, Tensor<DType>> parameters; /// Map of parameter names to their corresponding Tensors.
    std::map<std::string, Tensor<DType>> firstMoments; /// Map of parameter names to their first moment estimates (m).
    std::map<std::string, Tensor<DType>> secondMoments; /// Map of parameter names to their second moment estimates (v).

public:
    /**
     * @brief Constructs an Adam optimizer.
     * @param params Map of parameter names to the parameters Tensor.
     * @param grads Map of gradient names to the gradients Tensor. Keys should match parameter names.
     * @param lr Learning rate.
     * @param b1 Exponential decay rate for the first moment estimates.
     * @param b2 Exponential decay rate for the second moment estimates.
     * @param eps Small constant for numerical stability.
     * @param decay Weight decay (L2 regularization) coefficient.
     * @note The parameters and gradients are expected to be shallow copies of (i.e., shared with) the model's
     * parameters and gradients, so that updates to Tensors here are reflected in the model directly.
     */
    AdamOptimizer(
        std::map<std::string, Tensor<DType>> params,
        double lr = 0.001,
        double b1 = 0.9,
        double b2 = 0.999,
        double eps = 1e-8,
        double decay = 0.0
    ) : parameters(params),
        learningRate(lr), beta1(b1), beta2(b2), epsilon(eps), decay(decay), timeStep(0) 
    {
        // Initialize first and second moment tensors with the same shape as parameters, filled with zeros
        for (const auto& [name, param] : parameters) {
            firstMoments[name] = Tensor<DType>(param.shape(), (DType)0.0);
            secondMoments[name] = Tensor<DType>(param.shape(), (DType)0.0);
        }
    }
    
    /**
     * @brief Performs one Adam update step.
     * 
     * Updates all parameters using the Adam update rule:
     * m = beta1 * m + (1 - beta1) * grad
     * v = beta2 * v + (1 - beta2) * (grad * grad)
     * m_hat = m / (1 - beta1^t)
     * v_hat = v / (1 - beta2^t)
     * param = param - lr * (m_hat / (sqrt(v_hat) + epsilon) + decay * param)
     */
    void step(std::map<std::string, Tensor<DType>> gradients) {
        timeStep++;
        for (auto& [name, param] : parameters) {
            if (gradients.find(name) == gradients.end()) {
                throw std::invalid_argument("Gradient for parameter '" + name + "' not found in gradients map.");
            }
            if (param.size() != gradients[name].size()) {
                throw std::invalid_argument("Size mismatch between parameter '" + name + "' and its gradient.");
            }
            Tensor<DType> grad = gradients[name];
            Tensor<DType> m = firstMoments[name];
            Tensor<DType> v = secondMoments[name];
            int size = param.size();

            // Get raw pointers from Tensors for the kernel
            size_t numBlocks = (size + 255) / 256;
            adamUpdateKernel<DType><<<numBlocks, 256>>>(
                param.data(), grad.data(), m.data(), v.data(), size,
                learningRate, beta1, beta2, epsilon, decay, timeStep
            );
        }
        timeStep++;
    }

    /** @brief Get current learning rate. */
    double getLearningRate() const { return learningRate; }
    /** @brief Set learning rate. @param lr New learning rate. */
    void setLearningRate(double lr) { learningRate = lr; }
    /** @brief Get current beta1. */
    double getBeta1() const { return beta1; }
    /** @brief Set beta1. @param b1 New beta1 value. */
    void setBeta1(double b1) { beta1 = b1; }
    /** @brief Get current beta2. */
    double getBeta2() const { return beta2; }
    /** @brief Set beta2. @param b2 New beta2 value. */
    void setBeta2(double b2) { beta2 = b2; }
    /** @brief Get current epsilon. */
    double getEpsilon() const { return epsilon; }
    /** @brief Set epsilon. @param eps New epsilon value. */
    void setEpsilon(double eps) { epsilon = eps; }
    /** @brief Get current decay. */
    double getDecay() const { return decay; }
    /** @brief Set decay. @param d New decay value. */
    void setDecay(double d) { decay = d; }
    /** @brief Get current time step. */
    int getTimeStep() const { return timeStep; }
    /** @brief Set time step. @param t New time step value. */
    void setTimeStep(int t) { timeStep = t; }
};


template <typename DType = float> class AdamWOptimizer {
private:
    double learningRate; /// Learning rate for the AdamW optimizer.
    double beta1; /// Exponential decay rate for the first moment estimates.
    double beta2; /// Exponential decay rate for the second moment estimates.
    double epsilon; /// Small constant for numerical stability to prevent division by zero.
    double decay; /// Weight decay (L2 regularization) coefficient.
    int timeStep; /// Time step counter for bias correction in AdamW.

    std::map<std::string, Tensor<DType>> parameters; /// Map of parameter names to their corresponding Tensors.
    std::map<std::string, Tensor<DType>> firstMoments; /// Map of parameter names to their first moment estimates (m).
    std::map<std::string, Tensor<DType>> secondMoments; /// Map of parameter names to their second moment estimates (v).

public:
    /**
     * @brief Constructs an AdamW optimizer.
     * @param params Map of parameter names to the parameters Tensor.
     * @param grads Map of gradient names to the gradients Tensor. Keys should match parameter names.
     * @param lr Learning rate.
     * @param b1 Exponential decay rate for the first moment estimates.
     * @param b2 Exponential decay rate for the second moment estimates.
     * @param eps Small constant for numerical stability.
     * @param decay Weight decay (L2 regularization) coefficient.
     * @note The parameters and gradients are expected to be shallow copies of (i.e., shared with) the model's
     * parameters and gradients, so that updates to Tensors here are reflected in the model directly.
     */
    AdamWOptimizer(
        std::map<std::string, Tensor<DType>> params,
        double lr = 0.001,
        double b1 = 0.9,
        double b2 = 0.999,
        double eps = 1e-8,
        double decay = 0.0
    ) : parameters(params),
        learningRate(lr), beta1(b1), beta2(b2), epsilon(eps), decay(decay), timeStep(0)
    {
        // Initialize first and second moment tensors with the same shape as parameters, filled with zeros
        for (const auto& [name, param] : parameters) {
            firstMoments[name] = Tensor<DType>(param.shape(), (DType)0.0);
            secondMoments[name] = Tensor<DType>(param.shape(), (DType)0.0);
        }
    }
    
    /**
     * @brief Performs one AdamW update step.
     * 
     * Updates all parameters using the AdamW update rule, which decouples weight decay from the gradient-based update.
     * The update is:
     * m = beta1 * m + (1 - beta1) * grad
     * v = beta2 * v + (1 - beta2) * (grad * grad)
     * m_hat = m / (1 - beta1^t)
     * v_hat = v / (1 - beta2^t)
     * param = param - lr * (m_hat / (sqrt(v_hat) + epsilon) + decay * param)
     */
    void step(std::map<std::string, Tensor<DType>> gradients) {
        timeStep++;
        for (auto& [name, param] : parameters) {
            if (gradients.find(name) == gradients.end()) {
                throw std::invalid_argument("Gradient for parameter '" + name + "' not found in gradients map.");
            }
            if (param.size() != gradients[name].size()) {
                throw std::invalid_argument("Size mismatch between parameter '" + name + "' and its gradient.");
            }
            Tensor<DType> grad = gradients[name];
            Tensor<DType> m = firstMoments[name];
            Tensor<DType> v = secondMoments[name];
            int size = param.size();

            // Get raw pointers from Tensors for the kernel
            size_t numBlocks = (size + 255) / 256;
            adamWUpdateKernel<DType><<<numBlocks, 256>>>(
                param.data(), grad.data(), m.data(), v.data(), size,
                learningRate, beta1, beta2, epsilon, decay, timeStep
            );
        }
        timeStep++;
    }

    /** @brief Get current learning rate.*/
    double getLearningRate() const { return learningRate; }
    /** @brief Set learning rate. @param lr New learning rate.*/
    void setLearningRate(double lr) { learningRate = lr; }
    /** @brief Get current beta1. */
    double getBeta1() const { return beta1; }
    /** @brief Set beta1. @param b1 New beta1 value. */
    void setBeta1(double b1) { beta1 = b1; }
    /** @brief Get current beta2. */
    double getBeta2() const { return beta2; }
    /** @brief Set beta2. @param b2 New beta2 value. */
    void setBeta2(double b2) { beta2 = b2; }
    /** @brief Get current epsilon. */
    double getEpsilon() const { return epsilon; }
    /** @brief Set epsilon. @param eps New epsilon value. */
    void setEpsilon(double eps) { epsilon = eps; }
    /** @brief Get current decay. */
    double getDecay() const { return decay; }
    /** @brief Set decay. @param d New decay value. */
    void setDecay(double d) { decay = d; }
    /** @brief Get current time step. */
    int getTimeStep() const { return timeStep; }
    /** @brief Set time step. @param t New time step value. */
    void setTimeStep(int t) { timeStep = t; }
};


#endif