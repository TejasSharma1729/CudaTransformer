#include "loss_kernels.cu"
#include "Tensor.cu"

#ifndef SOFTMAX_LOSS
#define SOFTMAX_LOSS


/**
 * @brief Numerically stable softmax layer with optional temperature scaling.
 *
 * Converts a raw logit tensor into a probability distribution over the last dimension.
 * The temperature parameter T scales the logits before exponentiation:
 *   p[i] = exp(x[i] / T) / sum_j exp(x[j] / T)
 *
 * Higher temperatures (T > 1) produce softer distributions; lower temperatures (T < 1)
 * sharpen the distribution towards the argmax (greedy).
 *
 * Has no learnable parameters.  Designed to be paired with CrossEntropyLoss.
 *
 * @tparam DType Floating-point data type (float, double, __half, __nv_bfloat16).
 */
template <typename DType = float> struct Softmax {
    double temperature = 1.0; /// Temperature for scaling logits before exponentiation.

    /**
     * @brief Constructs a Softmax layer.
     * @param temperature Temperature parameter to scale the logits. Defaults to 1.0.
     */
    Softmax(double temperature = 1.0) : temperature(temperature) {}

    /**
     * @brief Computes the forward pass of the Softmax layer.
     * @param input Unscaled logit tensor.
     * @return Tensor<DType> The output of the Softmax function.
     */
    Tensor<DType> forward(Tensor<DType> input) {
        Tensor<DType> output(input.shape());
        int embeddingDim = input.shape()[input.nDim() - 1];
        int totalBatchSize = input.size() / embeddingDim;
        int threadsPerBlock = 256;
        int numBlocks = (totalBatchSize + threadsPerBlock - 1) / threadsPerBlock;
        softmaxKernel<<<numBlocks, threadsPerBlock>>>(
            input.get(), output.get(), embeddingDim, totalBatchSize, temperature, false
        );
        return output;
    }

    /**
     * @brief Calls the forward function.
     * @param input Unscaled logit tensor.
     * @return Tensor<DType> The output of the Softmax function.
     */
    Tensor<DType> operator()(Tensor<DType> input) {
        return forward(input);
    }

    /**
     * @brief Computes the backward pass of the Softmax layer.
     * @param input Input tensor originally passed to the forward layer.
     * @param gradOutput The upstream gradient from the network.
     * @return Tensor<DType> The input gradient.
     */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) {
        Tensor<DType> gradInput(input.shape());
        int embeddingDim = input.shape()[input.nDim() - 1];
        int totalBatchSize = input.size() / embeddingDim;
        int threadsPerBlock = 256;
        int numBlocks = (totalBatchSize + threadsPerBlock - 1) / threadsPerBlock;
        softmaxBackwardKernel<<<numBlocks, threadsPerBlock>>>(
            input.get(), gradOutput.get(), gradInput.get(), embeddingDim, totalBatchSize, temperature, false
        );
        return gradInput;
    }
};


/**
 * @brief Numerically stable log-softmax layer with optional temperature scaling.
 *
 * Computes the log of the softmax probabilities in a single pass, which is more
 * numerically stable than computing softmax first and then taking the log:
 *   log_p[i] = x[i] / T - max(x/T) - log(sum_j exp((x[j] - max(x)) / T))
 *
 * The output of LogSoftmax is the natural input to NLLLoss, and is mathematically
 * equivalent to but more stable than log(Softmax(x)).
 *
 * Has no learnable parameters.
 *
 * @tparam DType Floating-point data type.
 */
template <typename DType = float> struct LogSoftmax {
    double temperature = 1.0; /// Temperature for scaling logits.

    /**
     * @brief Constructs a LogSoftmax layer.
     * @param temperature Temperature parameter to scale the logits. Defaults to 1.0.
     */
    LogSoftmax(double temperature = 1.0) : temperature(temperature) {}

    /**
     * @brief Computes the forward pass of the LogSoftmax layer.
     * @param input Unscaled logit tensor.
     * @return Tensor<DType> The output of the LogSoftmax function.
     */
    Tensor<DType> forward(Tensor<DType> input) {
        Tensor<DType> output(input.shape());
        int embeddingDim = input.shape()[input.nDim() - 1];
        int totalBatchSize = input.size() / embeddingDim;
        int threadsPerBlock = 256;
        int numBlocks = (totalBatchSize + threadsPerBlock - 1) / threadsPerBlock;
        softmaxKernel<DType><<<numBlocks, threadsPerBlock>>>(
            input.get(), output.get(), embeddingDim, totalBatchSize, temperature, true
        );
        return output;
    }

    /**
     * @brief Calls the forward function.
     * @param input Unscaled logit tensor.
     * @return Tensor<DType> The output of the LogSoftmax function.
     */
    Tensor<DType> operator()(Tensor<DType> input) {
        return forward(input);
    }

    /**
     * @brief Computes the backward pass of the LogSoftmax layer.
     * @param input Input tensor originally passed to the forward layer.
     * @param gradOutput The upstream gradient from the network.
     * @return Tensor<DType> The input gradient.
     */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) {
        Tensor<DType> gradInput(input.shape());
        int embeddingDim = input.shape()[input.nDim() - 1];
        int totalBatchSize = input.size() / embeddingDim;
        int threadsPerBlock = 256;
        int numBlocks = (totalBatchSize + threadsPerBlock - 1) / threadsPerBlock;
        softmaxBackwardKernel<DType><<<numBlocks, threadsPerBlock>>>(
            input.get(), gradOutput.get(), gradInput.get(), embeddingDim, totalBatchSize, temperature, true
        );
        return gradInput;
    }
};


/**
 * @brief Cross-entropy loss between a probability distribution and integer class labels.
 *
 * Expects `input` to contain probabilities (output of Softmax, not raw logits).
 * The forward pass computes per-row loss:  L[i] = -log(p[i][target[i]])
 * The backward pass returns the gradient w.r.t. the probability vector.
 *
 * Typical usage:
 *   auto probs  = softmax(logits);
 *   auto loss   = cross_entropy.forward(probs, target);
 *   auto grad   = cross_entropy.backward(probs, target);
 *
 * Has no learnable parameters.
 *
 * @tparam DType   Floating-point data type of the probabilities.
 * @tparam IdType  Integer type of the target class indices (typically int).
 */
template <typename DType = float, typename IdType = int> struct CrossEntropyLoss {

    /**
     * @brief Computes the forward pass of the CrossEntropyLoss.
     * @param input The input tensor containing the probabilities (after LogSoftmax computation).
     * @param target The target labels tensor.
     * @return Tensor<DType> The resulting loss.
     */
    Tensor<DType> forward(Tensor<DType> input, Tensor<IdType> target) {
        Tensor<DType> lossOutput({input.shape()[0]});
        int embeddingDim = input.shape()[input.nDim() - 1];
        int totalBatchSize = input.size() / embeddingDim;
        int threadsPerBlock = 256;
        int numBlocks = (totalBatchSize + threadsPerBlock - 1) / threadsPerBlock;
        crossEntropyLossKernel<DType><<<numBlocks, threadsPerBlock>>>(
            input.get(), target.get(), lossOutput.get(), embeddingDim, totalBatchSize
        );
        return lossOutput;
    }

    /**
     * @brief Calls the forward function.
     * @param input The input tensor containing the probabilities.
     * @param target The target labels tensor.
     * @return Tensor<DType> The resulting loss.
     */
    Tensor<DType> operator()(Tensor<DType> input, Tensor<IdType> target) {
        return forward(input, target);
    }

    /**
     * @brief Computes the backward pass of the CrossEntropyLoss.
     * @param input The input probabilities tensor from the forward pass.
     * @param target The target labels tensor.
     * @return Tensor<DType> The gradients w.r.t the input tensor.
     */
    Tensor<DType> backward(Tensor<DType> input, Tensor<IdType> target) {
        Tensor<DType> gradInput(input.shape());
        int embeddingDim = input.shape()[input.nDim() - 1];
        int totalBatchSize = input.size() / embeddingDim;
        int threadsPerBlock = 256;
        int numBlocks = (totalBatchSize + threadsPerBlock - 1) / threadsPerBlock;
        crossEntropyLossBackwardKernel<DType><<<numBlocks, threadsPerBlock>>>(
            input.get(), target.get(), gradInput.get(), embeddingDim, totalBatchSize
        );
        return gradInput;
    }
};


/**
 * @brief Mean Squared Error (MSE) loss between predicted and target tensors.
 *
 * Computes per-row loss:  L[i] = (1/D) * sum_j (input[i][j] - target[i][j])^2
 *
 * where D is the feature dimension (last axis).  The backward pass returns
 * the gradient w.r.t. `input`:  dL/dinput[i][j] = 2 * (input[i][j] - target[i][j]) / D.
 *
 * Has no learnable parameters.  Suitable for regression tasks.
 *
 * @tparam DType Floating-point data type.
 */
template <typename DType = float> struct MSELoss {

    /**
     * @brief Computes the forward pass for Mean Squared Error loss.
     * @param input The predicted output tensor.
     * @param target The desired target tensor.
     * @return Tensor<DType> Computed MSE loss tensor.
     */
    Tensor<DType> forward(Tensor<DType> input, Tensor<DType> target) {
        Tensor<DType> lossOutput({input.shape()[0]});
        int embeddingDim = input.shape()[input.nDim() - 1];
        int totalBatchSize = input.size() / embeddingDim;
        int threadsPerBlock = 256;
        int numBlocks = (totalBatchSize + threadsPerBlock - 1) / threadsPerBlock;
        mseLossKernel<<<numBlocks, threadsPerBlock>>>(
            input.get(), target.get(), lossOutput.get(), embeddingDim, totalBatchSize
        );
        return lossOutput;
    }

    /**
     * @brief Calls the forward function.
     * @param input The predicted output tensor.
     * @param target The desired target tensor.
     * @return Tensor<DType> Computed MSE loss.
     */
    Tensor<DType> operator()(Tensor<DType> input, Tensor<DType> target) {
        return forward(input, target);
    }

    /**
     * @brief Computes the backward pass for Mean Squared Error loss.
     * @param input The input tensor originally passed to the forward layer.
     * @param target The desired target tensor.
     * @return Tensor<DType> The resulting input gradient.
     */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> target) {
        Tensor<DType> gradInput(input.shape());
        int embeddingDim = input.shape()[input.nDim() - 1];
        int totalBatchSize = input.size() / embeddingDim;
        int threadsPerBlock = 256;
        int numBlocks = (totalBatchSize + threadsPerBlock - 1) / threadsPerBlock;
        mseLossBackwardKernel<<<numBlocks, threadsPerBlock>>>(
            input.get(), target.get(), gradInput.get(), embeddingDim, totalBatchSize
        );
        return gradInput;
    }
};


#endif // SOFTMAX_LOSS