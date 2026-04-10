#include "headers.cu"

#ifndef LOSS_KERNELS
#define LOSS_KERNELS


/**
 * @brief Numerically stable softmax (or log-softmax) kernel; one thread per row.
 *
 * Each thread handles one full row of length `embeddingDim`.  The kernel first finds
 * the row maximum for numerical stability (max-subtraction trick), then computes the
 * sum of exponents, and finally writes normalized (or log-normalized) probabilities.
 *
 * When `logSoftmax = true` the output is:
 *   output[i] = x[i] - max(x) - log(sum_j exp((x[j] - max(x)) / T))
 * When `logSoftmax = false`:
 *   output[i] = exp((x[i] - max(x)) / T) / sum_j exp((x[j] - max(x)) / T)
 *
 * Launch config: <<<(totalBatchSize + 255) / 256, 256>>>
 *
 * @tparam DType         Floating-point data type (float, double, __half).
 * @param input          Device pointer to logit tensor [totalBatchSize, embeddingDim].
 * @param output         Device pointer to probability (or log-probability) tensor [totalBatchSize, embeddingDim].
 * @param embeddingDim   Number of classes / vocabulary size (last dimension).
 * @param totalBatchSize Total number of rows (batchSize * sequenceLength).
 * @param temperature    Softmax temperature; scales logits before exponentiation. Default 1.0.
 * @param logSoftmax     If true, outputs log-probabilities instead of probabilities. Default false.
 */
template <typename DType = float> __global__ void softmaxKernel(
    const DType* input,
    DType* output,
    int embeddingDim,
    int totalBatchSize,
    DType temperature = (DType)1.0,
    bool logSoftmax = false
) {
    int batchIdx = blockIdx.x * blockDim.x + threadIdx.x; // one thread per feature
    if (batchIdx >= totalBatchSize) {
        return;
    }
    DType maxVal;
    if constexpr (std::is_same_v<DType, half>) {
        maxVal = -1e4;
    } else if constexpr (std::is_same_v<DType, double>) {
        maxVal = -1e10;
    }
    for (int i = 0; i < embeddingDim; i++) {
        DType val = input[batchIdx * embeddingDim + i];
        if (val > maxVal) maxVal = val;
    }
    DType sumExp = 0;
    for (int i = 0; i < embeddingDim; i++) {
        DType expVal = (DType)exp((double)((input[batchIdx * embeddingDim + i] - maxVal) / temperature));
        sumExp += expVal;
    }

    for (int i = 0; i < embeddingDim; i++) {
        DType expVal = (DType)exp((double)((input[batchIdx * embeddingDim + i] - maxVal) / temperature));
        if (logSoftmax) {
            output[batchIdx * embeddingDim + i] = input[batchIdx * embeddingDim + i] - maxVal - (DType)log((double)sumExp);
        } else {
            output[batchIdx * embeddingDim + i] = expVal / sumExp;
        }
    }
}

/**
 * @brief Backward pass for softmax (and log-softmax); one thread per row.
 *
 * Given the original logit row `x` and the upstream gradient `dL/dy`, computes the
 * Jacobian-vector product d(softmax(x))/dx · (dL/dy) using the identity:
 *
 *   For softmax:
 *     dL/dx[i] = (dL/dy[i] - sum_j softmax(x[j]) * dL/dy[j]) * softmax(x[i]) / T
 *
 *   For log-softmax:
 *     dL/dx[i] = (dL/dy[i] - softmax(x[i]) * sum_j dL/dy[j]) / T
 *
 * Launch config: <<<(totalBatchSize + 255) / 256, 256>>>
 *
 * @tparam DType         Floating-point data type.
 * @param input          Device pointer to original logit tensor [totalBatchSize, embeddingDim].
 * @param gradOutput     Device pointer to upstream gradient tensor [totalBatchSize, embeddingDim].
 * @param gradInput      Device pointer to output input-gradient tensor [totalBatchSize, embeddingDim].
 * @param embeddingDim   Number of classes (last dimension).
 * @param totalBatchSize Total number of rows.
 * @param temperature    Temperature used in the forward pass. Must match the forward call.
 * @param logSoftmax     Must match the `logSoftmax` flag used in the forward pass.
 */
template <typename DType = float> __global__ void softmaxBackwardKernel(
    const DType* input,
    const DType* gradOutput,
    DType* gradInput,
    int embeddingDim,
    int totalBatchSize,
    DType temperature = (DType)1.0,
    bool logSoftmax = false
) {
    int batchIdx = blockIdx.x * blockDim.x + threadIdx.x; // one thread per feature
    if (batchIdx >= totalBatchSize) {
        return;
    }
    DType maxVal;
    if constexpr (std::is_same_v<DType, half>) {
        maxVal = -1e4;
    } else if constexpr (std::is_same_v<DType, double>) {
        maxVal = -1e10;
    }
    for (int i = 0; i < embeddingDim; i++) {
        DType val = input[batchIdx * embeddingDim + i];
        if (val > maxVal) maxVal = val;
    }
    DType sumExp = 0;
    for (int i = 0; i < embeddingDim; i++) {
        DType expVal = (DType)exp((double)((input[batchIdx * embeddingDim + i] - maxVal) / temperature));
        sumExp += expVal;
    }

    DType sumGrad = 0;
    for (int i = 0; i < embeddingDim; i++) {
        if (logSoftmax) {
            sumGrad += gradOutput[batchIdx * embeddingDim + i];
        } else {
            DType expVal = (DType)exp((double)((input[batchIdx * embeddingDim + i] - maxVal) / temperature));
            DType softmaxVal = expVal / sumExp;
            sumGrad += softmaxVal * gradOutput[batchIdx * embeddingDim + i];
        }
    }

    for (int i = 0; i < embeddingDim; i++) {
        DType expVal = (DType)exp((double)((input[batchIdx * embeddingDim + i] - maxVal) / temperature));
        DType softmaxVal = expVal / sumExp;
        DType gradOut;
        if (logSoftmax) {
            gradOut = gradOutput[batchIdx * embeddingDim + i] - softmaxVal * sumGrad;
        } else {
            gradOut = softmaxVal * (gradOutput[batchIdx * embeddingDim + i] - sumGrad);
        }
        gradInput[batchIdx * embeddingDim + i] = gradOut / temperature;
    }
}


/**
 * @brief Cross-entropy loss forward kernel: computes -log(p[target]) per row.
 *
 * Expects `input` to contain probabilities (i.e. softmax has already been applied).
 * Out-of-range target IDs produce zero loss for that position.
 *
 * Launch config: <<<(totalBatchSize + 255) / 256, 256>>>
 *
 * @tparam DType   Floating-point data type.
 * @tparam IdType  Integer type used for target class indices (e.g. int).
 * @param input          Device pointer to probability tensor [totalBatchSize, embeddingDim].
 * @param target         Device pointer to target class index array [totalBatchSize].
 * @param lossOutput     Device pointer to per-row loss output [totalBatchSize].
 * @param embeddingDim   Number of classes (vocabulary size).
 * @param totalBatchSize Total number of rows.
 */
template <typename DType = float, typename IdType = int> __global__ void crossEntropyLossKernel(
    const DType* input,
    const IdType* target,
    DType* lossOutput,
    int embeddingDim,
    int totalBatchSize
) {
    int batchIdx = blockIdx.x * blockDim.x + threadIdx.x; // one thread per feature
    if (batchIdx >= totalBatchSize) {
        return;
    }
    int targetId = (int)target[batchIdx];
    if (targetId < 0 || targetId >= embeddingDim) {
        lossOutput[batchIdx] = (DType)0;
        return;
    }
    int fullIdx = batchIdx * embeddingDim + targetId;
    double prob = (double)input[fullIdx];
    lossOutput[batchIdx] = (DType)(-log((double)prob));
}

/**
 * @brief Cross-entropy loss backward kernel: gradient w.r.t the probability vector.
 *
 * Given that the loss is L = -log(p[target]), the gradient is:
 *   dL/dp[i] = 0           for i != target
 *   dL/dp[target] = -1/p[target]
 *
 * Expects `input` to contain the softmax probabilities from the forward pass.
 * Out-of-range target IDs leave the gradient unchanged (zeros from allocation).
 *
 * Launch config: <<<(totalBatchSize + 255) / 256, 256>>>
 *
 * @tparam DType   Floating-point data type.
 * @tparam IdType  Integer type used for target class indices.
 * @param input          Device pointer to probability tensor [totalBatchSize, embeddingDim].
 * @param target         Device pointer to target class index array [totalBatchSize].
 * @param gradInput      Device pointer to output gradient tensor [totalBatchSize, embeddingDim].
 * @param embeddingDim   Number of classes (vocabulary size).
 * @param totalBatchSize Total number of rows.
 */
template <typename DType = float, typename IdType = int> __global__ void crossEntropyLossBackwardKernel(
    const DType* input,
    const IdType* target,
    DType* gradInput,
    int embeddingDim,
    int totalBatchSize
) {
    int batchIdx = blockIdx.x * blockDim.x + threadIdx.x; // one thread per feature
    if (batchIdx >= totalBatchSize) {
        return;
    }
    int targetId = (int)target[batchIdx];
    if (targetId < 0 || targetId >= embeddingDim) {
        return;
    }
    int fullIdx = batchIdx * embeddingDim + targetId;
    double prob = (double)input[fullIdx];
    for (int i = 0; i < targetId; i++) {
        gradInput[batchIdx * embeddingDim + i] = (DType)0;
    }
    gradInput[fullIdx] = (DType)(-1.0 / prob);
    for (int i = targetId + 1; i < embeddingDim; i++) {
        gradInput[batchIdx * embeddingDim + i] = (DType)0;
    }
}


/**
 * @brief Mean Squared Error (MSE) loss forward kernel: computes (1/D) * sum((input - target)^2).
 *
 * Each thread handles one row of length `embeddingDim`, accumulating the squared differences
 * and writing a single scalar loss per row.
 *
 * Launch config: <<<(totalBatchSize + 255) / 256, 256>>>
 *
 * @tparam DType         Floating-point data type.
 * @param input          Device pointer to predicted output tensor [totalBatchSize, embeddingDim].
 * @param target         Device pointer to ground-truth target tensor [totalBatchSize, embeddingDim].
 * @param lossOutput     Device pointer to per-row MSE loss output [totalBatchSize].
 * @param embeddingDim   Feature dimension (denominator of the mean).
 * @param totalBatchSize Total number of rows.
 */
template <typename DType = float> __global__ void mseLossKernel(
    const DType* input,
    const DType* target,
    DType* lossOutput,
    int embeddingDim,
    int totalBatchSize
) {
    int batchIdx = blockIdx.x * blockDim.x + threadIdx.x; // one thread per feature
    if (batchIdx >= totalBatchSize) {
        return;
    }
    DType sumSq = 0;
    for (int i = 0; i < embeddingDim; i++) {
        DType diff = input[batchIdx * embeddingDim + i] - target[batchIdx * embeddingDim + i];
        sumSq += diff * diff;
    }
    lossOutput[batchIdx] = sumSq / (DType)embeddingDim;
}

/**
 * @brief Mean Squared Error (MSE) loss backward kernel: gradient w.r.t. the predicted input.
 *
 * The derivative of MSE = (1/D) * sum((input - target)^2) w.r.t. input[i] is:
 *   dL/dinput[i] = 2 * (input[i] - target[i]) * gradOutput[batch] / D
 *
 * Launch config: <<<(totalBatchSize + 255) / 256, 256>>>
 *
 * @tparam DType         Floating-point data type.
 * @param input          Device pointer to predicted output tensor [totalBatchSize, embeddingDim].
 * @param target         Device pointer to ground-truth target tensor [totalBatchSize, embeddingDim].
 * @param gradInput      Device pointer to output gradient tensor [totalBatchSize, embeddingDim].
 * @param embeddingDim   Feature dimension (used as the MSE divisor).
 * @param totalBatchSize Total number of rows.
 */
template <typename DType = float> __global__ void mseLossBackwardKernel(
    const DType* input,
    const DType* target,
    DType* gradInput,
    int embeddingDim,
    int totalBatchSize
) {
    int batchIdx = blockIdx.x * blockDim.x + threadIdx.x; // one thread per feature
    if (batchIdx >= totalBatchSize) {
        return;
    }
    for (int i = 0; i < embeddingDim; i++) {
        gradInput[batchIdx * embeddingDim + i] = (DType)(2.0 * (input[batchIdx * embeddingDim + i] - target[batchIdx * embeddingDim + i]) * gradOutput[batchIdx] / embeddingDim);
    }
}


#endif // LOSS_KERNELS