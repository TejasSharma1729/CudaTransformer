#include "embedding_kernels.cu"
#include "linear_kernels.cu"
#include "Tensor.cu"

#ifndef EMBEDDING_LAYER
#define EMBEDDING_LAYER

template <typename DType = float, typename IdType = int>
struct TokenEmbedding {
    std::shared_ptr<DType[]> embeddingMatrix; ///< Embedding matrix of shape (vocabSize, embeddingDim).
    std::shared_ptr<DType[]> gradEmbeddingMatrix; ///< Gradient accumulator for embedding matrix.
    int vocabSize; ///< Size of the vocabulary.
    int embeddingDim; ///< Dimensionality of the embeddings.

    TokenEmbedding(int vocabSize, int embeddingDim) 
        : vocabSize(vocabSize), embeddingDim(embeddingDim) 
    {
        embeddingMatrix = cudaMakeShared<DType>(vocabSize * embeddingDim);
        gradEmbeddingMatrix = cudaMakeShared<DType>(vocabSize * embeddingDim);
    }

    /**
     * @brief Execute forward operation.
     */
    Tensor<DType> forward(Tensor<IdType> input) {
        std::vector<size_t> os = input.shape().toVector();
        os[os.size() - 1] = embeddingDim;
        Tensor<DType> output(os);
        int totalBatchSize = output.size() / embeddingDim;
        dim3 threadsPerBlock(BLOCKDIM);
        dim3 blocksPerGrid((embeddingDim + BLOCKDIM - 1) / BLOCKDIM, totalBatchSize);
        tokenEmbeddingForwardKernel<DType, IdType><<<blocksPerGrid, threadsPerBlock>>>(
            input.get(),
            output.get(),
            embeddingMatrix.get(),
            vocabSize,
            embeddingDim,
            totalBatchSize
        );
        cudaDeviceSynchronize();
        return output;
    }

    /**
     * @brief Execute operator() operation.
     */
    Tensor<DType> operator()(Tensor<IdType> input) {
        return forward(input);
    }

    /**
     * @brief Execute backward operation.
     */
    void backward(Tensor<IdType> input, Tensor<DType> gradOutput) {
        int totalBatchSize = gradOutput.size() / embeddingDim;
        dim3 threadsPerBlock(BLOCKDIM);
        dim3 blocksPerGrid((embeddingDim + BLOCKDIM - 1) / BLOCKDIM, totalBatchSize);
        tokenEmbeddingBackwardKernel<DType, IdType><<<blocksPerGrid, threadsPerBlock>>>(
            input.get(),
            gradOutput.get(),
            gradEmbeddingMatrix.get(),
            vocabSize,
            embeddingDim,
            totalBatchSize
        );
        cudaDeviceSynchronize();
    }

    /**
     * @brief Execute clone operation.
     */
    TokenEmbedding clone() {
        /**
         * @brief Execute copy operation.
         */
        TokenEmbedding copy(vocabSize, embeddingDim);
        cudaMemcpy(copy.embeddingMatrix.get(), embeddingMatrix.get(), vocabSize * embeddingDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        return copy;
    }

    /**
     * @brief Execute zeroGrad operation.
     */
    void zeroGrad() {
        cudaMemset(gradEmbeddingMatrix.get(), 0, vocabSize * embeddingDim * sizeof(DType));
    }

    /**
     * @brief Execute sgdUpdate operation.
     */
    void sgdUpdate(DType lr) {
        int totalSize = vocabSize * embeddingDim;
        int threads = 256;
        int blocks = (totalSize + threads - 1) / threads;
        sgdUpdateKernel<DType><<<blocks, threads>>>(
            embeddingMatrix.get(),
            gradEmbeddingMatrix.get(),
            lr,
            totalSize
        );
        cudaDeviceSynchronize();
    }

    /**
     * @brief Execute getParameters operation.
     */
    Tensor<DType> getParameters() {
        return Tensor<DType>(embeddingMatrix, {(size_t)vocabSize, (size_t)embeddingDim});
    }

    /**
     * @brief Execute setParameters operation.
     */
    void setParameters(Tensor<DType> t) {
        embeddingMatrix = t.dataPtr();
    }

    /**
     * @brief Execute getGradients operation.
     */
    Tensor<DType> getGradients() {
        return Tensor<DType>(gradEmbeddingMatrix, {(size_t)vocabSize, (size_t)embeddingDim});
    }
};


template <typename DType = float, typename IdType = int>
struct PositionEmbedding {
    std::shared_ptr<DType[]> embeddingMatrix; ///< Embedding matrix of shape (vocabSize, embeddingDim).
    std::shared_ptr<DType[]> gradEmbeddingMatrix; ///< Gradient accumulator for embedding matrix.
    int blockSize;
    int embeddingDim; ///< Dimensionality of the embeddings.

    PositionEmbedding(int blockSize, int embeddingDim) 
        : blockSize(blockSize), embeddingDim(embeddingDim) 
    {
        embeddingMatrix = cudaMakeShared<DType>(blockSize * embeddingDim);
        gradEmbeddingMatrix = cudaMakeShared<DType>(blockSize * embeddingDim);
    }

    /**
     * @brief Execute forward operation.
     */
    Tensor<DType> forward(Tensor<IdType> input) {
        std::vector<size_t> os = input.shape().toVector();
        os[os.size() - 1] = embeddingDim;
        Tensor<DType> output(os);
        int totalBatchSize = output.size() / embeddingDim;
        dim3 threadsPerBlock(BLOCKDIM);
        dim3 blocksPerGrid((embeddingDim + BLOCKDIM - 1) / BLOCKDIM, totalBatchSize);
        positionEmbeddingForwardKernel<DType, IdType><<<blocksPerGrid, threadsPerBlock>>>(
            input.get(),
            output.get(),
            embeddingMatrix.get(),
            embeddingDim,
            totalBatchSize
        );
        cudaDeviceSynchronize();
        return output;
    }

    /**
     * @brief Execute operator() operation.
     */
    Tensor<DType> operator()(Tensor<IdType> input) {
        return forward(input);
    }

    /**
     * @brief Execute backward operation.
     */
    void backward(Tensor<IdType> input, Tensor<DType> gradOutput) {
        int totalBatchSize = gradOutput.size() / embeddingDim;
        dim3 threadsPerBlock(BLOCKDIM);
        dim3 blocksPerGrid((embeddingDim + BLOCKDIM - 1) / BLOCKDIM, totalBatchSize);
        positionEmbeddingBackwardKernel<DType, IdType><<<blocksPerGrid, threadsPerBlock>>>(
            input.get(),
            gradOutput.get(),
            gradEmbeddingMatrix.get(),
            embeddingDim,
            totalBatchSize
        );
        cudaDeviceSynchronize();
    }

    /**
     * @brief Execute clone operation.
     */
    PositionEmbedding clone() {
        PositionEmbedding copy(blockSize, embeddingDim);
        cudaMemcpy(copy.embeddingMatrix.get(), embeddingMatrix.get(), blockSize * embeddingDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        return copy;
    }

    /**
     * @brief Execute zeroGrad operation.
     */
    void zeroGrad() {
        cudaMemset(gradEmbeddingMatrix.get(), 0, blockSize * embeddingDim * sizeof(DType));
    }

    /**
     * @brief Execute sgdUpdate operation.
     */
    void sgdUpdate(DType lr) {
        int totalSize = blockSize * embeddingDim;
        int threads = 256;
        int blocks = (totalSize + threads - 1) / threads;
        sgdUpdateKernel<DType><<<blocks, threads>>>(
            embeddingMatrix.get(),
            gradEmbeddingMatrix.get(),
            lr,
            totalSize
        );
        cudaDeviceSynchronize();
    }

    /**
     * @brief Execute getParameters operation.
     */
    Tensor<DType> getParameters() {
        return Tensor<DType>(embeddingMatrix, {(size_t)blockSize, (size_t)embeddingDim});
    }

    /**
     * @brief Execute setParameters operation.
     */
    void setParameters(Tensor<DType> t) {
        embeddingMatrix = t.dataPtr();
    }

    /**
     * @brief Execute getGradients operation.
     */
    Tensor<DType> getGradients() {
        return Tensor<DType>(gradEmbeddingMatrix, {(size_t)blockSize, (size_t)embeddingDim});
    }
};


template <typename DType = float> Tensor<DType> softmax(Tensor<DType> input, DType temperature = (DType)1.0) {
    std::vector<size_t> os = input.shape().toVector();
    Tensor<DType> output(os);
    int totalBatchSize = output.size() / os.back();
    dim3 threadsPerBlock(BLOCKDIM);
    dim3 blocksPerGrid((os.back() + BLOCKDIM - 1) / BLOCKDIM, totalBatchSize);
    softmaxKernel<DType><<<blocksPerGrid, threadsPerBlock>>>(
        input.get(),
        output.get(),
        os.back(),
        totalBatchSize,
        temperature,
        false
    );
    cudaDeviceSynchronize();
    return output;
}

template <typename DType > Tensor<DType> softmax_duplicate(Tensor<DType> input, DType temperature) {
    std::vector<size_t> os = input.shape().toVector();
    /**
     * @brief Execute output operation.
     */
    Tensor<DType> output(os);
    int totalBatchSize = output.size() / os.back();

    /**
     * @brief Execute threadsPerBlock operation.
     */
    dim3 threadsPerBlock(BLOCKDIM);
    dim3 blocksPerGrid((os.back() + BLOCKDIM - 1) / BLOCKDIM, totalBatchSize);
    softmaxKernel<DType><<<blocksPerGrid, threadsPerBlock>>>(
        input.get(),
        output.get(),
        os.back(),
        totalBatchSize,
        temperature,
        false
    );
    cudaDeviceSynchronize();
    return output;
}


template <typename DType = float>
struct UnEmbedding {
    std::shared_ptr<DType[]> embeddingMatrix; ///< Embedding matrix of shape (vocabSize, embeddingDim).
    std::shared_ptr<DType[]> gradEmbeddingMatrix; ///< Gradient accumulator for embedding matrix.
    int vocabSize; ///< Size of the vocabulary.
    int embeddingDim; ///< Dimensionality of the embeddings.

    UnEmbedding(int vocabSize, int embeddingDim) 
        : vocabSize(vocabSize), embeddingDim(embeddingDim) 
    {
        embeddingMatrix = cudaMakeShared<DType>(vocabSize * embeddingDim);
        gradEmbeddingMatrix = cudaMakeShared<DType>(vocabSize * embeddingDim);
    }

    /**
     * @brief Execute forward operation.
     */
    Tensor<DType> forward(Tensor<DType> input) {
        std::vector<size_t> os = input.shape().toVector();
        os[os.size() - 1] = vocabSize;
        Tensor<DType> output(os);
        int totalBatchSize = output.size() / vocabSize;

        dim3 threadsPerBlock(BLOCKDIM);
        dim3 blocksPerGrid((vocabSize + BLOCKDIM - 1) / BLOCKDIM, totalBatchSize);
        linearForward<DType><<<blocksPerGrid, threadsPerBlock>>>(
            input.get(),
            output.get(),
            embeddingMatrix.get(),
            nullptr,
            embeddingDim,
            vocabSize,
            totalBatchSize
        );
        cudaDeviceSynchronize();
        return output;
    }

    /**
     * @brief Execute operator() operation.
     */
    Tensor<DType> operator()(Tensor<DType> input) {
        return forward(input);
    }

    /**
     * @brief Execute backward operation.
     */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) {
        int totalBatchSize = gradOutput.size() / vocabSize;
        dim3 threadsPerBlock(BLOCKDIM);
        dim3 blocksPerGrid((embeddingDim + BLOCKDIM - 1) / BLOCKDIM, totalBatchSize);
        linearBackwardWB<DType><<<blocksPerGrid, threadsPerBlock>>>(
            input.get(),
            gradOutput.get(),
            gradEmbeddingMatrix.get(),
            nullptr,
            embeddingDim,
            vocabSize,
            totalBatchSize
        );

        Tensor<DType> gradInput(input.shape().toVector());
        blocksPerGrid = dim3((vocabSize + BLOCKDIM - 1) / BLOCKDIM, totalBatchSize);
        linearBackward<DType><<<blocksPerGrid, threadsPerBlock>>>(
            gradInput.get(),
            gradOutput.get(),
            embeddingMatrix.get(),
            embeddingDim,
            vocabSize,
            totalBatchSize
        );
        cudaDeviceSynchronize();
        return gradInput;
    }

    /**
     * @brief Execute clone operation.
     */
    UnEmbedding<DType> clone() {
        UnEmbedding copy(vocabSize, embeddingDim);
        cudaMemcpy(copy.embeddingMatrix.get(), embeddingMatrix.get(), vocabSize * embeddingDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        return copy;
    }

    /**
     * @brief Execute zeroGrad operation.
     */
    void zeroGrad() {
        cudaMemset(gradEmbeddingMatrix.get(), 0, vocabSize * embeddingDim * sizeof(DType));
    }

    /**
     * @brief Execute sgdUpdate operation.
     */
    void sgdUpdate(DType lr) {
        int totalSize = vocabSize * embeddingDim;
        int threads = 256;
        int blocks = (totalSize + threads - 1) / threads;
        sgdUpdateKernel<DType><<<blocks, threads>>>(
            embeddingMatrix.get(),
            gradEmbeddingMatrix.get(),
            lr,
            totalSize
        );
        cudaDeviceSynchronize();
    }

    /**
     * @brief Execute getParameters operation.
     */
    Tensor<DType> getParameters() {
        return Tensor<DType>(embeddingMatrix, {(size_t)vocabSize, (size_t)embeddingDim});
    }

    /**
     * @brief Execute setParameters operation.
     */
    void setParameters(Tensor<DType> t) {
        embeddingMatrix = t.dataPtr();
    }

    /**
     * @brief Execute getGradients operation.
     */
    Tensor<DType> getGradients() {
        return Tensor<DType>(gradEmbeddingMatrix, {(size_t)vocabSize, (size_t)embeddingDim});
    }
};


/**
 * @brief CrossEntropyLoss struct implementing forward and backward passes for cross-entropy loss with optional softmax.
 * @tparam DType Data type for loss and gradients (default: float).
 * @tparam IdType Data type for target class indices (default: int).
 * 
 * @note This is a helper class with no members, just functions.
 * Call forward() after softmax, and use backward() to compute gradients for
 * the pre-softmax outputs.
 */
template <typename DType = float, typename IdType = int>
struct CrossEntropyLoss {
    static Tensor<DType> forward(Tensor<DType> output, Tensor<IdType> target, DType temperature = (DType)1.0) {
        std::vector<size_t> os = output.shape().toVector();
        os.pop_back();
        /**
         * @brief Execute loss operation.
         */
        Tensor<DType> loss(os);
        int totalBatchSize = loss.size();

        /**
         * @brief Execute threadsPerBlock operation.
         */
        dim3 threadsPerBlock(BLOCKDIM);
        dim3 blocksPerGrid((totalBatchSize + BLOCKDIM - 1) / BLOCKDIM);
        crossEntropyLossKernel<DType, IdType><<<blocksPerGrid, threadsPerBlock>>>(
            output.get(),
            target.get(),
            loss.get(),
            nullptr,
            os.back(),
            totalBatchSize,
            temperature
        );
        cudaDeviceSynchronize();
        return loss;
    }

//     Tensor<DType> forward(Tensor<DType> output, Tensor<IdType> target, DType temperature = (DType)1.0) {
// return CrossEntropyLoss<DType, IdType>::forward(output, target, temperature);
// }

    Tensor<DType> operator()(Tensor<DType> output, Tensor<IdType> target, DType temperature = (DType)1.0) {
        return CrossEntropyLoss<DType, IdType>::forward(output, target, temperature);
    }

    static Tensor<DType> backward(Tensor<DType> output, Tensor<IdType> target, DType temperature = (DType)1.0) {
        std::vector<size_t> os = output.shape().toVector();
        /**
         * @brief Execute inputGrad operation.
         */
        Tensor<DType> inputGrad(os);
        int totalBatchSize = inputGrad.size() / os.back();

        /**
         * @brief Execute threadsPerBlock operation.
         */
        dim3 threadsPerBlock(BLOCKDIM);
        dim3 blocksPerGrid((totalBatchSize + BLOCKDIM - 1) / BLOCKDIM);
        crossEntropyLossKernel<DType, IdType><<<blocksPerGrid, threadsPerBlock>>>(
            output.get(),
            target.get(),
            nullptr,
            inputGrad.get(),
            os.back(),
            totalBatchSize,
            temperature
        );
        cudaDeviceSynchronize();
        return inputGrad;
    }

//     Tensor<DType> backward(Tensor<DType> output, Tensor<IdType> target, DType temperature = (DType)1.0) {
// return CrossEntropyLoss<DType, IdType>::backward(output, target, temperature);
// }
};


#endif // EMBEDDING_LAYER