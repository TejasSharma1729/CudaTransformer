#include "embedding_kernels.cu"
#include "linear_kernels.cu"
#include "Tensor.cu"

#ifndef EMBEDDING_LAYER
#define EMBEDDING_LAYER

/**
 * @brief Represents the token embedding layer that maps input token IDs to dense vectors.
 * @tparam DType The data type used for the embedding vectors (e.g., float).
 * @tparam IdType The data type used for token IDs (e.g., int).
 * The embedding layer maintains an embedding matrix of shape (vocabSize, embeddingDim) where 
 * each row corresponds to the embedding vector for a specific token ID. 
 * The forward pass retrieves the embedding vectors for the input token IDs, 
 * while the backward pass accumulates gradients for the embedding matrix based on the output gradients.
 */
template <typename DType = float, typename IdType = int>
struct TokenEmbedding {
    std::shared_ptr<DType[]> embeddingMatrix; /// Embedding matrix of shape (vocabSize, embeddingDim).
    std::shared_ptr<DType[]> gradEmbeddingMatrix; /// Gradient accumulator for embedding matrix.
    int vocabSize; /// Size of the vocabulary.
    int embeddingDim; /// Dimensionality of the embeddings.

    /**
     * @brief Constructor that initializes the embedding matrix and its gradient accumulator.
     * @param vocabSize The size of the vocabulary (number of unique tokens).
     * @param embeddingDim The dimensionality of the embedding vectors.
     */
    TokenEmbedding(int vocabSize, int embeddingDim) 
        : vocabSize(vocabSize), embeddingDim(embeddingDim) 
    {
        embeddingMatrix = cudaMakeShared<DType>(vocabSize * embeddingDim);
        gradEmbeddingMatrix = cudaMakeShared<DType>(vocabSize * embeddingDim);
    }

    /**
     * @brief The forward pass that takes a tensor of token IDs and returns the corresponding embeddings.
     * @param input A tensor of shape (batchSize, sequenceLength) containing token IDs.
     * @return A tensor of shape (batchSize, sequenceLength, embeddingDim) containing the
     * embedding vectors corresponding to the input token IDs. 
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
     * @brief Forward operator that allows the embedding layer to be called like a function.
     * @param input A tensor of shape (batchSize, sequenceLength) containing token IDs.
     * @return A tensor of shape (batchSize, sequenceLength, embeddingDim) containing the
     * embedding vectors corresponding to the input token IDs.
     */
    Tensor<DType> operator()(Tensor<IdType> input) {
        return forward(input);
    }

    /**
     * @brief Backward pass that takes the input token IDs and the gradient of the loss with respect to the output embeddings,
     * and accumulates the gradients for the embedding matrix.
     * @param input A tensor of shape (batchSize, sequenceLength) containing token IDs that were used in the forward pass.
     * @param gradOutput A tensor of shape (batchSize, sequenceLength, embeddingDim) containing the 
     *      gradient of the loss with respect to the output embeddings
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
     * @brief Creates a deep copy of the TokenEmbedding instance, 
     * including copying the embedding matrix to the new instance.
     */
    TokenEmbedding clone() {
        TokenEmbedding copy(vocabSize, embeddingDim);
        cudaMemcpy(copy.embeddingMatrix.get(), embeddingMatrix.get(), vocabSize * embeddingDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        return copy;
    }

    /**
     * @brief Zeros out the gradients in the gradEmbeddingMatrix, 
     * preparing it for the next backward pass.
     */
    void zeroGrad() {
        cudaMemset(gradEmbeddingMatrix.get(), 0, vocabSize * embeddingDim * sizeof(DType));
    }

    /**
     * @brief Performs an SGD update on the embedding matrix using the 
     * accumulated gradients in gradEmbeddingMatrix and the specified learning rate.
     * @param lr The learning rate to use for the SGD update.
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
     * @brief Returns a Tensor that shares the same underlying embedding matrix data.
     * This allows external code to access the embedding parameters for inspection, saving, or custom updates.
     * @return A Tensor of shape (vocabSize, embeddingDim) that shares the same data as the embedding matrix.
     */
    Tensor<DType> getParameters() {
        return Tensor<DType>(embeddingMatrix, {(size_t)vocabSize, (size_t)embeddingDim});
    }

    /**
     * @brief Sets the embedding matrix parameters from a given Tensor. The provided Tensor should have the same shape as the embedding matrix.
     * This allows external code to set the embedding parameters, for example when loading a saved model state.
     * @param t A Tensor of shape (vocabSize, embeddingDim) containing the new embedding parameters. 
     * The data from this Tensor will be used to update the embedding matrix.
     */
    void setParameters(Tensor<DType> t) {
        embeddingMatrix = t.dataPtr();
    }

    /**
     * @brief Returns a Tensor that shares the same underlying gradient data as the embedding matrix.
     * This allows external code to access the accumulated gradients for inspection or custom updates.
     * @return A Tensor of shape (vocabSize, embeddingDim) that shares the same data as the gradient accumulator.
     */
    Tensor<DType> getGradients() {
        return Tensor<DType>(gradEmbeddingMatrix, {(size_t)vocabSize, (size_t)embeddingDim});
    }
};


/**
 * @brief Represents the positional embedding layer that adds positional information to the token embeddings.
 * @tparam DType The data type used for the embedding vectors (e.g., float).
 * @tparam IdType The data type used for token IDs (e.g., int).
 * 
 * The positional embedding layer maintains an embedding matrix of shape (blockSize, embeddingDim) where
 * each row corresponds to the positional embedding vector for a specific position in the input sequence.
 */
template <typename DType = float, typename IdType = int>
struct PositionEmbedding {
    std::shared_ptr<DType[]> embeddingMatrix; /// Embedding matrix of shape (vocabSize, embeddingDim).
    std::shared_ptr<DType[]> gradEmbeddingMatrix; /// Gradient accumulator for embedding matrix.
    int blockSize; /// Maximum sequence length (block size) that this positional embedding can handle.
    int embeddingDim; /// Dimensionality of the embeddings.

    /**
     * @brief Constructor that initializes the positional embedding matrix and its gradient accumulator.
     * @param blockSize The maximum sequence length (block size) that this positional embedding can handle.
     * @param embeddingDim The dimensionality of the embedding vectors.
     */
    PositionEmbedding(int blockSize, int embeddingDim) 
        : blockSize(blockSize), embeddingDim(embeddingDim) 
    {
        embeddingMatrix = cudaMakeShared<DType>(blockSize * embeddingDim);
        gradEmbeddingMatrix = cudaMakeShared<DType>(blockSize * embeddingDim);
    }

    /**
     * @brief The forward pass that takes a tensor of token IDs and returns the corresponding positional embeddings.
     * @param input A tensor of shape (batchSize, sequenceLength) containing token IDs. 
     * The actual token IDs are not used in this layer, but the shape is used to determine the positions.
     * @return A tensor of shape (batchSize, sequenceLength, embeddingDim) containing the 
     * positional embedding vectors corresponding to the positions of the input token IDs.
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
     * @brief The forward pass that takes a tensor of token IDs and returns the corresponding positional embeddings.
     * @param input A tensor of shape (batchSize, sequenceLength) containing token IDs. 
     * The actual token IDs are not used in this layer, but the shape is used to determine the positions.
     * @return A tensor of shape (batchSize, sequenceLength, embeddingDim) containing the 
     * positional embedding vectors corresponding to the positions of the input token IDs.
     */
    Tensor<DType> operator()(Tensor<IdType> input) {
        return forward(input);
    }

    /**
     * @brief Backward pass that takes the input token IDs and the gradient of the loss with respect to the output positional embeddings,
     * @param input A tensor of shape (batchSize, sequenceLength) containing token IDs. 
     * The actual token IDs are not used in this layer, but the shape is used to determine the positions.
     * @param gradOutput The gradient of the loss with respect to the output positional embeddings.
     * The backward pass will accumulate gradients for the positional embedding matrix based on 
     * the output gradients and the positions of the input token IDs.
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
     * @brief Creates a deep copy of the PositionEmbedding instance, 
     * including copying the embedding matrix to the new instance.
     */
    PositionEmbedding clone() {
        PositionEmbedding copy(blockSize, embeddingDim);
        cudaMemcpy(copy.embeddingMatrix.get(), embeddingMatrix.get(), blockSize * embeddingDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        return copy;
    }

    /**
     * @brief Zeros out the gradients in the gradEmbeddingMatrix, 
     * preparing it for the next backward pass.
     */
    void zeroGrad() {
        cudaMemset(gradEmbeddingMatrix.get(), 0, blockSize * embeddingDim * sizeof(DType));
    }

    /**
     * @brief Performs an SGD update on the positional embedding matrix using the 
     * accumulated gradients in gradEmbeddingMatrix and the specified learning rate.
     * @param lr The learning rate to use for the SGD update.
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
     * @brief Returns a Tensor that shares the same underlying embedding matrix data.
     * This allows external code to access the positional embedding parameters for inspection, saving, or custom updates.
     * @return A Tensor of shape (blockSize, embeddingDim) that shares the
     */
    Tensor<DType> getParameters() {
        return Tensor<DType>(embeddingMatrix, {(size_t)blockSize, (size_t)embeddingDim});
    }

    /**
     * @brief Sets the positional embedding matrix parameters from a given Tensor. 
     * The provided Tensor should have the same shape as the embedding matrix.
     * @param t A Tensor of shape (blockSize, embeddingDim) containing the new positional embedding parameters.
     */
    void setParameters(Tensor<DType> t) {
        embeddingMatrix = t.dataPtr();
    }

    /**
     * @brief Returns a Tensor that shares the same underlying gradient data as the positional embedding matrix.
     * This allows external code to access the accumulated gradients for inspection or custom updates.
     */
    Tensor<DType> getGradients() {
        return Tensor<DType>(gradEmbeddingMatrix, {(size_t)blockSize, (size_t)embeddingDim});
    }
};


/**
 * @brief Represents the unembedding layer that maps the output of the transformer back to the vocabulary space for prediction.
 * @tparam DType The data type used for the embedding vectors (e.g., float).
 * The unembedding layer maintains an embedding matrix of shape (vocabSize, embeddingDim) where each row corresponds to the unembedding vector for a specific token ID. 
 * The forward pass computes the logits for each token in the vocabulary by performing a matrix multiplication between
 */
template <typename DType = float>
struct UnEmbedding {
    std::shared_ptr<DType[]> embeddingMatrix; /// Embedding matrix of shape (vocabSize, embeddingDim).
    std::shared_ptr<DType[]> gradEmbeddingMatrix; /// Gradient accumulator for embedding matrix.
    int vocabSize; /// Size of the vocabulary.
    int embeddingDim; /// Dimensionality of the embeddings.

    /**
     * @brief Constructor that initializes the unembedding matrix and its gradient accumulator.
     * @param vocabSize The size of the vocabulary (number of unique tokens).
     * @param embeddingDim The dimensionality of the embedding vectors.
     */
    UnEmbedding(int vocabSize, int embeddingDim) 
        : vocabSize(vocabSize), embeddingDim(embeddingDim) 
    {
        embeddingMatrix = cudaMakeShared<DType>(vocabSize * embeddingDim);
        gradEmbeddingMatrix = cudaMakeShared<DType>(vocabSize * embeddingDim);
    }

    /**
     * @brief Performs the forward pass for the unembedding layer, giving embeddings of size vocabSize.
     * Note that to get logits, a sofrmax should be done after this.
     * @param input The input tensor of shape (batchSize, sequenceLength, embeddingDim).
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
     * @brief Performs the forward pass for the unembedding layer, giving embeddings of size vocabSize.
     * Note that to get logits, a sofrmax should be done after this.
     * @param input The input tensor of shape (batchSize, sequenceLength, embeddingDim).
     */
    Tensor<DType> operator()(Tensor<DType> input) {
        return forward(input);
    }

    /**
     * @brief Performs the backward pass for the unembedding layer.
     * @param input The input tensor of shape (batchSize, sequenceLength, embeddingDim).
     * @param gradOutput The gradient tensor of shape (batchSize, sequenceLength, vocabSize).
     * @return The gradient tensor of shape (batchSize, sequenceLength, embeddingDim).
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
     * @brief Creates a deep copy of the UnEmbedding instance, including copying the embedding matrix to the new instance.
     */
    UnEmbedding<DType> clone() {
        UnEmbedding copy(vocabSize, embeddingDim);
        cudaMemcpy(copy.embeddingMatrix.get(), embeddingMatrix.get(), vocabSize * embeddingDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        return copy;
    }

    /**
     * @brief Zeros out the gradients in the gradEmbeddingMatrix, preparing it for the next backward pass.
     */
    void zeroGrad() {
        cudaMemset(gradEmbeddingMatrix.get(), 0, vocabSize * embeddingDim * sizeof(DType));
    }

    /**
     * @brief Performs an SGD update on the unembedding matrix using the 
     * accumulated gradients in gradEmbeddingMatrix and the specified learning rate.
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
     * @brief Returns a Tensor that shares the same underlying embedding matrix data. 
     * This allows external code to access the unembedding parameters for inspection, saving, or custom updates.
     */
    Tensor<DType> getParameters() {
        return Tensor<DType>(embeddingMatrix, {(size_t)vocabSize, (size_t)embeddingDim});
    }

    /**
     * @brief Sets the unembedding matrix parameters from a given Tensor. 
     * The provided Tensor should have the same shape as the embedding matrix.
     * This allows external code to set the unembedding parameters, 
     * for example when loading a saved model state.
     * @param t A Tensor of shape (vocabSize, embeddingDim) containing the new unembedding parameters. 
     * The data from this Tensor will be used to update the unembedding matrix.
     */
    void setParameters(Tensor<DType> t) {
        embeddingMatrix = t.dataPtr();
    }

    /**
     * @brief Returns a Tensor that shares the same underlying gradient data as the unembedding matrix. 
     * This allows external code to access the accumulated gradients for inspection or custom updates.
     */
    Tensor<DType> getGradients() {
        return Tensor<DType>(gradEmbeddingMatrix, {(size_t)vocabSize, (size_t)embeddingDim});
    }
};


#endif // EMBEDDING_LAYER