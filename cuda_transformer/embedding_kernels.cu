#include "headers.cu"

#ifndef EMBEDDING_KERNELS
#define EMBEDDING_KERNELS


/**
 * @brief Gathers embedding vectors from the embedding matrix for each token ID in the input.
 *
 * Each thread block covers one position in the flattened batch×sequence dimension (`batchIdx`)
 * and one tile of the embedding dimension (`featureIdx`).  Out-of-range token IDs are mapped
 * to the zero vector so the output is always safe to read.
 *
 * Grid:  ((embeddingDim + BLOCKDIM - 1) / BLOCKDIM, totalBatchSize)
 * Block: (BLOCKDIM)
 *
 * @tparam DType   Data type of the embedding vectors (e.g. float, __half).
 * @tparam IdType  Integer type used for token IDs (e.g. int).
 * @param input           Device pointer to flattened token ID array [totalBatchSize].
 * @param output          Device pointer to the output embedding tensor [totalBatchSize, embeddingDim].
 * @param embeddingMatrix Device pointer to the embedding table [vocabSize, embeddingDim].
 * @param vocabSize       Total number of tokens in the vocabulary.
 * @param embeddingDim    Dimensionality of each embedding vector.
 * @param totalBatchSize  Product of batchSize and sequenceLength (flattened token count).
 */
template <typename DType = float, typename IdType = int> __global__ void tokenEmbeddingForwardKernel(
    const IdType* input,
    DType* output,
    const DType* embeddingMatrix,
    int vocabSize,
    int embeddingDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIdx = blockIdx.y;
    if (featureIdx < embeddingDim && batchIdx < totalBatchSize) {
        int tokenId = (int)input[batchIdx];
        if (tokenId >= 0 && tokenId < vocabSize) {
            output[batchIdx * embeddingDim + featureIdx] = embeddingMatrix[tokenId * embeddingDim + featureIdx];
        } else {
            output[batchIdx * embeddingDim + featureIdx] = static_cast<DType>(0);
        }
    }
}

/**
 * @brief Broadcasts learned positional embedding vectors into the output tensor.
 *
 * Each position `batchIdx` in the flattened sequence dimension receives its corresponding
 * row from the positional embedding table.  The actual token IDs (`input`) are ignored;
 * only the position index (`batchIdx`) is used, so the kernel is the same for every batch.
 *
 * Grid:  ((embeddingDim + BLOCKDIM - 1) / BLOCKDIM, totalBatchSize)
 * Block: (BLOCKDIM)
 *
 * @tparam DType   Data type of the embedding vectors (e.g. float, __half).
 * @tparam IdType  Integer type of the token ID array (unused in this kernel beyond shape).
 * @param input           Device pointer to flattened token ID array [totalBatchSize] (shape only).
 * @param output          Device pointer to the output tensor [totalBatchSize, embeddingDim].
 * @param embeddingMatrix Device pointer to the positional embedding table [blockSize, embeddingDim].
 * @param embeddingDim    Dimensionality of each positional embedding vector.
 * @param totalBatchSize  Product of batchSize and sequenceLength (flattened position count).
 */
template <typename DType = float, typename IdType = int> __global__ void positionEmbeddingForwardKernel(
    const IdType* input,
    DType* output,
    const DType* embeddingMatrix,
    int embeddingDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIdx = blockIdx.y;
    if (featureIdx < embeddingDim && batchIdx < totalBatchSize) {
        output[batchIdx * embeddingDim + featureIdx] = embeddingMatrix[batchIdx * embeddingDim + featureIdx];
    }
}

/**
 * @brief Scatter-accumulates output gradients into the token embedding gradient matrix.
 *
 * For each position in the batch, the gradient w.r.t. the output embedding at that position
 * is atomically added into the row of `gradEmbeddingMatrix` corresponding to the token ID.
 * Out-of-range token IDs are silently ignored.
 *
 * Grid:  ((embeddingDim + BLOCKDIM - 1) / BLOCKDIM, totalBatchSize)
 * Block: (BLOCKDIM)
 *
 * @tparam DType   Data type of the gradients and embedding table (e.g. float, __half).
 * @tparam IdType  Integer type used for token IDs (e.g. int).
 * @param input               Device pointer to flattened token ID array [totalBatchSize].
 * @param gradOutput          Device pointer to upstream gradients [totalBatchSize, embeddingDim].
 * @param gradEmbeddingMatrix Device pointer to the gradient accumulator [vocabSize, embeddingDim].
 *                            Updated via atomicAdd (safe for concurrent writes from same token).
 * @param vocabSize           Total number of tokens in the vocabulary.
 * @param embeddingDim        Dimensionality of each embedding vector.
 * @param totalBatchSize      Product of batchSize and sequenceLength.
 */
template <typename DType = float, typename IdType = int> __global__ void tokenEmbeddingBackwardKernel(
    const IdType* input,
    const DType* gradOutput,
    DType* gradEmbeddingMatrix,
    int vocabSize,
    int embeddingDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIdx = blockIdx.y;
    if (featureIdx < embeddingDim && batchIdx < totalBatchSize) {
        int tokenId = (int)input[batchIdx];
        if (tokenId >= 0 && tokenId < vocabSize) {
            atomicAdd(&gradEmbeddingMatrix[tokenId * embeddingDim + featureIdx], gradOutput[batchIdx * embeddingDim + featureIdx]);
        }
    }
}

/**
 * @brief Accumulates output gradients into the positional embedding gradient matrix.
 *
 * The gradient for position `batchIdx` in the flattened sequence is atomically added into
 * the corresponding row of `gradEmbeddingMatrix`.  Unlike the token-embedding backward,
 * the row index equals the flattened position index directly (not a gathered token ID).
 *
 * Grid:  ((embeddingDim + BLOCKDIM - 1) / BLOCKDIM, totalBatchSize)
 * Block: (BLOCKDIM)
 *
 * @tparam DType   Data type of the gradients and embedding table (e.g. float, __half).
 * @tparam IdType  Integer type of the token ID array (shape only; values unused).
 * @param input               Device pointer to flattened token ID array [totalBatchSize] (unused).
 * @param gradOutput          Device pointer to upstream gradients [totalBatchSize, embeddingDim].
 * @param gradEmbeddingMatrix Device pointer to the gradient accumulator [blockSize, embeddingDim].
 *                            Updated via atomicAdd.
 * @param embeddingDim        Dimensionality of each positional embedding vector.
 * @param totalBatchSize      Product of batchSize and sequenceLength.
 */
template <typename DType = float, typename IdType = int> __global__ void positionEmbeddingBackwardKernel(
    const IdType* input,
    const DType* gradOutput,
    DType* gradEmbeddingMatrix,
    int embeddingDim,
    int totalBatchSize
) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIdx = blockIdx.y;
    if (featureIdx < embeddingDim && batchIdx < totalBatchSize) {
        atomicAdd(&gradEmbeddingMatrix[batchIdx * embeddingDim + featureIdx], gradOutput[batchIdx * embeddingDim + featureIdx]);
    }
}

// unembedding == just a linear layer.

#endif // EMBEDDING_KERNELS