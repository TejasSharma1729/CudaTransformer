#include "ModuleLayer.cu"
#include "attention_qkv_kernels.cu"

#ifndef QKV_PROJECTION_LAYER
#define QKV_PROJECTION_LAYER

/**
 * @brief QKV projection layer for multi-head attention.
 * NOT a Layer - internal utility used by AttentionLayer.
 * Projects input to Q, K, V and reshapes to multi-head format.
 * @tparam DType The data type used for computations.
 */
template <typename DType = float> struct QKVProjectionLayer {
    std::shared_ptr<DType[]> weightsQuery = nullptr;
    std::shared_ptr<DType[]> weightsQueryGrad = nullptr;
    std::shared_ptr<DType[]> biasesQuery = nullptr;
    std::shared_ptr<DType[]> biasesQueryGrad = nullptr;

    std::shared_ptr<DType[]> weightsKey = nullptr;
    std::shared_ptr<DType[]> weightsKeyGrad = nullptr;
    std::shared_ptr<DType[]> biasesKey = nullptr;
    std::shared_ptr<DType[]> biasesKeyGrad = nullptr;

    std::shared_ptr<DType[]> weightsValue = nullptr;
    std::shared_ptr<DType[]> weightsValueGrad = nullptr;
    std::shared_ptr<DType[]> biasesValue = nullptr;
    std::shared_ptr<DType[]> biasesValueGrad = nullptr;

    int inputDim = 1;
    int numHeads = 1;
    int headDim = 1;

    /**
     * @brief Constructs a QKVProjectionLayer.
     * @param inputDim Size of input features.
     * @param numHeads Number of attention heads.
     * @param headDim Dimension per head.
     */
    QKVProjectionLayer(int inputDim, int numHeads, int headDim)
        : inputDim(inputDim), numHeads(numHeads), headDim(headDim)
    {
        int projSize = inputDim * headDim * numHeads;
        int biasSize = headDim * numHeads;
        weightsQuery = cudaMakeShared<DType>(projSize);
        biasesQuery = cudaMakeShared<DType>(biasSize);
        weightsKey = cudaMakeShared<DType>(projSize);
        biasesKey = cudaMakeShared<DType>(biasSize);
        weightsValue = cudaMakeShared<DType>(projSize);
        biasesValue = cudaMakeShared<DType>(biasSize);
    }

    /** @brief Clones the QKV projection layer. */
    std::shared_ptr<QKVProjectionLayer<DType>> clone() const {
        auto clonedLayer = std::make_shared<QKVProjectionLayer<DType>>(inputDim, numHeads, headDim);
        int projSize = inputDim * headDim * numHeads;
        int biasSize = headDim * numHeads;
        cudaMemcpy(clonedLayer->weightsQuery.get(), this->weightsQuery.get(), projSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->biasesQuery.get(), this->biasesQuery.get(), biasSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->weightsKey.get(), this->weightsKey.get(), projSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->biasesKey.get(), this->biasesKey.get(), biasSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->weightsValue.get(), this->weightsValue.get(), projSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->biasesValue.get(), this->biasesValue.get(), biasSize * sizeof(DType), cudaMemcpyDeviceToDevice);
        return clonedLayer;
    }
};

#endif // QKV_PROJECTION_LAYER
