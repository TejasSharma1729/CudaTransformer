#include "ModuleLayer.cu"
#include "attention_proj_kernels.cu"

#ifndef OUTPUT_PROJECTION_LAYER
#define OUTPUT_PROJECTION_LAYER

/**
 * @brief Output projection layer for multi-head attention.
 * NOT a Layer - internal utility used by AttentionLayer.
 * Projects concatenated attention heads back to input dimension.
 * @tparam DType The data type used for computations.
 */
template <typename DType = float> struct OutputProjectionLayer {
    std::shared_ptr<DType[]> weightsProj = nullptr;
    std::shared_ptr<DType[]> weightsProjGrad = nullptr;
    std::shared_ptr<DType[]> biasesProj = nullptr;
    std::shared_ptr<DType[]> biasesProjGrad = nullptr;

    int inputDim = 1;
    int headDim = 1;
    int numHeads = 1;

    /**
     * @brief Constructs an OutputProjectionLayer.
     * @param inputDim Size of output features (same as input to attention).
     * @param headDim Dimension per attention head.
     * @param numHeads Number of attention heads.
     */
    OutputProjectionLayer(int inputDim, int headDim, int numHeads)
        : inputDim(inputDim), headDim(headDim), numHeads(numHeads)
    {
        int totalHeadDim = headDim * numHeads;
        weightsProj = cudaMakeShared<DType>(totalHeadDim * inputDim);
        biasesProj = cudaMakeShared<DType>(inputDim);
    }

    /** @brief Clones the output projection layer. */
    std::shared_ptr<OutputProjectionLayer<DType>> clone() const {
        auto clonedLayer = std::make_shared<OutputProjectionLayer<DType>>(inputDim, headDim, numHeads);
        int totalHeadDim = headDim * numHeads;
        cudaMemcpy(clonedLayer->weightsProj.get(), this->weightsProj.get(), totalHeadDim * inputDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clonedLayer->biasesProj.get(), this->biasesProj.get(), inputDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        return clonedLayer;
    }

    /**
     * @brief Forward pass: project attention output.
     * @param input Attention output [batch, seq_len, head_dim * num_heads].
     * @param saveStates Flag to save states (unused here as no states are saved).
     * @return Projected output [batch, seq_len, input_dim].
     */
    Tensor<DType> forward(Tensor<DType> input) {
        int sequenceLength = input.shape()[input.nDim() - 2];
        int batchSize = input.size() / (sequenceLength * (headDim * numHeads));
        std::vector<size_t> outputShape = {(size_t)batchSize, (size_t)sequenceLength, (size_t)inputDim};
        Tensor<DType> output(outputShape);
        
        dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
        dim3 gridProj((inputDim + BLOCKDIM - 1) / BLOCKDIM, (batchSize * sequenceLength + BLOCKDIM - 1) / BLOCKDIM);
        attentionProj<DType><<<gridProj, threadsPerBlock, 2 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            input.get(), output.get(), weightsProj.get(), biasesProj.get(),
            inputDim, headDim, sequenceLength, numHeads, batchSize
        );

        return output;
    }

    /** @brief Backward pass. */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) {
        if (weightsProjGrad == nullptr) {
            int totalHeadDim = headDim * numHeads;
            weightsProjGrad = cudaMakeShared<DType>(totalHeadDim * inputDim);
            biasesProjGrad = cudaMakeShared<DType>(inputDim);
        }

        int sequenceLength = input.shape()[input.nDim() - 2];
        int batchSize = input.size() / (sequenceLength * (headDim * numHeads));
        dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
        int totalHeadDim = headDim * numHeads;

        // Compute gradients w.r.t. weights and biases
        dim3 gridProjWB((totalHeadDim + BLOCKDIM - 1) / BLOCKDIM, (inputDim + BLOCKDIM - 1) / BLOCKDIM);
        projBackwardWB<DType><<<gridProjWB, threadsPerBlock, 2 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            input.get(), gradOutput.get(), weightsProjGrad.get(), biasesProjGrad.get(),
            inputDim, headDim, sequenceLength, numHeads, batchSize
        );

        // Compute gradients w.r.t. input
        Tensor<DType> inputGrad(input.shape().toVector());
        dim3 gridProj((totalHeadDim + BLOCKDIM - 1) / BLOCKDIM, (batchSize * sequenceLength + BLOCKDIM - 1) / BLOCKDIM);
        projBackward<DType><<<gridProj, threadsPerBlock, 2 * BLOCKDIM * BLOCKDIM * sizeof(DType)>>>(
            inputGrad.get(), gradOutput.get(), weightsProj.get(),
            inputDim, headDim, sequenceLength, numHeads, batchSize
        );

        return inputGrad;
    }
};

#endif // OUTPUT_PROJECTION_LAYER
