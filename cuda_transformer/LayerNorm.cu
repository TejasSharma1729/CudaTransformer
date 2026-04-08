#include "ModuleLayer.cu"
#include "layer_norm_kernels.cu"

#ifndef LAYERNORM_LAYER
#define LAYERNORM_LAYER


template <typename DType = float> struct LayerNormLayer : public Layer<DType> {
    int inputDim;
    int cachedN;
    DType epsilon;
    
    std::shared_ptr<DType[]> weights;
    std::shared_ptr<DType[]> biases;
    std::shared_ptr<DType[]> weightsGrad;
    std::shared_ptr<DType[]> biasesGrad;

    std::shared_ptr<DType[]> cache_mean;
    std::shared_ptr<DType[]> cache_inv_std;
    
    LayerNormLayer(int inputDim, DType epsilon = 1e-5) 
        : inputDim(inputDim), epsilon(epsilon), cachedN(0) {
        weights = cudaMakeShared<DType>(inputDim, 1.0f);
        biases = cudaMakeShared<DType>(inputDim, 0.0f);
        weightsGrad = cudaMakeShared<DType>(inputDim, 0.0f);
        biasesGrad = cudaMakeShared<DType>(inputDim, 0.0f);
    }

    /**
     * @brief Execute getParameters operation.
     */
    std::map<std::string, Tensor<DType>> getParameters() override {
        return {
            {"weights", Tensor<DType>(weights, {(size_t)inputDim})},
            {"biases", Tensor<DType>(biases, {(size_t)inputDim})}
        };
    }

    /**
     * @brief Execute setParameters operation.
     */
    void setParameters(const std::map<std::string, Tensor<DType>>& params) override {
        if (params.count("weights")) {
            cudaMemcpy(weights.get(), params.at("weights").get(), inputDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        }
        if (params.count("biases")) {
            cudaMemcpy(biases.get(), params.at("biases").get(), inputDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        }
    }

    /**
     * @brief Execute clone operation.
     */
    std::shared_ptr<Layer<DType>> clone() override {
        auto cloned = std::make_shared<LayerNormLayer<DType>>(inputDim, epsilon);
        cudaMemcpy(cloned->weights.get(), this->weights.get(), inputDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        cudaMemcpy(cloned->biases.get(), this->biases.get(), inputDim * sizeof(DType), cudaMemcpyDeviceToDevice);
        return cloned;
    }

    /**
     * @brief Execute zeroGrad operation.
     */
    void zeroGrad() override {
        cudaMemset(weightsGrad.get(), 0, inputDim * sizeof(DType));
        cudaMemset(biasesGrad.get(), 0, inputDim * sizeof(DType));
    }

    /**
     * @brief Execute sgdUpdate operation.
     */
    void sgdUpdate(DType lr) override {
        layerNormUpdateKernel<<< (inputDim + 255) / 256, 256 >>>(
            weights.get(), biases.get(), weightsGrad.get(), biasesGrad.get(), inputDim, lr
        );
        cudaDeviceSynchronize();
    }

    /**
     * @brief Execute forward operation.
     */
    Tensor<DType> forward(Tensor<DType> input) override {
        int D = input.shape()[input.nDim() - 1];
        assert(D == inputDim);
        int N = input.size() / D;
        
        Tensor<DType> output(input.shape().toVector());

        if (cachedN < N) {
            cache_mean = cudaMakeShared<DType>(N);
            cache_inv_std = cudaMakeShared<DType>(N);
            cachedN = N;
        }

        int threads = 256;
        size_t smem = threads * sizeof(DType);
        layerNormForwardKernel<<<N, threads, smem>>>(
            input.get(), output.get(), weights.get(), biases.get(),
            D, N, epsilon, cache_mean.get(), cache_inv_std.get()
        );

        return output;
    }

    /**
     * @brief Execute backward operation.
     */
    Tensor<DType> backward(Tensor<DType> input, Tensor<DType> gradOutput) override {
        int D = input.shape()[input.nDim() - 1];
        int N = input.size() / D;
        
        Tensor<DType> gradInput(input.shape().toVector());

        int threads = 256;
        size_t smem = 2 * threads * sizeof(DType);
        layerNormBackwardKernel<<<N, threads, smem>>>(
            gradOutput.get(), input.get(), gradInput.get(),
            weights.get(), weightsGrad.get(), biasesGrad.get(),
            cache_mean.get(), cache_inv_std.get(), D, N
        );

        return gradInput;
    }
};

template <typename DType> std::shared_ptr<Layer<DType>> LayerNorm(int inputDim, DType epsilon = (DType)1e-5) {
    return std::make_shared<LayerNormLayer<DType>>(inputDim, epsilon);
}

#endif
