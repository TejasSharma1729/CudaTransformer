#include "TransformerLayer.cu"
#include "Embedding.cu"

#ifndef NANOGPT
#define NANOGPT


enum SamplingMode { Greedy, Random, Nucleus, TopK };
// Not supported: beam search, 


template <typename DType = float, typename IdType = int> struct NanoGPT {
    TokenEmbedding<DType, IdType> tokenEmbedding;
    PositionEmbedding<DType, IdType> positionEmbedding;
    TransformerLayer<DType> transformerLayer;
    UnEmbedding<DType> unEmbedding;
    CheckpointLayer<DType> preUnembeddingCheckpoint;  // final, for efficient backprop of un-embedding
    CrossEntropyLoss<DType, IdType> crossEntropyLoss;
    DType temperature = (DType)1.0;

    NanoGPT(
        int vocabSize, int maxSeqLen, int embeddingDim, int numHeads, int headDim, int mlpDim, int numLayers,
        int checkpointGap = 0, double temperature = 1.0, ActivationType activationType = ActivationType::ReLU
    ) : tokenEmbedding(vocabSize, embeddingDim), positionEmbedding(maxSeqLen, embeddingDim),
        transformerLayer(embeddingDim, numHeads, headDim, mlpDim, numLayers, checkpointGap, activationType),
        unEmbedding(vocabSize, embeddingDim), preUnembeddingCheckpoint(), 
        temperature((DType)temperature), crossEntropyLoss() { }
    
    int vocabSize() const { return tokenEmbedding.vocabSize; }
    int maxSeqLen() const { return positionEmbedding.blockSize; }
    int embeddingDim() const { return transformerLayer.inputDim; }
    int numHeads() const { return transformerLayer.numHeads; }
    int headDim() const { return transformerLayer.headDim; }
    int mlpDim() const { return transformerLayer.mlpDim; }
    int numLayers() const { return transformerLayer.numLayers; }
    int checkpointGap() const { return transformerLayer.checkpointGap; }
    double getTemperature() const { return (DType)temperature; }
    void setTemperature(double temp) { temperature = (DType)temp; }
    ActivationType activationType() const { return transformerLayer.activationType; }

    /**
     * @brief Execute forward operation.
     */
    Tensor<DType> forward(Tensor<IdType> input) {
        Tensor<DType> embeddings = tokenEmbedding(input) + positionEmbedding(input);
        embeddings = transformerLayer(embeddings);
        embeddings = preUnembeddingCheckpoint(embeddings);
        Tensor<DType> preLogits = unEmbedding(embeddings);
        return softmax(preLogits, temperature);
    }

    /**
     * @brief Execute forward operation.
     */
    Tensor<DType> forward(pybind11::array_t<IdType> input) {
        return forward(Tensor<IdType>::fromNumpy(input));
    }

    /**
     * @brief Execute forward operation.
     */
    Tensor<DType> forward(std::vector<IdType> input) {
        Tensor<IdType> inputTensor = Tensor<IdType>::fromPointer({(size_t)input.size()}, input.data());
        return forward(inputTensor);
    }

    /**
     * @brief Execute operator() operation.
     */
    Tensor<DType> operator()(Tensor<IdType> input) {
        return forward(input);
    }

    /**
     * @brief Execute operator() operation.
     */
    Tensor<DType> operator()(pybind11::array_t<IdType> input) {
        return forward(input);
    }

    /**
     * @brief Execute operator() operation.
     */
    Tensor<DType> operator()(std::vector<IdType> input) {
        return forward(input);
    }

    /**
     * @brief Execute predict operation.
     */
    IdType predict(Tensor<DType> logits, SamplingMode mode, int K = 10, double P = 0.1) {
        assert(logits.nDim() == 2); // (seqLen, vocabSize) -- does not support batched requests.
        int seqLen = logits.shape(0);
        int vocabSize = logits.shape(1);
        /**
         * @brief Execute hostLogits operation.
         */
        std::vector<DType> hostLogits(vocabSize);
        size_t offset = (seqLen - 1) * vocabSize; // Get logits for the last token
        cudaMemcpy(hostLogits.data(), logits.get() + offset, vocabSize * sizeof(DType), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        if (mode == SamplingMode::Greedy) {
            return std::distance(hostLogits.begin(), std::max_element(hostLogits.begin(), hostLogits.end()));
        } else if (mode == SamplingMode::Random) {
            std::discrete_distribution<> dist(hostLogits.begin(), hostLogits.end());
            std::mt19937 gen(std::random_device{}());
            return dist(gen);
        }
        std::vector<std::pair<IdType, DType>> idLogitPairs;
        if (mode == SamplingMode::Nucleus) {
            for (int i = 0; i < vocabSize; i++) {
                if (hostLogits[i] > (DType)P) { // Filter out logits that are effectively -inf
                    idLogitPairs.emplace_back(i, hostLogits[i]);
                }
            }
        } else if (mode == SamplingMode::TopK) {
            for (int i = 0; i < vocabSize; i++) {
                idLogitPairs.emplace_back(i, hostLogits[i]);
            }
            std::partial_sort(idLogitPairs.begin(), idLogitPairs.begin() + K, idLogitPairs.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
            idLogitPairs.resize(K);
        }
        if (idLogitPairs.empty()) {
            return std::distance(hostLogits.begin(), std::max_element(hostLogits.begin(), hostLogits.end()));
        }
        std::vector<DType> filteredLogits;
        for (const auto& pair : idLogitPairs) {
            filteredLogits.push_back(pair.second);
        }
        std::discrete_distribution<> dist(filteredLogits.begin(), filteredLogits.end());
        std::mt19937 gen(std::random_device{}());
        return idLogitPairs[dist(gen)].first;
    }

    /**
     * @brief Execute sample operation.
     */
    IdType sample(Tensor<IdType> input, SamplingMode mode, int K = 10, double P = 0.1) {
        Tensor<DType> logits = forward(input);
        return predict(logits, mode, K, P);
    }

    /**
     * @brief Execute sample operation.
     */
    IdType sample(pybind11::array_t<IdType> input, SamplingMode mode, int K = 10, double P = 0.1) {
        Tensor<DType> logits = forward(input);
        return predict(logits, mode, K, P);
    }

    /**
     * @brief Execute sample operation.
     */
    IdType sample(std::vector<IdType> input, SamplingMode mode, int K = 10, double P = 0.1) {
        Tensor<DType> logits = forward(input);
        return predict(logits, mode, K, P);
    }

    /**
     * @brief Execute generate operation.
     */
    std::vector<IdType> generate(std::vector<IdType> input, int numTokens, SamplingMode mode, int K = 10, double P = 0.1) {
        std::vector<IdType> generatedIds = input;
        for (int i = 0; i < numTokens; i++) {
            Tensor<DType> logits = forward(generatedIds);
            IdType nextId = predict(logits, mode, K, P);
            generatedIds.push_back(nextId);
        }
        return generatedIds;
    }

    /**
     * @brief Execute generate operation.
     */
    pybind11::array_t<IdType> generate(pybind11::array_t<IdType> input, int numTokens, SamplingMode mode, int K = 10, double P = 0.1) {
        assert(input.ndim() == 1); // Only support 1D input for generation
        std::vector<IdType> inputVec(input.size());
        for (ssize_t i = 0; i < input.size(); i++) {
            inputVec[i] = input.data()[i];
        }
        std::vector<IdType> generatedIds = generate(inputVec, numTokens, mode, K, P);
        pybind11::array_t<IdType> output(generatedIds.size());
        for (size_t i = 0; i < generatedIds.size(); i++) {
            output.mutable_data()[i] = generatedIds[i];
        }
        return output;
    }

    /**
     * @brief Execute generate operation.
     */
    Tensor<IdType> generate(Tensor<IdType> input, int numTokens, SamplingMode mode, int K = 10, double P = 0.1) {
        std::vector<IdType> inputVec(input.size());
        cudaMemcpy(inputVec.data(), input.get(), input.size() * sizeof(IdType), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        std::vector<IdType> generatedIds = generate(inputVec, numTokens, mode, K, P);
        Tensor<IdType> output(generatedIds.size());
        cudaMemcpy(output.get(), generatedIds.data(), generatedIds.size() * sizeof(IdType), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        return output;
    }

    /**
     * @brief Execute loss operation.
     */
    Tensor<DType> loss(Tensor<DType> logits, Tensor<IdType> target) {
        return crossEntropyLoss(logits, target, temperature);
    }

    /**
     * @brief Execute backward operation.
     */
    void backward(Tensor<IdType> input, Tensor<DType> logits, Tensor<IdType> target) {
        Tensor<DType> embedding = tokenEmbedding(input) + positionEmbedding(input);
        Tensor<DType> propGrad = crossEntropyLoss.backward(logits, target, temperature);
        propGrad = unEmbedding.backward(preUnembeddingCheckpoint.activationStorage, propGrad);
        propGrad = transformerLayer.backward(embedding, propGrad);
        positionEmbedding.backward(input, propGrad);
        tokenEmbedding.backward(input, propGrad);
    }

    /**
     * @brief Execute zeroGrad operation.
     */
    void zeroGrad() {
        tokenEmbedding.zeroGrad();
        positionEmbedding.zeroGrad();
        transformerLayer.zeroGrad();
        unEmbedding.zeroGrad();
    }

    /**
     * @brief Execute sgdUpdate operation.
     */
    void sgdUpdate(DType lr) {
        tokenEmbedding.sgdUpdate(lr);
        positionEmbedding.sgdUpdate(lr);
        transformerLayer.sgdUpdate(lr);
        unEmbedding.sgdUpdate(lr);
    }

    /**
     * @brief Execute clone operation.
     */
    NanoGPT clone() {
        NanoGPT copy = *this; // Use default copy constructor since we clone all members anyway.
        copy.tokenEmbedding = tokenEmbedding.clone();
        copy.positionEmbedding = positionEmbedding.clone();
        copy.transformerLayer = *std::dynamic_pointer_cast<TransformerLayer<DType>>(transformerLayer.clone());
        copy.unEmbedding = unEmbedding.clone();
        return copy;
    }

    /**
     * @brief Execute getParameters operation.
     */
    std::map<std::string, Tensor<DType>> getParameters() {
        std::map<std::string, Tensor<DType>> params = transformerLayer.getParameters();
        params["tokenEmbedding"] = tokenEmbedding.getParameters();
        params["positionEmbedding"] = positionEmbedding.getParameters();
        params["unEmbedding"] = unEmbedding.getParameters();
        return params;
    }

    /**
     * @brief Execute setParameters operation.
     */
    void setParameters(const std::map<std::string, Tensor<DType>>& params) {
        if (params.count("tokenEmbedding")) {
            tokenEmbedding.setParameters(params.at("tokenEmbedding"));
        }
        if (params.count("positionEmbedding")) {
            positionEmbedding.setParameters(params.at("positionEmbedding"));
        }
        if (params.count("unEmbedding")) {
            unEmbedding.setParameters(params.at("unEmbedding"));
        }
        transformerLayer.setParameters(params);
    }

    /**
     * @brief Execute getGradients operation.
     */
    std::map<std::string, Tensor<DType>> getGradients() {
        std::map<std::string, Tensor<DType>> grads = transformerLayer.getGradients();
        grads["tokenEmbedding"] = tokenEmbedding.getGradients();
        grads["positionEmbedding"] = positionEmbedding.getGradients();
        grads["unEmbedding"] = unEmbedding.getGradients();
        return grads;
    }

    /**
     * @brief Execute train operation.
     */
    void train(std::vector<IdType> input, int batchSize, int numEpochs, DType learningRate) {
        // A huge batch of inputs, all random ranges.
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, input.size() - batchSize - 1);
        std::cout << "Training for " << numEpochs << " epochs with batch size " << batchSize << " and learning rate " << (double)learningRate << std::endl;

        for (size_t epoch = 0; epoch < numEpochs; epoch++) {
            std::vector<IdType> batchInput(batchSize * maxSeqLen());
            std::vector<IdType> batchTarget(batchSize * maxSeqLen());
            for (size_t batch = 0; batch < batchSize; batch++) {
                // Fill batchInput and batchTarget with data from input
                size_t startIdx = dis(gen);
                for (size_t i = 0; i < maxSeqLen(); i++) {
                    batchInput[batch * maxSeqLen() + i] = input[startIdx + i];
                    batchTarget[batch * maxSeqLen() + i] = input[startIdx + i + 1];
                }
            }
            Tensor<IdType> inputTensor = Tensor<IdType>::fromPointer({(size_t)batchSize, (size_t)maxSeqLen()}, batchInput.data());
            Tensor<IdType> targetTensor = Tensor<IdType>::fromPointer({(size_t)batchSize, (size_t)maxSeqLen()}, batchTarget.data());
            Tensor<DType> logits = forward(inputTensor);
            Tensor<DType> loss = crossEntropyLoss(logits, targetTensor, temperature);
            backward(inputTensor, logits, targetTensor);
            sgdUpdate(learningRate);

            double netLoss = (double)0.0;
            std::vector<DType> hostLoss = loss.cpu();
            for (const auto& l : hostLoss) {
                netLoss += (double)l;
            }
            std::cout << "\rEpoch " << epoch + 1 << "/" << numEpochs << ", Loss: " << netLoss << std::flush;
        }
        std::cout << std::endl;
    }

    /**
     * @brief Execute train operation.
     */
    void train(pybind11::array_t<IdType> input, int batchSize, int numEpochs, DType learningRate) {
        std::vector<IdType> inputVec(input.size()); // assume a 1D huge array of tokens.
        for (ssize_t i = 0; i < input.size(); i++) {
            inputVec[i] = input.data()[i];
        }
        train(inputVec, batchSize, numEpochs, learningRate);
    }
};


#endif // NANOGPT