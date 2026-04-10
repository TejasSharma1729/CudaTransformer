#include "TransformerLayer.cu"
#include "Embedding.cu"
#include "SoftmaxLoss.cu"

#ifndef NANOGPT
#define NANOGPT


/** @brief Enum for specifying the sampling mode during inference. */
enum class SamplingMode { Greedy, Random, Nucleus, TopK };
// Not supported: beam search, 


/**
 * @brief NanoGPT model structure (or any LLM for that matter).
 * This encapsulates all components of the model: 
 * token embedding, position embedding, transformer layers, unembedding, and loss.
 * This supports any model shape and configuration by adjusting the constructor parameters.
 * This supports any of the 4 activation types (ReLU, GELU, Sigmoid, Tanh) for the MLPs in the transformer blocks.
 * This also allows modifying the temperature for softmax sampling during inference.
 * 
 * @tparam DType Data type for computations.
 * @tparam IdType Data type for token IDs.
 */
template <typename DType = float, typename IdType = int> struct NanoGPT {
    TokenEmbedding<DType, IdType> tokenEmbedding; /// Token embedding layer
    PositionEmbedding<DType, IdType> positionEmbedding; /// Position embedding layer
    TransformerLayer<DType> transformerLayer; /// Stacked transformer layers
    UnEmbedding<DType> unEmbedding; /// Unembedding layer to project back to vocab size
    CheckpointLayer<DType> preUnembeddingCheckpoint;  /// Checkpoint for unembedding
    Softmax<DType> softmax; /// Softmax layer for output probabilities
    CrossEntropyLoss<DType, IdType> crossEntropyLoss; /// Cross-entropy loss for training

    /**
     * @brief Constructs a NanoGPT model.
     * @param vocabSize Size of the vocabulary.
     * @param maxSeqLen Maximum sequence length.
     * @param embeddingDim Embedding dimension.
     * @param numHeads Number of attention heads.
     * @param headDim Dimension of each attention head.
     * @param mlpDim Dimension of the MLP in transformer blocks.
     * @param numLayers Number of transformer layers.
     * @param checkpointGap Interval for checkpointing activations (default: 0, checkpointing).
     * @param temperature Temperature for softmax sampling (default: 1.0, no scaling).
     * @param activationType Activation function type for transformer layers (default: ReLU).
     */
    NanoGPT(
        int vocabSize,
        int maxSeqLen,
        int embeddingDim,
        int numHeads,
        int headDim,
        int mlpDim,
        int numLayers,
        int checkpointGap = 0,
        double temperature = 1.0,
        ActivationType activationType = ActivationType::ReLU
    ) : tokenEmbedding(vocabSize, embeddingDim),
        positionEmbedding(maxSeqLen, embeddingDim),
        transformerLayer(embeddingDim, numHeads, headDim, mlpDim, numLayers, checkpointGap, activationType),
        unEmbedding(vocabSize, embeddingDim),
        preUnembeddingCheckpoint(), 
        softmax(temperature),
        crossEntropyLoss() { }
    
    int vocabSize() const { return tokenEmbedding.vocabSize; }
    int maxSeqLen() const { return positionEmbedding.blockSize; }
    int embeddingDim() const { return transformerLayer.inputDim; }
    int numHeads() const { return transformerLayer.numHeads; }
    int headDim() const { return transformerLayer.headDim; }
    int mlpDim() const { return transformerLayer.mlpDim; }
    int numLayers() const { return transformerLayer.numLayers; }
    int checkpointGap() const { return transformerLayer.checkpointGap; }
    double getTemperature() const { return (DType)softmax.temperature; }
    void setTemperature(double temp) { softmax.temperature = (DType)temp; }
    ActivationType activationType() const { return transformerLayer.activationType; }

    /**
     * @brief Full forward pass from a Tensor input (batch of sequences of IDs), returns logits.
     * @param input Input tensor of token IDs with shape [seqLen].
     * @return Tensor of logits with shape [seqLen, vocabSize].
     */
    Tensor<DType> forward(Tensor<IdType> input) {
        std::vector<size_t> inputShape = input.shape().toVector();
        input.reshape({input.size() / inputShape.back(), inputShape.back()}); 
        // Ensure shape is [batch, seqLen]    
        Tensor<DType> embeddings = tokenEmbedding(input) + positionEmbedding(input);
        embeddings = transformerLayer(embeddings);
        embeddings = preUnembeddingCheckpoint(embeddings);
        Tensor<DType> preLogits = unEmbedding(embeddings);

        inputShape.push_back((size_t)vocabSize());
        return softmax(preLogits).reshape(inputShape); // Reshape back to [batch, seqLen, vocabSize]
    }

    /**
     * @brief Full forward pass from a numpy array input, returns logits.
     * @param input Input numpy array of token IDs with shape [seqLen].
     * @return Tensor of logits with shape [seqLen, vocabSize].
     */
    Tensor<DType> forward(pybind11::array_t<IdType> input) {
        return forward(Tensor<IdType>::fromNumpy(input));
    }

    /**
     * @brief Full forward pass from a vector input (sequence of IDs), returns logits.
     * @param input Input vector of token IDs with shape [seqLen].
     * @return Tensor of logits with shape [seqLen, vocabSize].
     */
    Tensor<DType> forward(std::vector<IdType> input) {
        Tensor<IdType> inputTensor = Tensor<IdType>::fromPointer({(size_t)input.size()}, input.data());
        Tensor<DType> output = forward(inputTensor);
        int nDim = output.nDim(); // May be more than 1.
        output.reshape({output.shape(nDim - 2), output.shape(nDim - 1)});
        return output;
    }

    /**
     * @brief Forward pass from a Tensor input (batch of sequences of IDs), returns logits.
     * @param input Input tensor of token IDs with shape [seqLen].
     * @return Tensor of logits with shape [seqLen, vocabSize].
     */
    Tensor<DType> operator()(Tensor<IdType> input) {
        return forward(input);
    }

    /**
     * @brief Forward pass from an array input, returns logits.
     * @param input Input numpy array of token IDs with shape [seqLen].
     * @return Tensor of logits with shape
     */
    Tensor<DType> operator()(pybind11::array_t<IdType> input) {
        return forward(input);
    }

    /**
     * @brief Forward pass from a vector input, returns logits.
     * @param input Input vector of token IDs with shape [seqLen].
     * @return Tensor of logits with shape [seqLen, vocabSize].
     */
    Tensor<DType> operator()(std::vector<IdType> input) {
        return forward(input);
    }

    /**
     * @brief Execute predict operation, given the logits and sampling mode, outputs the predicted next token ID.
     * @note The logits should correspond to a 1D sequence, batching is not supported.
     * @param logits Logits tensor from the forward pass with shape [seqLen, vocabSize].
     * @param mode Sampling mode to use for prediction.
     * @param K Number of top tokens to consider for TopK sampling (default: 10).
     * @param P Cumulative probability threshold for Nucleus sampling (default: 0.1).
     * @return Predicted token ID for the next token in the sequence
     */
    IdType predict(
        Tensor<DType> logits,
        SamplingMode mode,
        int K = 10,
        double P = 0.1
    ) {
        assert(logits.nDim() == 2); // (seqLen, vocabSize) -- does not support batched requests.
        int seqLen = logits.shape(0);
        int vocabSize = logits.shape(1);
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
     * @brief Sample the next token ID given an input sequence Tensor and sampling mode.
     * @note The Tensor should be 1D, batched sampling is not supported.
     * @param input Input tensor of token IDs with shape [seqLen].
     * @param mode Sampling mode to use for prediction.
     * @param K Number of top tokens to consider for TopK sampling (default: 10).
     * @param P Cumulative probability threshold for Nucleus sampling (default: 0.1).
     * @return Predicted token ID for the next token in the sequence
     */
    IdType sample(
        Tensor<IdType> input,
        SamplingMode mode,
        int K = 10,
        double P = 0.1
    ) {
        assert(input.nDim() == 1); // Only support 1D input for sampling
        Tensor<DType> logits = forward(input);
        logits.reshape({input.shape(0), (size_t)vocabSize()}); // Ensure logits are in shape [seqLen, vocabSize]
        return predict(logits, mode, K, P);
    }

    /**
     * @brief Sample the next token ID given an input sequence numpy array and sampling mode.
     * @note The input array should be 1D, batched sampling is not supported.
     * @param input Input numpy array of token IDs with shape [seqLen].
     * @param mode Sampling mode to use for prediction
     * @param K Number of top tokens to consider for TopK sampling (default: 10).
     * @param P Cumulative probability threshold for Nucleus sampling (default: 0.1).
     * @return Predicted token ID for the next token in the sequence
     */
    IdType sample(
        pybind11::array_t<IdType> input,
        SamplingMode mode,
        int K = 10,
        double P = 0.1
    ) {
        assert(input.ndim() == 1); // Only support 1D input for sampling
        Tensor<DType> logits = forward(input);
        return predict(logits, mode, K, P);
    }

    /**
     * @brief Sample the next token ID given an input sequence vector (1D) and sampling mode.
     * @param input Input vector of token IDs with shape [seqLen].
     * @param mode Sampling mode to use for prediction
     * @param K Number of top tokens to consider for TopK sampling (default: 10).
     * @param P Cumulative probability threshold for Nucleus sampling (default: 0.1).
     * @return Predicted token ID for the next token in the sequence
     */
    IdType sample(std::vector<IdType> input, SamplingMode mode, int K = 10, double P = 0.1) {
        Tensor<DType> logits = forward(input);
        return predict(logits, mode, K, P);
    }

    /**
     * @brief Generate a sequence of token IDs, appended to the input sequence (1D vector).
     * @note Batched generation and KV-caching are not supported.
     * @param input Input vector of token IDs with shape [seqLen].
     * @param numTokens Number of tokens to generate.
     * @param mode Sampling mode to use for prediction
     * @param K Number of top tokens to consider for TopK sampling (default: 10).
     * @param P Cumulative probability threshold for Nucleus sampling (default: 0.1).
     * @return Vector of token IDs for the generated sequence (including input
     */
    std::vector<IdType> generate(
        std::vector<IdType> input,
        int numTokens,
        SamplingMode mode,
        int K = 10,
        double P = 0.1
    ) {
        std::vector<IdType> generatedIds = input;
        for (int i = 0; i < numTokens; i++) {
            Tensor<DType> logits = forward(generatedIds);
            IdType nextId = predict(logits, mode, K, P);
            generatedIds.push_back(nextId);
        }
        return generatedIds;
    }

    /**
     * @brief Generate a sequence of token IDs, appended to the input sequence (1D numpy array).
     * @note The input array should be 1D, batched generation and KV-caching are not supported.
     * @param input Input numpy array of token IDs with shape [seqLen].
     * @param numTokens Number of tokens to generate.
     * @param mode Sampling mode to use for prediction
     * @param K Number of top tokens to consider for TopK sampling (default: 10).
     * @param P Cumulative probability threshold for Nucleus sampling (default: 0.1).
     * @return Numpy array of token IDs for the generated sequence (including input
     */
    pybind11::array_t<IdType> generate(
        pybind11::array_t<IdType> input,
        int numTokens,
        SamplingMode mode,
        int K = 10,
        double P = 0.1
    ) {
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
     * @brief Generate a sequence of token IDs, appended to the input sequence (1D Tensor of IDs).
     * @note The input Tensor should be 1D, batched generation and KV-caching are not supported.
     * @param input Input numpy array of token IDs with shape [seqLen].
     * @param numTokens Number of tokens to generate.
     * @param mode Sampling mode to use for prediction
     * @param K Number of top tokens to consider for TopK sampling (default: 10).
     * @param P Cumulative probability threshold for Nucleus sampling (default: 0.1).
     * @return Numpy array of token IDs for the generated sequence (including input
     */
    Tensor<IdType> generate(
        Tensor<IdType> input,
        int numTokens,
        SamplingMode mode,
        int K = 10,
        double P = 0.1
    ) {
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
     * @brief Compute the cross-entropy loss given the logits and target token IDs.
     * @param logits Logits tensor from the forward pass with shape [batchSize, seqLen, vocabSize].
     * @param target Target tensor of token IDs with shape [batchSize, seqLen].
     * @return Tensor of loss values with shape [batchSize, seqLen].
     */
    Tensor<DType> loss(Tensor<DType> logits, Tensor<IdType> target) {
        return crossEntropyLoss(logits, target);
    }

    /**
     * @brief Full backward pass given the input token IDs, logits from the forward pass, and target token IDs. 
     * Computes gradients for all parameters in the model.
      * @param input Input tensor of token IDs with shape [batchSize, seqLen].
      * @param logits Logits tensor from the forward pass with shape [batchSize, seqLen, vocabSize].
      * @param target Target tensor of token IDs with shape [batchSize, seqLen].
      * @note The pre-embedding output is checkpointed to save memory during backward pass.
     */
    void backward(Tensor<IdType> input, Tensor<DType> logits, Tensor<IdType> target) {
        std::vector<size_t> inputShape = input.shape().toVector();
        input.reshape({input.size() / inputShape.back(), inputShape.back()});
        inputShape.push_back((size_t)vocabSize());
        logits.reshape(inputShape);

        // Now with proper shapes, get all required tensors.
        Tensor<DType> embedding = tokenEmbedding(input) + positionEmbedding(input);
        Tensor<DType> propGrad = crossEntropyLoss.backward(logits, target);
        Tensor<DType> activation = preUnembeddingCheckpoint.activationStorage; // Get the checkpointed activation for unembedding backward
        Tensor<DType> unembeddings = unEmbedding.forward(activation);
        propGrad = softmax.backward(unembeddings, propGrad);
        propGrad = unEmbedding.backward(activation, propGrad);
        propGrad = transformerLayer.backward(embedding, propGrad);
        positionEmbedding.backward(input, propGrad);
        tokenEmbedding.backward(input, propGrad);

        // Reset shapes back to original for any further operations
        logits.reshape(inputShape);
        inputShape.pop_back();
        input.reshape(inputShape);
    }

    /**
     * @brief Zeros out all the gradients in the model, should be called before a new backward pass.
     */
    void zeroGrad() {
        tokenEmbedding.zeroGrad();
        positionEmbedding.zeroGrad();
        transformerLayer.zeroGrad();
        unEmbedding.zeroGrad();
    }

    /**
     * @brief Update all the parameters with specified learning rate, using SGD algorithm.
     * @param lr Learning rate for the SGD update.
     */
    void sgdUpdate(DType lr) {
        tokenEmbedding.sgdUpdate(lr);
        positionEmbedding.sgdUpdate(lr);
        transformerLayer.sgdUpdate(lr);
        unEmbedding.sgdUpdate(lr);
    }

    /**
     * @brief Create a deep copy of the NanoGPT model, including all parameters and internal states. 
     * This is useful for operations like cloning the model for parallel training or inference.
     * @return A new instance of NanoGPT that is a deep copy of the current model.
     * @note The cloned model will have its own separate memory for parameters and gradients, so updates to one model will not affect the other.
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
     * @brief Get all the parameters of the model as a map from parameter names to their corresponding Tensors. 
     * This can be used for saving the model state, inspecting parameters, 
     * or implementing custom optimization algorithms.
     * @return A map where keys are parameter names to the corresponding Tensors.
     */
    std::map<std::string, Tensor<DType>> getParameters() {
        std::map<std::string, Tensor<DType>> params = transformerLayer.getParameters();
        params["tokenEmbedding"] = tokenEmbedding.getParameters();
        params["positionEmbedding"] = positionEmbedding.getParameters();
        params["unEmbedding"] = unEmbedding.getParameters();
        return params;
    }

    /**
     * @brief Set all the parameters of the model from a map of parameter names to their corresponding Tensors.
     * This can be used for loading a saved model state or initializing parameters.
     * @param params A map where keys are parameter names to the corresponding Tensors.
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
     * @brief Obtain all the gradients of the model parameters as a map from parameter names to corresponding Tensors.
     * This can be used for inspecting gradients, implementing custom optimization algorithms, 
     * or debugging the training process.
     * @return A map where keys are parameter names to the corresponding gradient Tensors.
     */
    std::map<std::string, Tensor<DType>> getGradients() {
        std::map<std::string, Tensor<DType>> grads = transformerLayer.getGradients();
        grads["tokenEmbedding"] = tokenEmbedding.getGradients();
        grads["positionEmbedding"] = positionEmbedding.getGradients();
        grads["unEmbedding"] = unEmbedding.getGradients();
        return grads;
    }

    /**
     * @brief Clear all the stored checkpoint activations in the model to free up memory.
     * This should be called after the backward pass is complete and the gradients have been used for updates.
     */
    void clear() {
        transformerLayer.clear();
    }
};


#endif // NANOGPT