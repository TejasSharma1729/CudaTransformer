#include "NanoGPT.cu"
#include "SGDOptimizer.cu"
#include "AdamOptimizer.cu"
// Above includes auto-include everything; this is the final compilation unit.

namespace py = pybind11;

/**
 * @brief Registers the typed Tensor class and all its methods with the pybind11 module.
 *
 * The Tensor class is the fundamental GPU buffer abstraction.  Every concrete numeric type
 * (float, double, __half, __nv_bfloat16, int, long, short, unsigned char, bool) gets its own
 * Python class whose name is `type_suffix + "Tensor"` (e.g. "FloatTensor", "IntTensor").
 *
 * Copy/assignment semantics are shallow (shared underlying `shared_ptr<DType[]>`).
 * Use `.clone()` to obtain an independent deep copy.
 *
 * @tparam DType      C++ element type of the tensor buffer.
 * @param m           The pybind11 module to register into.
 * @param type_suffix String appended to the class name, e.g. "Float", "Int".
 */
template <typename DType>
void declare_tensor(py::module &m, const std::string &type_suffix) {
    std::string tensor_name = type_suffix + "Tensor";
    py::class_<Tensor<DType>>(m, tensor_name.c_str(),
        "N-dimensional GPU tensor. Assignments and copies are shallow (shared buffer); "
        "call clone() to obtain an independent copy.")
        .def(py::init<>(),
            "Constructs an empty tensor with no allocated memory.")
        .def(py::init<std::vector<size_t>>(), py::arg("shape"),
            "Constructs a zero-initialised tensor with the given shape on the GPU.")
        .def_static("fromNumpy", &Tensor<DType>::fromNumpy, py::arg("array"),
            "Copies a NumPy array to a new GPU tensor. The array must be C-contiguous.")
        .def("shape", [](Tensor<DType> &t) { return t.shape().toVector(); },
            "Returns the tensor shape as a list of ints.")
        .def("clone", &Tensor<DType>::clone,
            "Returns a deep copy of the tensor with its own independent GPU buffer.")
        .def("numpy", &Tensor<DType>::numpy,
            "Copies the tensor to the CPU and returns it as a NumPy array.")
        .def("cpu", &Tensor<DType>::cpu,
            "Copies the tensor to a CPU-side buffer and returns it.")
        .def("item", &Tensor<DType>::item,
            "Returns the scalar value of a single-element tensor.")
        .def("fill", &Tensor<DType>::fill, py::arg("value"),
            "Fills every element of the tensor in-place with the given scalar value.")
        .def("reshape", (Tensor<DType> (Tensor<DType>::*)(const std::vector<size_t>&) const) &Tensor<DType>::reshape, py::arg("shape"),
            "Returns a view of the tensor with a new shape. Total element count must be unchanged.")
        .def("transpose", [](const Tensor<DType> &t, const std::vector<size_t> &perm) { return t.transpose(perm); },
            py::arg("perm"),
            "Returns a tensor with axes permuted according to perm (e.g. [1, 0] to swap the first two axes).")
        .def("transpose", [](const Tensor<DType> &t, size_t a1, size_t a2) { return t.transpose(a1, a2); },
            py::arg("axis1"), py::arg("axis2"),
            "Returns a tensor with axis1 and axis2 swapped.")
        .def("reduce", &Tensor<DType>::reduce,
            "Reduces the tensor with a specified ReductionOp.")
        .def("kron", &Tensor<DType>::kron, py::arg("other"),
            "Returns the Kronecker product of this tensor with other.")
        .def("matmul", &Tensor<DType>::matmul, py::arg("other"),
            "Returns the matrix product of this tensor and other (last two dims).")
        // Type conversions
        .def("float",  &Tensor<DType>::template to<float>,          "Casts to 32-bit float.")
        .def("half",   &Tensor<DType>::template to<__half>,         "Casts to 16-bit float (FP16).")
        .def("bfloat", &Tensor<DType>::template to<__nv_bfloat16>,  "Casts to bfloat16.")
        .def("double", &Tensor<DType>::template to<double>,         "Casts to 64-bit float.")
        .def("int",    &Tensor<DType>::template to<int>,            "Casts to 32-bit int.")
        .def("long",   &Tensor<DType>::template to<long>,           "Casts to 64-bit int.")
        .def("short",  &Tensor<DType>::template to<short>,          "Casts to 16-bit int.")
        .def("byte",   &Tensor<DType>::template to<unsigned char>,  "Casts to unsigned 8-bit int.")
        .def("bool",   &Tensor<DType>::template to<bool>,           "Casts to boolean.")
        // Arithmetic operators
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self <= py::self)
        .def(py::self > py::self)
        .def(py::self >= py::self)
        .def("__add__",      [](const Tensor<DType>& a, float b) { return a + static_cast<DType>(b); }, py::is_operator())
        .def("__add__",      [](const Tensor<DType>& a, const Tensor<DType>& b) { return a + b; }, py::is_operator())
        .def("__radd__",     [](const Tensor<DType>& a, float b) { return a + static_cast<DType>(b); }, py::is_operator())
        .def("__radd__",     [](const Tensor<DType>& a, const Tensor<DType>& b) { return a + b; }, py::is_operator())
        .def("__sub__",      [](const Tensor<DType>& a, float b) { return a - static_cast<DType>(b); }, py::is_operator())
        .def("__sub__",      [](const Tensor<DType>& a, const Tensor<DType>& b) { return a - b; }, py::is_operator())
        .def("__rsub__",     [](const Tensor<DType>& a, float b) { return Tensor<DType>({1}, static_cast<DType>(b)) - a; }, py::is_operator())
        .def("__rsub__",     [](const Tensor<DType>& a, const Tensor<DType>& b) { return b - a; }, py::is_operator())
        .def("__mul__",      [](const Tensor<DType>& a, float b) { return a * static_cast<DType>(b); }, py::is_operator())
        .def("__mul__",      [](const Tensor<DType>& a, const Tensor<DType>& b) { return a * b; }, py::is_operator())
        .def("__rmul__",     [](const Tensor<DType>& a, float b) { return a * static_cast<DType>(b); }, py::is_operator())
        .def("__rmul__",     [](const Tensor<DType>& a, const Tensor<DType>& b) { return a * b; }, py::is_operator())
        .def("__truediv__",  [](const Tensor<DType>& a, float b) { return a / static_cast<DType>(b); }, py::is_operator())
        .def("__truediv__",  [](const Tensor<DType>& a, const Tensor<DType>& b) { return a / b; }, py::is_operator())
        .def("__rtruediv__", [](const Tensor<DType>& a, float b) { return Tensor<DType>({1}, static_cast<DType>(b)) / a; }, py::is_operator())
        .def("__rtruediv__", [](const Tensor<DType>& a, const Tensor<DType>& b) { return a / b; }, py::is_operator())
        .def("__matmul__",   [](const Tensor<DType>& a, const Tensor<DType>& b) { return a.matmul(b); }, py::is_operator())
        .def("__rmatmul__",  [](const Tensor<DType>& a, const Tensor<DType>& b) { return b.matmul(a); }, py::is_operator())
        .def("__mod__",      [](const Tensor<DType>& a, float b) { return a % Tensor<DType>({1}, static_cast<DType>(b)); }, py::is_operator())
        .def("__mod__",      [](const Tensor<DType>& a, const Tensor<DType>& b) { return a % b; }, py::is_operator())
        .def(-py::self)
        // Reductions
        .def("sum",      [](const Tensor<DType> &t, int dim) { return t.sum(dim); },      py::arg("dim") = -1,
            "Sums elements. dim=-1 reduces the whole tensor to a scalar.")
        .def("mean",     [](const Tensor<DType> &t, int dim) { return t.mean(dim); },     py::arg("dim") = -1,
            "Computes the mean. dim=-1 reduces the whole tensor to a scalar.")
        .def("max",      [](const Tensor<DType> &t, int dim) { return t.max(dim); },      py::arg("dim") = -1,
            "Computes the maximum. dim=-1 reduces the whole tensor to a scalar.")
        .def("min",      [](const Tensor<DType> &t, int dim) { return t.min(dim); },      py::arg("dim") = -1,
            "Computes the minimum. dim=-1 reduces the whole tensor to a scalar.")
        .def("manhattan", [](const Tensor<DType> &t, int dim) { return t.norm_l1(dim); }, py::arg("dim") = -1,
            "Computes the L1 (Manhattan) norm. Alias for norm_l1.")
        .def("norm_l1",  [](const Tensor<DType> &t, int dim) { return t.norm_l1(dim); },  py::arg("dim") = -1,
            "Computes the L1 norm (sum of absolute values).")
        .def("norm",     [](const Tensor<DType> &t, int dim) { return t.norm_l2(dim); },  py::arg("dim") = -1,
            "Computes the L2 (Euclidean) norm. Alias for norm_l2.")
        .def("norm_l2",  [](const Tensor<DType> &t, int dim) { return t.norm_l2(dim); },  py::arg("dim") = -1,
            "Computes the L2 norm (square root of sum of squares).");
}

/**
 * @brief Registers all layer classes, their concrete sub-classes, and layer factory functions
 * with the pybind11 module under the given type suffix.
 *
 * Registered classes (each prefixed with `type_suffix`):
 *   - Module         — abstract base; `forward`, `backward`, `sgdUpdate`, `zeroGrad`, get/setParameters, get/setGradients
 *   - LayerNormLayer — layer normalization with learnable scale and shift
 *   - ActivationLayer — stateless activation (ReLU, GELU, Sigmoid, Tanh)
 *   - LinearLayer    — fully-connected layer (output = input @ W.T + b)
 *   - AttentionLayer — multi-head attention (QKV projection + flash attention + output projection)
 *   - SGDOptimizer   — explicit map-based SGD (alternative to layer.sgdUpdate)
 *
 * Factory functions (each suffixed with `type_suffix`):
 *   - LayerNorm(input_dim, epsilon)
 *   - Linear(input_dim, output_dim)
 *   - Activation(input_dim, activation_type)
 *   - MLP(layers)
 *   - Attention(input_dim, num_heads, head_dim)
 *   - TransformerBlock(input_dim, num_heads, head_dim, mlp_dim [, activation_type])
 *   - Transformer(input_dim, num_heads, head_dim, mlp_dim, num_layers [, checkpoint_gap, activation_type])
 *
 * @tparam DType      C++ element type used by all registered layers and tensors.
 * @param m           The pybind11 module to register into.
 * @param type_suffix String appended to class and factory names, e.g. "Float", "BFloat".
 */
template <typename DType>
void declare_modules(py::module &m, const std::string &type_suffix) {
    // ------------------------------------------------------------------
    // Base Layer class — polymorphic handle visible from Python.
    // ------------------------------------------------------------------
    std::string module_name = type_suffix + "Module";
    py::class_<Layer<DType>, std::shared_ptr<Layer<DType>>>(m, module_name.c_str(),
        "Abstract base class for all layers. Concrete sub-classes are registered separately "
        "so Python sees the full derived interface when a factory function returns one.")
        .def("forward", &Layer<DType>::forward, py::arg("input"),
            "Runs the forward pass and returns the output tensor.")
        .def("backward", &Layer<DType>::backward, py::arg("input"), py::arg("grad_output"),
            "Runs the backward pass given the original input and the upstream gradient. "
            "Accumulates parameter gradients internally and returns the input gradient.")
        .def("getParameters", &Layer<DType>::getParameters,
            "Returns a dictionary mapping parameter names to their Tensors.")
        .def("setParameters", &Layer<DType>::setParameters, py::arg("params"),
            "Sets parameter tensors from a dictionary. Unmatched keys are ignored.")
        .def("getGradients", &Layer<DType>::getGradients,
            "Returns a dictionary mapping parameter names to their gradient Tensors.")
        .def("zeroGrad", &Layer<DType>::zeroGrad,
            "Resets all parameter gradients to zero. For container layers, this propagates recursively to all sub-layers.")
        .def("sgdUpdate", &Layer<DType>::sgdUpdate, py::arg("lr"),
            "Applies a single SGD step in-place: param -= lr * grad. "
            "For container layers (MLP, TransformerBlock, Transformer) this propagates "
            "recursively to all sub-layers. Stateless layers (Activation) are a no-op.")
        .def("__call__", &Layer<DType>::operator(), py::arg("input"),
            "Shorthand for forward(input).");
    
    // ------------------------------------------------------------------
    // LayerNormLayer — per-token layer normalization with learnable gain/shift.
    // ------------------------------------------------------------------
    py::class_<LayerNormLayer<DType>, Layer<DType>, std::shared_ptr<LayerNormLayer<DType>>>(
            m, (type_suffix + "LayerNormLayer").c_str(),
            "Layer Normalization Layer.")
        .def("forward", &LayerNormLayer<DType>::forward, py::arg("input"))
        .def("backward", &LayerNormLayer<DType>::backward, py::arg("input"), py::arg("grad_output"))
        .def("zeroGrad", &LayerNormLayer<DType>::zeroGrad)
        .def("sgdUpdate", &LayerNormLayer<DType>::sgdUpdate, py::arg("lr"))
        .def("__call__", &LayerNormLayer<DType>::operator(), py::arg("input"));

        m.def(("LayerNorm" + type_suffix).c_str(), [](int inputDim, DType epsilon) {
        return std::make_shared<LayerNormLayer<DType>>(inputDim, epsilon);
    }, py::arg("input_dim"), py::arg("epsilon") = 1e-5,
    "Creates a LayerNormLayer.");

    // ------------------------------------------------------------------
    // ActivationLayer — dispatches ReLU / GELU / Sigmoid / Tanh by enum.
    // ------------------------------------------------------------------
    py::class_<ActivationLayer<DType>, Layer<DType>, std::shared_ptr<ActivationLayer<DType>>>(
            m, (type_suffix + "ActivationLayer").c_str(),
            "Stateless activation layer. Supports ReLU, GELU, Sigmoid, and Tanh. "
            "No learnable parameters; sgdUpdate is a no-op.")
        .def("forward",  &ActivationLayer<DType>::forward,  py::arg("input"),
            "Applies the activation element-wise and returns the output tensor.")
        .def("backward", &ActivationLayer<DType>::backward, py::arg("input"), py::arg("grad_output"),
            "Computes the element-wise activation gradient and returns the input gradient.")
        .def("__call__", &ActivationLayer<DType>::operator(), py::arg("input"),
            "Shorthand for forward(input).");

    // ------------------------------------------------------------------
    // LinearLayer — fully connected layer with weight/bias get & set.
    // ------------------------------------------------------------------
    py::class_<LinearLayer<DType>, Layer<DType>, std::shared_ptr<LinearLayer<DType>>>(
            m, (type_suffix + "LinearLayer").c_str(),
            "Fully connected linear layer: output = input @ weights.T + biases.\n"
            "Weights shape: [outputDim, inputDim]. Bias shape: [outputDim].\n"
            "All get/set methods perform shallow copies (shared GPU buffer, no data movement).")
        .def("forward",       &LinearLayer<DType>::forward,      py::arg("input"),
            "Computes output = input @ weights.T + biases.")
        .def("backward",      &LinearLayer<DType>::backward,     py::arg("input"), py::arg("grad_output"),
            "Accumulates weight/bias gradients and returns the input gradient.")
        .def("sgdUpdate",     &LinearLayer<DType>::sgdUpdate,    py::arg("lr"),
            "Applies SGD in-place: weights -= lr * weight_grad, biases -= lr * bias_grad.")
        .def("getWeights",    &LinearLayer<DType>::getWeights,
            "Returns a Tensor sharing the weight buffer [outputDim, inputDim]. No copy.")
        .def("setWeights",    &LinearLayer<DType>::setWeights,   py::arg("tensor"),
            "Adopts tensor's GPU buffer as this layer's weights. No copy.")
        .def("getBiases",     &LinearLayer<DType>::getBiases,
            "Returns a Tensor sharing the bias buffer [outputDim]. No copy.")
        .def("setBiases",     &LinearLayer<DType>::setBiases,    py::arg("tensor"),
            "Adopts tensor's GPU buffer as this layer's biases. No copy.")
        .def("getWeightGrad", &LinearLayer<DType>::getWeightGrad,
            "Returns a Tensor sharing the weight-gradient buffer (lazy-allocated if needed). No copy.")
        .def("setWeightGrad", &LinearLayer<DType>::setWeightGrad, py::arg("tensor"),
            "Adopts tensor's GPU buffer as this layer's weight gradient. No copy.")
        .def("getBiasGrad",   &LinearLayer<DType>::getBiasGrad,
            "Returns a Tensor sharing the bias-gradient buffer (lazy-allocated if needed). No copy.")
        .def("setBiasGrad",   &LinearLayer<DType>::setBiasGrad,  py::arg("tensor"),
            "Adopts tensor's GPU buffer as this layer's bias gradient. No copy.")
        .def("__call__",      &LinearLayer<DType>::operator(),   py::arg("input"),
            "Shorthand for forward(input).");

    // ------------------------------------------------------------------
    // CheckpointLayer — gradient checkpointing layer.
    // ------------------------------------------------------------------
    py::class_<CheckpointLayer<DType>, Layer<DType>, std::shared_ptr<CheckpointLayer<DType>>>(
            m, (type_suffix + "CheckpointLayer").c_str(),
            "Gradient Checkpointing Layer. Stored forward activations during backward pass to save time.")
        .def("forward", &CheckpointLayer<DType>::forward, py::arg("input"),
            "Runs the Checkpointed forward pass.")
        .def("backward", &CheckpointLayer<DType>::backward, py::arg("input"), py::arg("grad_output"),
            "Runs the Checkpointed backward pass.")
        .def("__call__", &CheckpointLayer<DType>::operator(), py::arg("input"),
            "Shorthand for forward(input).")
        .def("clear", &CheckpointLayer<DType>::clear,
            "Clears any stored checkpoint activations to free up memory. Does not modify parameters or gradients.");

    // ------------------------------------------------------------------
    // AttentionLayer — multi-head attention with QKV + output projection.
    // ------------------------------------------------------------------
    py::class_<AttentionLayer<DType>, Layer<DType>, std::shared_ptr<AttentionLayer<DType>>>(
            m, (type_suffix + "AttentionLayer").c_str(),
            "Multi-head attention layer composed of QKV projection, flash attention, and output projection.\n"
            "All get/set methods perform shallow copies (shared GPU buffer, no data movement).\n"
            "QKV weight shape: [inputDim, headDim*numHeads]. Output weight shape: [headDim*numHeads, inputDim].")
        .def("forward",          static_cast<Tensor<DType> (AttentionLayer<DType>::*)(Tensor<DType>)>(&AttentionLayer<DType>::forward),         py::arg("input"),
            "Runs QKV projection -> flash attention -> output projection and returns [batch, seq, inputDim].")
        .def("forward_caching",          [](AttentionLayer<DType>& layer, Tensor<DType> input) {
                Tensor<DType>* states = new Tensor<DType>[4];
                Tensor<DType> out = layer.forward(input, states);
                py::list py_states;
                for(int i = 0; i < 4; i++) py_states.append(states[i]);
                delete[] states;
                return py::make_tuple(out, py_states);
            }, py::arg("input"),
            "Runs QKV projection -> flash attention -> output projection and returns a tuple: (output, [Q, K, V, AttnOut]).")
        .def("backward",         static_cast<Tensor<DType> (AttentionLayer<DType>::*)(Tensor<DType>, Tensor<DType>)>(&AttentionLayer<DType>::backward),        py::arg("input"), py::arg("grad_output"),
            "Backpropagates through all three sub-components, accumulates all parameter gradients, "
            "and returns the input gradient. Automatically caches forwards state internally if none provided.")
        .def("backward_caching", [](AttentionLayer<DType>& layer, Tensor<DType> input, Tensor<DType> gradOutput, py::list py_states) {
                if(py_states.size() != 4) throw std::runtime_error("States list must have exactly 4 Tensors");
                Tensor<DType>* states = new Tensor<DType>[4];
                for(int i = 0; i < 4; i++) states[i] = py_states[i].cast<Tensor<DType>>();
                Tensor<DType> outGrad = layer.backward(input, gradOutput, states);
                delete[] states;
                return outGrad;
            }, py::arg("input"), py::arg("grad_output"), py::arg("states"),
            "Backpropagates using provided standard cached forward states [Q, K, V, AttnOut].")
        .def("sgdUpdate",        &AttentionLayer<DType>::sgdUpdate,       py::arg("lr"),
            "Applies SGD in-place to all 8 parameter buffers (Q/K/V weights+biases, output weight+bias).")
        // Query projection
        .def("getQueryWeights",  &AttentionLayer<DType>::getQueryWeights,
            "Returns a Tensor sharing the Query weight buffer [inputDim, headDim*numHeads]. No copy.")
        .def("setQueryWeights",  &AttentionLayer<DType>::setQueryWeights, py::arg("tensor"),
            "Adopts tensor's GPU buffer as the Query projection weights. No copy.")
        .def("getQueryBiases",   &AttentionLayer<DType>::getQueryBiases,
            "Returns a Tensor sharing the Query bias buffer [headDim*numHeads]. No copy.")
        .def("setQueryBiases",   &AttentionLayer<DType>::setQueryBiases,  py::arg("tensor"),
            "Adopts tensor's GPU buffer as the Query projection biases. No copy.")
        // Key projection
        .def("getKeyWeights",    &AttentionLayer<DType>::getKeyWeights,
            "Returns a Tensor sharing the Key weight buffer [inputDim, headDim*numHeads]. No copy.")
        .def("setKeyWeights",    &AttentionLayer<DType>::setKeyWeights,   py::arg("tensor"),
            "Adopts tensor's GPU buffer as the Key projection weights. No copy.")
        .def("getKeyBiases",     &AttentionLayer<DType>::getKeyBiases,
            "Returns a Tensor sharing the Key bias buffer [headDim*numHeads]. No copy.")
        .def("setKeyBiases",     &AttentionLayer<DType>::setKeyBiases,    py::arg("tensor"),
            "Adopts tensor's GPU buffer as the Key projection biases. No copy.")
        // Value projection
        .def("getValueWeights",  &AttentionLayer<DType>::getValueWeights,
            "Returns a Tensor sharing the Value weight buffer [inputDim, headDim*numHeads]. No copy.")
        .def("setValueWeights",  &AttentionLayer<DType>::setValueWeights, py::arg("tensor"),
            "Adopts tensor's GPU buffer as the Value projection weights. No copy.")
        .def("getValueBiases",   &AttentionLayer<DType>::getValueBiases,
            "Returns a Tensor sharing the Value bias buffer [headDim*numHeads]. No copy.")
        .def("setValueBiases",   &AttentionLayer<DType>::setValueBiases,  py::arg("tensor"),
            "Adopts tensor's GPU buffer as the Value projection biases. No copy.")
        // Output projection
        .def("getOutputWeights", &AttentionLayer<DType>::getOutputWeights,
            "Returns a Tensor sharing the output projection weight buffer [headDim*numHeads, inputDim]. No copy.")
        .def("setOutputWeights", &AttentionLayer<DType>::setOutputWeights, py::arg("tensor"),
            "Adopts tensor's GPU buffer as the output projection weights. No copy.")
        .def("getOutputBiases",  &AttentionLayer<DType>::getOutputBiases,
            "Returns a Tensor sharing the output projection bias buffer [inputDim]. No copy.")
        .def("setOutputBiases",  &AttentionLayer<DType>::setOutputBiases,  py::arg("tensor"),
            "Adopts tensor's GPU buffer as the output projection biases. No copy.")
        .def("__call__",         &AttentionLayer<DType>::operator(),        py::arg("input"),
            "Shorthand for forward(input).");

    // ------------------------------------------------------------------
    // MLPLayer -- sequential container of sub-layers (e.g. Linear -> Activation -> Linear).
    // ------------------------------------------------------------------
    py::class_<MLPLayer<DType>, Layer<DType>, std::shared_ptr<MLPLayer<DType>>>(
            m, (type_suffix + "MLPLayer").c_str(),
            "Sequential container for a list of sub-layers.")
        .def("forward", &MLPLayer<DType>::forward, py::arg("input"),
            "Runs the sequential forward pass.")
        .def("backward", &MLPLayer<DType>::backward, py::arg("input"), py::arg("grad_output"),
            "Runs the sequential backward pass.")
        .def("__call__", &MLPLayer<DType>::operator(), py::arg("input"),
            "Shorthand for forward(input).")
        .def("clear", &MLPLayer<DType>::clear,
            "Remove all checkpoint activations, if present. Does not modify parameters or gradients.");

    // ------------------------------------------------------------------
    // TransformerBlockLayer — single Transformer block: LayerNorm -> Attention -> LayerNorm -> MLP.
    // ------------------------------------------------------------------
    py::class_<TransformerBlockLayer<DType>, Layer<DType>, std::shared_ptr<TransformerBlockLayer<DType>>>(
            m, (type_suffix + "TransformerBlockLayer").c_str(),
            "TransformerBlock layer: LayerNorm -> Attention -> LayerNorm -> MLP.")
        .def("forward", &TransformerBlockLayer<DType>::forward, py::arg("input"),
            "Runs the Transformer Block forward pass.")
        .def("backward", &TransformerBlockLayer<DType>::backward, py::arg("input"), py::arg("grad_output"),
            "Runs the Transformer Block backward pass.")
        .def("__call__", &TransformerBlockLayer<DType>::operator(), py::arg("input"),
            "Shorthand for forward(input).");

    // ------------------------------------------------------------------
    // TransformerLayer — stack of multiple Transformer blocks with optional checkpointing.
    // ------------------------------------------------------------------
    py::class_<TransformerLayer<DType>, Layer<DType>, std::shared_ptr<TransformerLayer<DType>>>(
            m, (type_suffix + "TransformerLayer").c_str(),
            "Transformer layer containing multiple Transformer blocks.")
        .def("forward", &TransformerLayer<DType>::forward, py::arg("input"),
            "Runs the multi-block Transformer forward pass.")
        .def("backward", &TransformerLayer<DType>::backward, py::arg("input"), py::arg("grad_output"),
            "Runs the multi-block Transformer backward pass.")
        .def("__call__", &TransformerLayer<DType>::operator(), py::arg("input"),
            "Shorthand for forward(input).")
        .def("clear", &TransformerLayer<DType>::clear,
            "Remove all checkpoint activations, if present. Does not modify parameters or gradients.");

    // ------------------------------------------------------------------
    // State-based optimizers (SGD, Adam, AdamW).
    // ------------------------------------------------------------------
    py::class_<SGDOptimizer<DType>>(m, (type_suffix + "SGDOptimizer").c_str(),
        "Explicit SGD optimizer that operates on parameter/gradient pointer maps.")
        .def(py::init<std::map<std::string, Tensor<DType>>, DType>(), 
            py::arg("params"), py::arg("lr"),
            "Constructs an SGDOptimizer with the given parameters and learning rate.")
        .def("step", &SGDOptimizer<DType>::step, py::arg("grads"),
            "Applies one SGD step: for each matching key, params[k] -= lr * grads[k].")
        .def("setLearningRate", &SGDOptimizer<DType>::setLearningRate, py::arg("lr"),
            "Updates the learning rate.")
        .def("getLearningRate", &SGDOptimizer<DType>::getLearningRate,
            "Returns the current learning rate.");

    py::class_<AdamOptimizer<DType>>(m, (type_suffix + "AdamOptimizer").c_str(),
        "Explicit Adam optimizer that operates on parameter/gradient pointer maps.")
        .def(py::init<std::map<std::string, Tensor<DType>>, double, double, double, double, double>(), 
            py::arg("params"), py::arg("lr") = 0.001, py::arg("b1") = 0.9, py::arg("b2") = 0.999, py::arg("eps") = 1e-8, py::arg("decay") = 0.0,
            "Constructs an Adam optimizer with the given parameters and hyperparameters.")
        .def("step", &AdamOptimizer<DType>::step, py::arg("grads"),
            "Applies one Adam step to all parameters using the provided gradients map.")
        .def("setLearningRate", &AdamOptimizer<DType>::setLearningRate, py::arg("lr"),
            "Updates the learning rate.")
        .def("getLearningRate", &AdamOptimizer<DType>::getLearningRate,
            "Returns the current learning rate.");

    py::class_<AdamWOptimizer<DType>>(m, (type_suffix + "AdamWOptimizer").c_str(),
        "Explicit AdamW optimizer that operates on parameter/gradient pointer maps.")
        .def(py::init<std::map<std::string, Tensor<DType>>, double, double, double, double, double>(), 
            py::arg("params"), py::arg("lr") = 0.001, py::arg("b1") = 0.9, py::arg("b2") = 0.999, py::arg("eps") = 1e-8, py::arg("decay") = 0.0,
            "Constructs an AdamW optimizer with the given parameters and hyperparameters.")
        .def("step", &AdamWOptimizer<DType>::step, py::arg("grads"),
            "Applies one AdamW step to all parameters using the provided gradients map.")
        .def("setLearningRate", &AdamWOptimizer<DType>::setLearningRate, py::arg("lr"),
            "Updates the learning rate.")
        .def("getLearningRate", &AdamWOptimizer<DType>::getLearningRate,
            "Returns the current learning rate.");

    // ------------------------------------------------------------------
    // Factory functions (return base Module, resolved to concrete type by pybind11).
    // ------------------------------------------------------------------
    m.def(("LayerNorm" + type_suffix).c_str(), &LayerNorm<DType>,
        py::arg("input_dim"), py::arg("epsilon") = 1e-5,
        "Creates a LayerNormLayer with the given input dimension and epsilon for numerical stability.");
    m.def(("Linear" + type_suffix).c_str(),           &Linear<DType>,
        py::arg("input_dim"), py::arg("output_dim"),
        "Creates a LinearLayer: output = input @ weights.T + biases.");
    m.def(("Activation" + type_suffix).c_str(),       &Activation<DType>,
        py::arg("input_dim"), py::arg("activation_type"),
        "Creates an ActivationLayer with the given ActivationType (ReLU, GELU, Sigmoid, Tanh).");
    m.def(("Sigmoid" + type_suffix).c_str(),          &SigmoidActivation<DType>,
        py::arg("input_dim"),
        "Creates a Sigmoid activation layer.");
    m.def(("Tanh" + type_suffix).c_str(),             &TanhActivation<DType>,
        py::arg("input_dim"),
        "Creates a Tanh activation layer.");
    m.def(("ReLU" + type_suffix).c_str(),             &ReLUActivation<DType>,
        py::arg("input_dim"),
        "Creates a ReLU activation layer.");
    m.def(("GELU" + type_suffix).c_str(),            &GELUActivation<DType>,
        py::arg("input_dim"),
        "Creates a GELU activation layer.");
    m.def(("Checkpoint" + type_suffix).c_str(),        &Checkpoint<DType>,
        "Layer for gradient checkpointing (storing activations), does nothing.");
    m.def(("MLP" + type_suffix).c_str(),              &MLP<DType>,
        py::arg("layers"),
        "Creates an MLPLayer from a list of sub-modules executed sequentially.");
    m.def(("Attention" + type_suffix).c_str(),        &Attention<DType>,
        py::arg("input_dim"), py::arg("num_heads"), py::arg("head_dim"),
        "Creates a multi-head AttentionLayer (QKV projection + flash attention + output projection).");
    m.def(("TransformerBlock" + type_suffix).c_str(), &TransformerBlock<DType>,
        py::arg("input_dim"), py::arg("num_heads"), py::arg("head_dim"),
        py::arg("mlp_dim"), py::arg("activation_type") = ActivationType::ReLU,
        "Creates a single Transformer block (Attention followed by MLP).");
    m.def(("Transformer" + type_suffix).c_str(),      &Transformer<DType>,
        py::arg("input_dim"), py::arg("num_heads"), py::arg("head_dim"),
        py::arg("mlp_dim"), py::arg("num_layers"),
        py::arg("checkpoint_gap") = 0, py::arg("activation_type") = ActivationType::ReLU,
        "Creates a full Transformer model with num_layers blocks. "
        "checkpoint_gap > 0 inserts gradient checkpoints every N layers to reduce memory.");
}

/**
 * @brief Convenience wrapper that calls declare_tensor followed by declare_modules.
 *
 * Registers both the typed Tensor class and all layer / optimizer types for a single
 * numeric type in one shot.  Called during PYBIND11_MODULE for each floating-point
 * precision variant (Float, Double, Half, BFloat).
 *
 * @tparam DType      C++ element type.
 * @param m           The pybind11 module to register into.
 * @param type_suffix String appended to class and function names.
 */
template <typename DType>
void declare_tensor_and_modules(py::module &m, const std::string &type_suffix) {
    declare_tensor<DType>(m, type_suffix);
    declare_modules<DType>(m, type_suffix);
}


/**
 * @brief Registers the NanoGPT model and its component classes (embeddings, loss functions,
 * softmax) with the pybind11 module.
 *
 * Registered classes (prefixed with `type_suffix`):
 *   - TokenEmbedding    — maps integer token IDs to dense embedding vectors
 *   - PositionEmbedding — adds learnable positional embeddings to token embeddings
 *   - UnEmbedding       — projects transformer output back to vocabulary logits
 *   - CrossEntropyLoss  — -log(p[target]) loss; expects softmax probabilities as input
 *   - MSELoss           — (1/D)*sum((pred-target)^2) loss for regression tasks
 *   - Softmax           — numerically stable softmax with temperature scaling
 *   - NanoGPT           — full language model: token+position embedding → transformer → unembedding → softmax
 *
 * Also exposes the SamplingMode enum (Greedy, Random, Nucleus, TopK) for use with NanoGPT.predict / sample / generate.
 *
 * @tparam DType   C++ floating-point type for weights and activations (float, __half, etc.).
 * @tparam IdType  C++ integer type for token IDs (typically int).
 * @param m           The pybind11 module to register into.
 * @param type_suffix String prepended to each class name, e.g. "Float", "BFloat".
 */
template <typename DType = float, typename IdType = int>
void declare_nanogpt(py::module &m, const std::string &type_suffix) {
    // ------------------------------------------------------------------
    // TokenEmbedding — converts token ID sequences to dense embedding tensors.
    // ------------------------------------------------------------------
    py::class_<TokenEmbedding<DType, IdType>>(m, (type_suffix + "TokenEmbedding").c_str(),
        "Learnable token embedding table of shape [vocab_size, embedding_dim].\n"
        "forward(input [batchSize, seqLen]) → [batchSize, seqLen, embedding_dim].\n"
        "Gradients are accumulated via atomic scatter in the backward pass.")
        .def(py::init<int, int>(), py::arg("vocab_size"), py::arg("embedding_dim"),
            "Constructs a zero-initialised token embedding table.")
        .def("forward", &TokenEmbedding<DType, IdType>::forward, py::arg("input"),
            "Gathers embedding vectors for each token ID. "
            "input: [batchSize, seqLen] of IdType. Returns [batchSize, seqLen, embedding_dim].")
        .def("__call__", &TokenEmbedding<DType, IdType>::operator(), py::arg("input"),
            "Shorthand for forward(input).")
        .def("backward", &TokenEmbedding<DType, IdType>::backward,
            py::arg("input"), py::arg("grad_output"),
            "Scatter-accumulates upstream gradients into the embedding gradient table.")
        .def("getParameters", &TokenEmbedding<DType, IdType>::getParameters,
            "Returns a Tensor view of the embedding matrix [vocab_size, embedding_dim]. No copy.")
        .def("setParameters", &TokenEmbedding<DType, IdType>::setParameters, py::arg("params"),
            "Adopts the buffer of the provided Tensor as this layer's embedding matrix. No copy.")
        .def("getGradients", &TokenEmbedding<DType, IdType>::getGradients,
            "Returns a Tensor view of the accumulated embedding gradient [vocab_size, embedding_dim]. No copy.")
        .def("zeroGrad", &TokenEmbedding<DType, IdType>::zeroGrad,
            "Resets the gradient accumulator to zero. Call before each backward pass.")
        .def("sgdUpdate", &TokenEmbedding<DType, IdType>::sgdUpdate, py::arg("lr"),
            "In-place SGD step: embeddingMatrix -= lr * gradEmbeddingMatrix.");

    // ------------------------------------------------------------------
    // PositionEmbedding — adds a learned positional encoding to token embeddings.
    // ------------------------------------------------------------------
    py::class_<PositionEmbedding<DType, IdType>>(m, (type_suffix + "PositionEmbedding").c_str(),
        "Learnable positional embedding table of shape [max_seq_len, embedding_dim].\n"
        "forward(input [batchSize, seqLen]) → [batchSize, seqLen, embedding_dim].\n"
        "The token IDs in input are ignored; only the sequence positions matter.")
        .def(py::init<int, int>(), py::arg("max_seq_len"), py::arg("embedding_dim"),
            "Constructs a zero-initialised positional embedding table.")
        .def("forward", &PositionEmbedding<DType, IdType>::forward, py::arg("input"),
            "Broadcasts positional embeddings over the batch dimension. "
            "input shape: [batchSize, seqLen]. Returns [batchSize, seqLen, embedding_dim].")
        .def("__call__", &PositionEmbedding<DType, IdType>::operator(), py::arg("input"),
            "Shorthand for forward(input).")
        .def("backward", &PositionEmbedding<DType, IdType>::backward,
            py::arg("input"), py::arg("grad_output"),
            "Accumulates upstream gradients into the positional embedding gradient table.")
        .def("getParameters", &PositionEmbedding<DType, IdType>::getParameters,
            "Returns a Tensor view of the positional embedding matrix [max_seq_len, embedding_dim]. No copy.")
        .def("setParameters", &PositionEmbedding<DType, IdType>::setParameters, py::arg("params"),
            "Adopts the buffer of the provided Tensor as this layer's positional embedding matrix. No copy.")
        .def("getGradients", &PositionEmbedding<DType, IdType>::getGradients,
            "Returns a Tensor view of the accumulated positional gradient [max_seq_len, embedding_dim]. No copy.")
        .def("zeroGrad", &PositionEmbedding<DType, IdType>::zeroGrad,
            "Resets the gradient accumulator to zero. Call before each backward pass.")
        .def("sgdUpdate", &PositionEmbedding<DType, IdType>::sgdUpdate, py::arg("lr"),
            "In-place SGD step: embeddingMatrix -= lr * gradEmbeddingMatrix.");

    // ------------------------------------------------------------------
    // UnEmbedding — linear projection from embedding space to vocabulary logits.
    // ------------------------------------------------------------------
    py::class_<UnEmbedding<DType>>(m, (type_suffix + "UnEmbedding").c_str(),
        "Learnable unembedding matrix of shape [vocab_size, embedding_dim].\n"
        "Performs output = input @ matrix.T (no bias); apply Softmax afterwards for probabilities.\n"
        "forward([batchSize, seqLen, embedding_dim]) → [batchSize, seqLen, vocab_size].")
        .def(py::init<int, int>(), py::arg("vocab_size"), py::arg("embedding_dim"),
            "Constructs a zero-initialised unembedding matrix.")
        .def("forward", &UnEmbedding<DType>::forward, py::arg("input"),
            "Projects transformer output to vocabulary logits via matrix multiplication. "
            "input: [batchSize, seqLen, embedding_dim]. Returns [batchSize, seqLen, vocab_size].")
        .def("__call__", &UnEmbedding<DType>::operator(), py::arg("input"),
            "Shorthand for forward(input).")
        .def("backward", &UnEmbedding<DType>::backward, py::arg("input"), py::arg("grad_output"),
            "Accumulates weight gradients and returns the input gradient. "
            "grad_output: [batchSize, seqLen, vocab_size]. Returns [batchSize, seqLen, embedding_dim].")
        .def("getParameters", &UnEmbedding<DType>::getParameters,
            "Returns a Tensor view of the unembedding matrix [vocab_size, embedding_dim]. No copy.")
        .def("setParameters", &UnEmbedding<DType>::setParameters, py::arg("params"),
            "Adopts the buffer of the provided Tensor as this layer's unembedding matrix. No copy.")
        .def("getGradients", &UnEmbedding<DType>::getGradients,
            "Returns a Tensor view of the accumulated weight gradient [vocab_size, embedding_dim]. No copy.")
        .def("zeroGrad", &UnEmbedding<DType>::zeroGrad,
            "Resets the gradient accumulator to zero. Call before each backward pass.")
        .def("sgdUpdate", &UnEmbedding<DType>::sgdUpdate, py::arg("lr"),
            "In-place SGD step: embeddingMatrix -= lr * gradEmbeddingMatrix.");

    // ------------------------------------------------------------------
    // CrossEntropyLoss — -log(p[target]) loss over softmax probability tensors.
    // ------------------------------------------------------------------
    py::class_<CrossEntropyLoss<DType, IdType>>(m, (type_suffix + "CrossEntropyLoss").c_str(),
        "Cross-entropy loss between a probability distribution and integer class labels.\n"
        "Expects input to be softmax probabilities (NOT raw logits).\n"
        "forward(probs [N, C], targets [N]) → loss [N]  where loss[i] = -log(probs[i, target[i]]).\n"
        "backward returns dL/dprobs, which is then passed back through Softmax.backward.")
        .def(py::init<>(), "Constructs a CrossEntropyLoss (no parameters).")
        .def("forward", &CrossEntropyLoss<DType, IdType>::forward,
            py::arg("input"), py::arg("target"),
            "Computes per-row cross-entropy: loss[i] = -log(input[i, target[i]]). "
            "input must contain softmax probabilities.")
        .def("__call__", &CrossEntropyLoss<DType, IdType>::operator(),
            py::arg("input"), py::arg("target"),
            "Shorthand for forward(input, target).")
        .def("backward", &CrossEntropyLoss<DType, IdType>::backward,
            py::arg("input"), py::arg("target"),
            "Returns gradient w.r.t. the probability vector: grad[i, target[i]] = -1/p, else 0.");

    // ------------------------------------------------------------------
    // MSELoss — mean squared error loss for regression tasks.
    // ------------------------------------------------------------------
    py::class_<MSELoss<DType>>(m, (type_suffix + "MSELoss").c_str(),
        "Mean Squared Error loss: loss[i] = (1/D) * sum_j (input[i][j] - target[i][j])^2.\n"
        "Has no learnable parameters.  Suitable for regression tasks.")
        .def(py::init<>(), "Constructs an MSELoss (no parameters).")
        .def("forward", &MSELoss<DType>::forward,
             py::arg("input"), py::arg("target"),
             "Computes per-row MSE loss. input and target must have the same shape.")
        .def("__call__", &MSELoss<DType>::operator(),
             py::arg("input"), py::arg("target"),
             "Shorthand for forward(input, target).")
        .def("backward", &MSELoss<DType>::backward,
             py::arg("input"), py::arg("target"),
             "Returns gradient w.r.t. input: grad[i][j] = 2*(input[i][j]-target[i][j])/D.");

    // ------------------------------------------------------------------
    // Softmax — numerically stable softmax with temperature.
    // ------------------------------------------------------------------
    py::class_<Softmax<DType>>(m, (type_suffix + "Softmax").c_str(),
        "Numerically stable softmax with optional temperature scaling.\n"
        "p[i] = exp(x[i] / T) / sum_j exp(x[j] / T)\n"
        "Higher T → softer distribution; lower T → sharper (more greedy) distribution.\n"
        "Has no learnable parameters.")
        .def(py::init<double>(), py::arg("temperature") = 1.0,
            "Constructs a Softmax layer with the given temperature (default 1.0).")
        .def("forward", &Softmax<DType>::forward, py::arg("input"),
             "Applies softmax along the last dimension. Returns a probability tensor of the same shape.")
        .def("__call__", &Softmax<DType>::operator(), py::arg("input"),
             "Shorthand for forward(input).")
        .def("backward", &Softmax<DType>::backward,
             py::arg("input"), py::arg("grad_output"),
             "Computes the Jacobian-vector product dL/dx = softmax(x) * (dL/dy - dot(softmax(x), dL/dy)).");

    // ------------------------------------------------------------------
    // NanoGPT — full autoregressive language model.
    // ------------------------------------------------------------------
    py::class_<NanoGPT<DType, IdType>>(m, (type_suffix + "NanoGPT").c_str(),
        "Full autoregressive language model: TokenEmbedding + PositionEmbedding\n"
        "  → TransformerLayer → UnEmbedding → Softmax.\n\n"
        "Supports training (forward / backward / sgdUpdate) and generation\n"
        "(predict / sample / generate) with four sampling modes.\n\n"
        "All three input forms (Tensor, numpy array, Python list) are accepted for\n"
        "forward, sample, and generate to simplify integration with Python data pipelines.")
        .def(py::init<int, int, int, int, int, int, int, int, DType, ActivationType>(),
             py::arg("vocab_size"), py::arg("max_seq_len"), py::arg("embedding_dim"),
             py::arg("num_heads"), py::arg("head_dim"), py::arg("mlp_dim"), py::arg("num_layers"),
             py::arg("checkpoint_gap") = 0, py::arg("temperature") = 1.0,
             py::arg("activation_type") = ActivationType::ReLU,
            "Constructs a NanoGPT model.\n\n"
            "Args:\n"
            "  vocab_size:      Number of unique tokens (vocabulary size).\n"
            "  max_seq_len:     Maximum sequence length the model can handle.\n"
            "  embedding_dim:   Width of all hidden representations.\n"
            "  num_heads:       Number of attention heads per transformer block.\n"
            "  head_dim:        Dimensionality of each attention head.\n"
            "  mlp_dim:         Hidden width of the feed-forward MLP in each block.\n"
            "  num_layers:      Number of transformer blocks.\n"
            "  checkpoint_gap:  Insert a gradient checkpoint every N blocks (0 = no checkpointing).\n"
            "  temperature:     Softmax temperature for generation (default 1.0).\n"
            "  activation_type: Activation function for MLP sub-layers (default ReLU).")
        // Configuration accessors
        .def("vocabSize",      &NanoGPT<DType, IdType>::vocabSize,
            "Returns the vocabulary size passed at construction.")
        .def("maxSeqLen",      &NanoGPT<DType, IdType>::maxSeqLen,
            "Returns the maximum sequence length passed at construction.")
        .def("embeddingDim",   &NanoGPT<DType, IdType>::embeddingDim,
            "Returns the hidden embedding dimension.")
        .def("numHeads",       &NanoGPT<DType, IdType>::numHeads,
            "Returns the number of attention heads per block.")
        .def("headDim",        &NanoGPT<DType, IdType>::headDim,
            "Returns the per-head dimension.")
        .def("mlpDim",         &NanoGPT<DType, IdType>::mlpDim,
            "Returns the MLP hidden dimension.")
        .def("numLayers",      &NanoGPT<DType, IdType>::numLayers,
            "Returns the total number of transformer blocks.")
        .def("checkpointGap",  &NanoGPT<DType, IdType>::checkpointGap,
            "Returns the gradient checkpoint interval (0 = none).")
        .def("getTemperature", &NanoGPT<DType, IdType>::getTemperature,
            "Returns the current softmax temperature used for generation.")
        .def("setTemperature", &NanoGPT<DType, IdType>::setTemperature, py::arg("temperature"),
            "Sets the softmax temperature. Values > 1 soften the distribution; < 1 sharpen it.")
        .def("activationType", &NanoGPT<DType, IdType>::activationType,
            "Returns the ActivationType enum value used by the MLP sub-layers.")
        // Forward overloads
        .def("forward",
            (Tensor<DType> (NanoGPT<DType, IdType>::*)(Tensor<IdType>)) &NanoGPT<DType, IdType>::forward,
            py::arg("input"),
            "Runs the full forward pass on a GPU Tensor of token IDs.\n"
            "input: [batchSize, seqLen]. Returns softmax probabilities [batchSize, seqLen, vocabSize].")
        .def("forward",
            (Tensor<DType> (NanoGPT<DType, IdType>::*)(pybind11::array_t<IdType>)) &NanoGPT<DType, IdType>::forward,
            py::arg("input"),
            "Runs the full forward pass on a 1-D or 2-D NumPy array of token IDs. Returns probability tensor.")
        .def("forward",
            (Tensor<DType> (NanoGPT<DType, IdType>::*)(std::vector<IdType>)) &NanoGPT<DType, IdType>::forward,
            py::arg("input"),
            "Runs the full forward pass on a Python list of token IDs. Returns probability tensor.")
        .def("__call__",
            (Tensor<DType> (NanoGPT<DType, IdType>::*)(Tensor<IdType>)) &NanoGPT<DType, IdType>::operator(),
            py::arg("input"), "Shorthand for forward(input).")
        .def("__call__",
            (Tensor<DType> (NanoGPT<DType, IdType>::*)(pybind11::array_t<IdType>)) &NanoGPT<DType, IdType>::operator(),
            py::arg("input"), "Shorthand for forward(input).")
        .def("__call__",
            (Tensor<DType> (NanoGPT<DType, IdType>::*)(std::vector<IdType>)) &NanoGPT<DType, IdType>::operator(),
            py::arg("input"), "Shorthand for forward(input).")
        // Generation
        .def("predict", &NanoGPT<DType, IdType>::predict,
            py::arg("logits"), py::arg("mode"), py::arg("K") = 10, py::arg("P") = 0.1,
            "Samples the next token from pre-computed logits [seqLen, vocabSize].\n"
            "mode: Greedy, Random, Nucleus (threshold P), or TopK (top K tokens).\n"
            "Does not support batched logits.")
        .def("sample",
            (IdType (NanoGPT<DType, IdType>::*)(Tensor<IdType>, SamplingMode, int, double)) &NanoGPT<DType, IdType>::sample,
            py::arg("input"), py::arg("mode"), py::arg("K") = 10, py::arg("P") = 0.1,
            "Runs forward on a 1-D token-ID Tensor and returns the sampled next token ID.")
        .def("sample",
            (IdType (NanoGPT<DType, IdType>::*)(pybind11::array_t<IdType>, SamplingMode, int, double)) &NanoGPT<DType, IdType>::sample,
            py::arg("input"), py::arg("mode"), py::arg("K") = 10, py::arg("P") = 0.1,
            "Runs forward on a 1-D NumPy array and returns the sampled next token ID.")
        .def("sample",
            (IdType (NanoGPT<DType, IdType>::*)(std::vector<IdType>, SamplingMode, int, double)) &NanoGPT<DType, IdType>::sample,
            py::arg("input"), py::arg("mode"), py::arg("K") = 10, py::arg("P") = 0.1,
            "Runs forward on a Python list and returns the sampled next token ID.")
        .def("generate",
            (Tensor<IdType> (NanoGPT<DType, IdType>::*)(Tensor<IdType>, int, SamplingMode, int, double)) &NanoGPT<DType, IdType>::generate,
            py::arg("input"), py::arg("num_tokens"), py::arg("mode"), py::arg("K") = 10, py::arg("P") = 0.1,
            "Autoregressively appends `num_tokens` tokens to a 1-D token-ID Tensor.")
        .def("generate",
            (pybind11::array_t<IdType> (NanoGPT<DType, IdType>::*)(pybind11::array_t<IdType>, int, SamplingMode, int, double)) &NanoGPT<DType, IdType>::generate,
            py::arg("input"), py::arg("num_tokens"), py::arg("mode"), py::arg("K") = 10, py::arg("P") = 0.1,
            "Autoregressively appends `num_tokens` tokens to a 1-D NumPy array and returns the full sequence.")
        .def("generate",
            (std::vector<IdType> (NanoGPT<DType, IdType>::*)(std::vector<IdType>, int, SamplingMode, int, double)) &NanoGPT<DType, IdType>::generate,
            py::arg("input"), py::arg("num_tokens"), py::arg("mode"), py::arg("K") = 10, py::arg("P") = 0.1,
            "Autoregressively appends `num_tokens` tokens to a Python list and returns the full sequence.")
        .def("loss", &NanoGPT<DType, IdType>::loss,
            py::arg("logits"), py::arg("target"),
            "Computes cross-entropy loss given logits [batchSize, seqLen, vocabSize] and targets [batchSize, seqLen].")
        .def("backward", &NanoGPT<DType, IdType>::backward,
            py::arg("input"), py::arg("logits"), py::arg("target"),
            "Full backward pass: computes and accumulates gradients for all parameters.\n"
            "Must be called with the same input and logits that were used in the forward pass.\n"
            "Call zeroGrad() before each backward pass to clear stale gradients.")
        .def("zeroGrad", &NanoGPT<DType, IdType>::zeroGrad,
            "Resets all parameter gradients to zero across the entire model.")
        .def("sgdUpdate", &NanoGPT<DType, IdType>::sgdUpdate, py::arg("lr"),
            "Applies one SGD step in-place to all trainable parameters: param -= lr * grad.")
        .def("getParameters", &NanoGPT<DType, IdType>::getParameters,
            "Returns a dict mapping fully-qualified parameter names to their Tensor views.\n"
            "Keys include 'tokenEmbedding', 'positionEmbedding', 'unEmbedding', "
            "and 'layer_N.attention.*' / 'layer_N.mlp.*' for each transformer block.")
        .def("setParameters", &NanoGPT<DType, IdType>::setParameters, py::arg("params"),
            "Loads parameters from a name→Tensor dict. Unrecognised keys are ignored.\n"
            "Key format must match getParameters() output.")
        .def("getGradients", &NanoGPT<DType, IdType>::getGradients,
            "Returns a dict mapping parameter names to their accumulated gradient Tensors.\n"
            "Gradient buffers are lazily allocated; call backward() first for meaningful values.")
        .def("clear", &NanoGPT<DType, IdType>::clear,
            "Clears all stored checkpoint activations in the model to free up memory.");
}

PYBIND11_MODULE(cuda_transformer, m) {
    m.doc() =
        "cuda_transformer: a CUDA-accelerated neural network module library.\n\n"
        "Tensors:\n"
        "  FloatTensor, DoubleTesor, HalfTensor, BFloatTensor — GPU tensors for floating-point types.\n"
        "  IntTensor, LongTensor, ShortTensor, ByteTensor, BoolTensor — GPU tensors for integer types.\n\n"
        "Layers (default float; append 'Float'/'Double'/'Half'/'BFloat' for typed variants):\n"
        "  LayerNorm(input_dim) — layer normalization with learnable scale and bias.\n"
        "  Linear(input_dim, output_dim)                              — fully connected layer\n"
        "  Activation(input_dim, ActivationType)                      — ReLU / GELU / Sigmoid / Tanh\n"
        "  MLP([layer, ...])                                           — sequential container\n"
        "  Attention(input_dim, num_heads, head_dim)                  — multi-head attention\n"
        "  TransformerBlock(input_dim, num_heads, head_dim, mlp_dim)  — Attention + MLP block\n"
        "  Transformer(input_dim, num_heads, head_dim, mlp_dim,\n"
        "              num_layers [, checkpoint_gap, activation_type]) — full transformer\n\n"
        "Others:\n"
        "  TokenEmbedding(vocab_size, embedding_dim) — maps token IDs to dense vectors.\n"
        "  PositionEmbedding(max_seq_len, embedding_dim) — adds learnable positional encodings.\n"
        "  UnEmbedding(vocab_size, embedding_dim) — projects from embedding space back to vocabulary logits.\n"
        "  Softmax(temperature) — numerically stable softmax with temperature scaling for generation.\n"
        "  CrossEntropyLoss() — -log(p[target]) loss for classification tasks; expects softmax probabilities as input.\n"
        "  MSELoss() — mean squared error loss for regression tasks.\n"
        "NanoGPT(vocab_size, max_seq_len, embedding_dim, num_heads, head_dim, mlp_dim, num_layers,\n"
        "        [, checkpoint_gap, activation_type]) — full LLM with training and generation methods.\n\n"
        "\n"
        "Optimizers:\n"
        "  SGDOptimizer(lr) — explicit map-based SGD; all layers also expose layer.sgdUpdate(lr).";

    // Activation type enum — scoped so values don't pollute the Python namespace.
    py::enum_<ActivationType>(m, "ActivationType",
        "Enumeration of supported activation functions for ActivationLayer and Transformer variants.")
        .value("ReLU",    ActivationType::ReLU,    "Rectified Linear Unit: max(0, x).")
        .value("GELU",    ActivationType::GELU,    "Gaussian Error Linear Unit (tanh approximation).")
        .value("Sigmoid", ActivationType::Sigmoid, "Sigmoid: 1 / (1 + exp(-x)).")
        .value("Tanh",    ActivationType::Tanh,    "Hyperbolic tangent: tanh(x).")
        .export_values();

    // Sampling mode enum.
    py::enum_<SamplingMode>(m, "SamplingMode",
        "Token sampling strategy used by NanoGPT.predict / sample / generate.")
        .value("Greedy",  SamplingMode::Greedy,
            "Always picks the highest-probability token (argmax).")
        .value("Random",  SamplingMode::Random,
            "Samples a token according to the full probability distribution.")
        .value("Nucleus", SamplingMode::Nucleus,
            "Nucleus (top-p) sampling: retains only tokens with probability > P, then samples.")
        .value("TopK",    SamplingMode::TopK,
            "Top-K sampling: retains the K highest-probability tokens, then samples.")
        .export_values();

    // Default float factory functions (no suffix — most convenient for interactive use).
    m.def("LayerNorm",        &LayerNorm<float>,
        py::arg("input_dim"), py::arg("epsilon") = 1e-5,
        "Creates a float LayerNormLayer with the given input dimension and epsilon for numerical stability.");
    m.def("Linear",           &Linear<float>,
        py::arg("input_dim"), py::arg("output_dim"),
        "Creates a float LinearLayer.");
    m.def("Activation",       &Activation<float>,
        py::arg("input_dim"), py::arg("activation_type"),
        "Creates a float ActivationLayer (ReLU, GELU, Sigmoid, or Tanh).");
    m.def("Checkpoint",        &Checkpoint<float>,
        "Layer for gradient checkpointing (storing activations), does nothing.");
    m.def("MLP",              &MLP<float>,
        py::arg("layers"),
        "Creates a float MLPLayer from a list of sub-modules.");
    m.def("Attention",        &Attention<float>,
        py::arg("input_dim"), py::arg("num_heads"), py::arg("head_dim"),
        "Creates a float multi-head AttentionLayer.");
    m.def("TransformerBlock", &TransformerBlock<float>,
        py::arg("input_dim"), py::arg("num_heads"), py::arg("head_dim"),
        py::arg("mlp_dim"), py::arg("activation_type") = ActivationType::ReLU,
        "Creates a single float Transformer block (Attention + MLP).");
    m.def("Transformer",      &Transformer<float>,
        py::arg("input_dim"), py::arg("num_heads"), py::arg("head_dim"),
        py::arg("mlp_dim"), py::arg("num_layers"),
        py::arg("checkpoint_gap") = 0, py::arg("activation_type") = ActivationType::ReLU,
        "Creates a full float Transformer model. checkpoint_gap > 0 inserts gradient checkpoints.");

    // Multi-precision variants (tensors + typed layers + typed optimizers).
    declare_tensor_and_modules<__nv_bfloat16>(m, "BFloat");
    declare_tensor_and_modules<__half>(m, "Half");
    declare_tensor_and_modules<float>(m, "Float");
    declare_tensor_and_modules<double>(m, "Double");

    // Integer tensor variants (no layer bindings — integers don't support gradients).
    declare_tensor<int>(m, "Int");
    declare_tensor<long>(m, "Long");
    declare_tensor<short>(m, "Short");
    declare_tensor<signed char>(m, "Byte");
    declare_tensor<bool>(m, "Bool");

    declare_nanogpt<float, int>(m, "Float");
    declare_nanogpt<__nv_bfloat16, int>(m, "BFloat");
    declare_nanogpt<__half, int>(m, "Half");
    declare_nanogpt<double, int>(m, "Double");
}
