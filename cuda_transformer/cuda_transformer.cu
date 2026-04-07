#include "TransformerLayer.cu"
#include "SGDOptimizer.cu"
// Above includes auto-include everything; this is the final compilation unit.

namespace py = pybind11;

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
        .def("reshape", &Tensor<DType>::reshape, py::arg("shape"),
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
        .def("sgdUpdate", &Layer<DType>::sgdUpdate, py::arg("lr"),
            "Applies a single SGD step in-place: param -= lr * grad. "
            "For container layers (MLP, TransformerBlock, Transformer) this propagates "
            "recursively to all sub-layers. Stateless layers (Activation) are a no-op.")
        .def("__call__", &Layer<DType>::operator(), py::arg("input"),
            "Shorthand for forward(input).");

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
    // AttentionLayer — multi-head attention with QKV + output projection.
    // ------------------------------------------------------------------
    py::class_<AttentionLayer<DType>, Layer<DType>, std::shared_ptr<AttentionLayer<DType>>>(
            m, (type_suffix + "AttentionLayer").c_str(),
            "Multi-head attention layer composed of QKV projection, flash attention, and output projection.\n"
            "All get/set methods perform shallow copies (shared GPU buffer, no data movement).\n"
            "QKV weight shape: [inputDim, headDim*numHeads]. Output weight shape: [headDim*numHeads, inputDim].")
        .def("forward",          &AttentionLayer<DType>::forward,         py::arg("input"),
            "Runs QKV projection -> flash attention -> output projection and returns [batch, seq, inputDim].")
        .def("backward",         &AttentionLayer<DType>::backward,        py::arg("input"), py::arg("grad_output"),
            "Backpropagates through all three sub-components, accumulates all parameter gradients, "
            "and returns the input gradient.")
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
    // SGDOptimizer — explicit map-based optimizer (alternative to layer.sgdUpdate).
    // ------------------------------------------------------------------
    py::class_<SGDOptimizer<DType>>(m, (type_suffix + "SGDOptimizer").c_str(),
        "Explicit SGD optimizer that operates on parameter/gradient pointer maps.\n"
        "Useful with LinearLayer.getParameters() / getGradients().\n"
        "For whole-layer-tree updates, prefer layer.sgdUpdate(lr) instead.")
        .def(py::init<DType>(), py::arg("lr"),
            "Constructs an SGDOptimizer with the given learning rate.")
        .def("step", &SGDOptimizer<DType>::step,
            py::arg("params"), py::arg("grads"),
            "Applies one SGD step: for each matching key, params[k] -= lr * grads[k].")
        .def("setLearningRate", &SGDOptimizer<DType>::setLearningRate, py::arg("lr"),
            "Updates the learning rate.")
        .def("getLearningRate", &SGDOptimizer<DType>::getLearningRate,
            "Returns the current learning rate.");

    // ------------------------------------------------------------------
    // Factory functions (return base Module, resolved to concrete type by pybind11).
    // ------------------------------------------------------------------
    m.def(("Linear" + type_suffix).c_str(),           &Linear<DType>,
        py::arg("input_dim"), py::arg("output_dim"),
        "Creates a LinearLayer: output = input @ weights.T + biases.");
    m.def(("Activation" + type_suffix).c_str(),       &Activation<DType>,
        py::arg("input_dim"), py::arg("activation_type"),
        "Creates an ActivationLayer with the given ActivationType (ReLU, GELU, Sigmoid, Tanh).");
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

template <typename DType>
void declare_tensor_and_modules(py::module &m, const std::string &type_suffix) {
    declare_tensor<DType>(m, type_suffix);
    declare_modules<DType>(m, type_suffix);
}

PYBIND11_MODULE(cuda_transformer, m) {
    m.doc() =
        "sysml: a CUDA-accelerated neural network module library.\n\n"
        "Tensors:\n"
        "  FloatTensor, DoubleTesor, HalfTensor, BFloatTensor — GPU tensors for floating-point types.\n"
        "  IntTensor, LongTensor, ShortTensor, ByteTensor, BoolTensor — GPU tensors for integer types.\n\n"
        "Layers (default float; append 'Float'/'Double'/'Half'/'BFloat' for typed variants):\n"
        "  Linear(input_dim, output_dim)                              — fully connected layer\n"
        "  Activation(input_dim, ActivationType)                      — ReLU / GELU / Sigmoid / Tanh\n"
        "  MLP([layer, ...])                                           — sequential container\n"
        "  Attention(input_dim, num_heads, head_dim)                  — multi-head attention\n"
        "  TransformerBlock(input_dim, num_heads, head_dim, mlp_dim)  — Attention + MLP block\n"
        "  Transformer(input_dim, num_heads, head_dim, mlp_dim,\n"
        "              num_layers [, checkpoint_gap, activation_type]) — full transformer\n\n"
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

    // Default float factory functions (no suffix — most convenient for interactive use).
    m.def("Linear",           &Linear<float>,
        py::arg("input_dim"), py::arg("output_dim"),
        "Creates a float LinearLayer.");
    m.def("Activation",       &Activation<float>,
        py::arg("input_dim"), py::arg("activation_type"),
        "Creates a float ActivationLayer (ReLU, GELU, Sigmoid, or Tanh).");
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
}
