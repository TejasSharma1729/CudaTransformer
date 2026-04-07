#include "tensor_kernels.cu"
#ifndef CLASS_TENSOR
#define CLASS_TENSOR

/**
 * @brief Central interface defining robust N-Dimensional arbitrary computational matrix relationships.
 * Generates dynamic device mappings bridging native scalable GPU evaluation seamlessly from user layers.
 */
template <typename DType = float>
class Tensor {
    std::shared_ptr<DType[]> data_ = nullptr;
    Shape8 shape_;
    size_t size_ = 0;

public:
    /**
     * @brief Default constructor natively evaluating implicitly hollow tensor memory structures dynamically.
     */
    Tensor() : size_(0), data_(nullptr) { }
    
    /**
     * @brief Constructs natively explicit contiguous 1-Dimensional bounds saturated instantly symmetrically.
     * @param size The maximum geometric boundary constraint governing absolute total volume.
     * @param value Default raw filling block saturating allocated bounds inherently.
     */
    Tensor(size_t size, DType value = 0) : shape_({size}), size_(size) {
        data_ = cudaMakeShared<DType>(size, value);
    }
    
    /**
     * @brief Instantiates geometric N-Dimensional grids automatically allocating matrix volume scaling securely.
     * @param s Mathematical standard vector defining layout capacities up to extensive permutations dynamically.
     * @param value Initial default constant inherently occupying raw parameter space identically logically.
     */
    Tensor(const std::vector<size_t> &s, DType value = 0) : shape_(s) {
        size_ = 1;
        for (size_t i = 0; i < shape_.nDim; ++i) {
            size_ *= shape_[i];
        }
        data_ = cudaMakeShared<DType>(size_, value);
    }
    
    /**
     * @brief Shallow-wraps an existing device buffer without allocating new memory.
     * The resulting Tensor shares ownership of the buffer via the shared_ptr.
     * @param data  Shared pointer to the existing device buffer.
     * @param shape Shape of the tensor matching the buffer size.
     */
    Tensor(std::shared_ptr<DType[]> data, const std::vector<size_t> &shape) : shape_(shape) {
        size_ = 1;
        for (size_t i = 0; i < shape_.nDim; ++i) size_ *= shape_[i];
        data_ = data;
    }

    /**
     * @brief Internal tracking constructing implicit mapping correctly bridging references to identical locations.
     * @param other Target object projecting parameters into this mirrored instance successfully dynamically.
     */
    Tensor(const Tensor &other) : size_(other.size_), shape_(other.shape_), data_(other.data_) { }
    
    /**
     * @brief Relocates identical pointer representation completely inherently moving uncopied memory rapidly.
     * @param other Source temporary memory architecture fully divested safely during migration efficiently.
     */
    Tensor(Tensor &&other) noexcept : 
        size_(other.size_), shape_(other.shape_), data_(std::move(other.data_)) {
        other.size_ = 0;
        other.shape_ = Shape8();
    }
    
    /**
     * @brief Evaluates identically matching logical references seamlessly modifying internal states carefully.
     * @param other Standard immutable configuration being deeply referenced fundamentally appropriately.
     * @return Tensor& Immediate corresponding mutable reference targeting current assignment inherently natively.
     */
    Tensor &operator=(const Tensor &other) {
        if (this != &other) {
            data_ = other.data_;
            shape_ = other.shape_;
            size_ = other.size_;
        }
        return *this;
    }
    
    /**
     * @brief Safely transplants internal memory graphs unrestrictedly isolating target mapping logically fundamentally.
     * @param other Source architecture systematically invalidated during rigorous displacement mathematically structurally.
     * @return Tensor& Strict immediate referencing modifying identical configurations mapping appropriately natively.
     */
    Tensor &operator=(Tensor &&other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            shape_ = other.shape_;
            size_ = other.size_;
            other.size_ = 0;
            other.shape_ = Shape8();
        }
        return *this;
    }

    /**
     * @brief Independently segregates physically identical mapped graphs creating complete independent matrix clones.
     * @return Tensor Natively detached clone fundamentally holding completely uncoupled copies natively geometrically.
     */
    Tensor clone() const {
        Tensor other = *this;
        other.data_ = cudaCloneShared<DType>(data_, size_);
        return other;
    }
    
    /**
     * @brief Accesses exact current geometric bounds capturing inherent shapes natively.
     * @return const Shape8& Strict geometric reference immutable representation completely.
     */
    const Shape8& shape() const { return shape_; }
    
    /**
     * @brief Fetches explicitly constrained lengths inherently targeting single axes safely.
     * @param dim Independent numerical boundary capturing absolute geometric dimension logically.
     * @return size_t Structural magnitude exclusively mapping specified constraints.
     */
    size_t shape(size_t dim) const {
        if (dim >= shape_.nDim) throw std::runtime_error("Invalid dimension");
        return shape_[dim];
    }
    
    /**
     * @brief Measures unified physical capacities bounding exactly memory capacities locally.
     * @return size_t Aggregate element magnitude mapping geometrically reliably natively.
     */
    size_t size() const { return size_; }
    
    /**
     * @brief Returns absolute topological depths counting exactly defined dimensions currently.
     * @return size_t Dimension length geometrically natively explicitly bounding grids.
     */
    size_t nDim() const { return shape_.nDim; }
    
    /**
     * @brief Unlocks raw fundamental pointer references cleanly managing matrix arrays dynamically.
     * @return DType* Unchecked raw memory pointer explicitly bypassing strict encapsulation logic.
     */
    DType *data() { return data_.get(); }
    const DType *data() const { return data_.get(); }
    DType *mutableData() { return data_.get(); }
    DType *get() const { return data_.get(); }

    /**
     * @brief Returns the shared_ptr to the underlying device buffer (shallow ownership).
     * Used by layer get/set methods to share the buffer without copying.
     * @return std::shared_ptr<DType[]> Shared pointer to the device buffer.
     */
    std::shared_ptr<DType[]> dataPtr() const { return data_; }

    /**
     * @brief Converts host pointer maps systematically directly mirroring into isolated device copies.
     * @param s Fundamental structural standard identifying vector geometric array scaling bounds natively.
     * @param ptr Unmanaged source constant identifying raw array extraction targets independently carefully.
     * @return Tensor Exact instantiated representation logically bounding complete device memory matrices explicitly.
     */
    static Tensor fromPointer(const std::vector<size_t>& s, const DType* ptr) {
        Tensor t(s);
        cudaMemcpy(t.get(), ptr, t.size() * sizeof(DType), cudaMemcpyHostToDevice);
        return t;
    }

    /**
     * @brief Consumes native NumPy python layer array bindings seamlessly avoiding system overhead.
     * @param arr Target PyBind explicit architecture tracking python variables correctly cleanly globally.
     * @return Tensor Natively decoupled device matrix translating absolutely structural dependencies easily.
     */
    static Tensor fromNumpy(const pybind11::array_t<DType> &arr) {
        const pybind11::ssize_t *shape_py = arr.shape();
        std::vector<size_t> shape;
        for (size_t i = 0; i < arr.ndim(); i++) {
            shape.push_back(static_cast<size_t>(shape_py[i]));
        }
        if (shape.size() >= 8) {
            throw std::runtime_error("Tensor dimension exceeds 8");
        }
        Tensor t(shape);
        cudaMemcpy(t.get(), arr.data(), t.size() * sizeof(DType), cudaMemcpyHostToDevice);
        return t;
    }

    /**
     * @brief Extracts identical memory maps natively porting strictly bound elements into the CPU.
     * @return std::vector<DType> A comprehensive native array containing identically matched variable values.
     */
    std::vector<DType> cpu() const {
        if constexpr (std::is_same_v<DType, bool>) {
            std::vector<bool> host_vector(size_);
            bool *host_data = new bool[size_];
            cudaMemcpy(
                host_data,
                data_.get(),
                size_,
                cudaMemcpyDeviceToHost
            );
            for (int i = 0; i < size_; i++) {
                host_vector[i] = host_data[i];
            }
            delete[] host_data;
            return host_vector;
        } else {
            std::vector<DType> host_data(size_);
            cudaMemcpy(
                host_data.data(),
                data_.get(),
                size_ * sizeof(DType),
                cudaMemcpyDeviceToHost
            );
            return host_data;
        }
    }

    auto numpy() const {
        if constexpr (std::is_same_v<DType, __half> || std::is_same_v<DType, __nv_bfloat16>) {
            Tensor<float> temp = this->to<float>();
            float *host_data = new float[temp.size()];
            cudaMemcpy(host_data, temp.get(), temp.size() * sizeof(float), cudaMemcpyDeviceToHost);
            auto capsule = pybind11::capsule(host_data, [](void *p) { delete[] reinterpret_cast<float *>(p); });
            return pybind11::array_t<float>(shape_.toVector(), host_data, capsule);
        } else {
            DType *host_data = new DType[size_];
            cudaMemcpy(host_data, data_.get(), size_ * sizeof(DType), cudaMemcpyDeviceToHost);
            auto capsule = pybind11::capsule(host_data, [](void *p) { delete[] reinterpret_cast<DType *>(p); });
            return pybind11::array_t<DType>(shape_.toVector(), host_data, capsule);
        }
    }

    /**
     * @brief Transforms exactly mirrored native internal data structures strictly conforming directly to new requested precisions dynamically.
     * @return Tensor<NewDType> The thoroughly transformed completely type-safe array.
     */
    template <typename NewDType>
    Tensor<NewDType> to() const {
        Tensor<NewDType> result(shape_.toVector());
        if (size_ > 0) {
            castKernel<DType, NewDType><<<(size_ + 255) / 256, 256>>>(data_.get(), result.get(), size_);
        }
        return result;
    }
    /**
     * @brief Explicit wrapper retrieving singular scalar extractions immediately natively.
     * @return DType Strict extracted natively explicit standard structural scalar value thoroughly.
     */
    DType item() const {
        if (size_ != 1) {
            throw std::runtime_error("Tensor is not a scalar");
        }
        DType val;
        cudaMemcpy(&val, data_.get(), sizeof(DType), cudaMemcpyDeviceToHost);
        return val;
    }

    /**
     * @brief Transient internal assignment encapsulation managing structural scalar variable injection dynamically.
     */
    struct TensorElemAssign_ {
        Tensor *tensor;
        std::vector<size_t> indices;
        
        /**
         * @brief Evaluates inherently explicit bounding geometry implicitly identifying values statically.
         * @param t Mathematical struct bound parent logical representation actively maintaining structures natively.
         * @param idx Specific spatial explicit dimension index tracking correctly inherently accurately identically effectively explicitly cleanly carefully structurally explicitly cleanly appropriately cleanly securely effortlessly accurately systematically.
         */
        TensorElemAssign_(Tensor *t, const std::vector<size_t> &idx) : 
            tensor(t), indices(idx) {}

        /**
         * @brief Automatic raw element retrieval logic extracting deeply safely natively identically automatically.
         * @return DType Structural exact element extracted structurally effectively properly reliably dynamically natively.
         */
        operator DType() const {
            size_t offset = 0, stride = 1;
            for (int i = (int)tensor->nDim() - 1; i >= 0; i--) {
                offset += indices[i] * stride;
                stride *= tensor->shape_[i];
            }
            DType val;
            cudaMemcpy(&val, tensor->data_.get() + offset, sizeof(DType), cudaMemcpyDeviceToHost);
            return val;
        }
        
        /**
         * @brief Unifies injection structurally pushing value arrays deeply perfectly natively mapping strictly efficiently safely.
         * @param value Pure numeric data modifying underlying tensor locations securely smoothly inherently cleanly directly accurately natively natively reliably identically correctly precisely completely appropriately fundamentally appropriately directly effectively easily securely.
         * @return DType Exact unmodified value passed tracking appropriately appropriately safely.
         */
        DType operator=(DType value) {
            size_t offset = 0, stride = 1;
            for (int i = (int)tensor->nDim() - 1; i >= 0; i--) {
                offset += indices[i] * stride;
                stride *= tensor->shape_[i];
            }
            cudaMemcpy(
                tensor->data_.get() + offset, 
                &value, 
                sizeof(DType), 
                cudaMemcpyHostToDevice
            );
            return value;
        }
    };

    /**
     * @brief Binds scalar manipulation immediately natively securely structurally.
     * @param indices Dimension lookup coordinates mapping appropriately mathematically mathematically effectively cleanly successfully quickly.
     * @return TensorElemAssign_ Proxy modifier capturing elements cleanly precisely perfectly effortlessly flawlessly safely rapidly natively smoothly functionally rigorously strictly explicitly.
     */
    TensorElemAssign_ operator()(const std::vector<size_t> &indices) {
        return TensorElemAssign_(this, indices);
    }
    
    /**
     * @brief Validates absolute values inherently reading scalar arrays statically cleanly explicitly explicitly clearly cleanly securely directly directly consistently safely safely perfectly gracefully reliably easily independently confidently dynamically intelligently flawlessly flawlessly gracefully effectively elegantly accurately comprehensively cleanly.
     * @param indices Dimensional targets exactly specifying bounds exactly functionally safely intuitively dynamically flawlessly seamlessly fundamentally flawlessly correctly carefully identically structurally natively efficiently cleanly securely statically strictly correctly fundamentally correctly dynamically efficiently reliably quickly logically dynamically implicitly.
     * @return DType Scalar isolated representation mapping efficiently smoothly intuitively.
     */
    DType operator()(const std::vector<size_t> &indices) const {
        return static_cast<DType>(TensorElemAssign_(const_cast<Tensor *>(this), indices));
    }

    /**
     * @brief Repopulates strictly explicitly sweeping constants universally instantly replacing contents.
     * @param value The scalar to fill the tensor with globally.
     */
    void fill(DType value) {
        fillKernel<DType><<<(size_ + 255) / 256, 256>>>(data_.get(), size_, value);
    }

    /**
     * @brief Parses complicated sub-indexing expressions slicing memory subsets cleanly.
     * Evaluates geometric mapping immediately on the CPU to completely avoid GPU pointer divergence.
     * @param view A vector of dimensional IndexRanges specifying the subset explicitly.
     * @return Tensor A perfectly detached mapped memory instance targeting the strict slice parameters.
     */
    Tensor getSlice(const TensorView &view) const {
        std::vector<size_t> newShape;
        std::vector<size_t> sliceStrides;
        size_t baseOffset = 0;
        
        std::vector<size_t> origStrides(nDim() > 0 ? nDim() : 1, 1);
        if (nDim() > 1) {
            for (int i = (int)nDim() - 2; i >= 0; i--) {
                origStrides[i] = origStrides[i+1] * shape_[i+1];
            }
        }

        for (size_t i = 0; i < nDim(); i++) {
            size_t start = 0, end = shape_[i];
            bool is_range = true;
            
            if (i < view.size()) {
                if (view[i].is_all) {
                    // Implicitly ignores processing leaving parameters unmodified
                } else if (view[i].is_range) {
                    start = view[i].start; 
                    end = view[i].end;
                } else {
                    start = view[i].start; 
                    end = view[i].end;
                    is_range = false;
                }
            }
            
            baseOffset += start * origStrides[i];
            
            if (is_range) {
                newShape.push_back(end - start);
                sliceStrides.push_back(origStrides[i]);
            }
        }
        
        Tensor result(newShape);
        
        sliceKernelFast<DType><<<(result.size() + 255) / 256, 256>>>(
            const_cast<DType *>(data_.get()),
            result.data_.get(),
            baseOffset,
            Shape8(newShape),
            Shape8(sliceStrides),
            Shape8(newShape),
            newShape.size(),
            newShape.size(),
            result.size(),
            0
        );
        return result;
    }

    /**
     * @brief Merges disjoint overlapping tensors directly projecting parameters effectively.
     * @param view An expansive geometric boundary establishing where data maps exactly.
     * @param val Secondary source tensor inherently mapped directly onto targeted layout structurally.
     */
    void set(const TensorView &view, const Tensor &val) {
        std::vector<size_t> newShape;
        std::vector<size_t> sliceStrides;
        size_t baseOffset = 0;
        
        std::vector<size_t> origStrides(nDim() > 0 ? nDim() : 1, 1);
        if (nDim() > 1) {
            for (int i = (int)nDim() - 2; i >= 0; i--) {
                origStrides[i] = origStrides[i+1] * shape_[i+1];
            }
        }

        for (size_t i = 0; i < nDim(); i++) {
            size_t start = 0, end = shape_[i];
            bool is_range = true;
            
            if (i < view.size()) {
                if (view[i].is_all) {
                } else if (view[i].is_range) {
                    start = view[i].start; 
                    end = view[i].end;
                } else {
                    start = view[i].start; 
                    end = view[i].end;
                    is_range = false;
                }
            }
            
            baseOffset += start * origStrides[i];
            if (is_range) {
                newShape.push_back(end - start);
                sliceStrides.push_back(origStrides[i]);
            }
        }
        
        size_t total_elements = 1;
        for (size_t s : newShape) total_elements *= s;

        sliceKernelFast<DType><<<(total_elements + 255) / 256, 256>>>(
            data_.get(),
            const_cast<DType *>(val.data()),
            baseOffset,
            Shape8(newShape),
            Shape8(sliceStrides),
            val.shape(),
            newShape.size(),
            val.nDim(),
            total_elements,
            1
        );
    }

    /**
     * @brief Applies exact scalar proxy bounds inherently scaling parameter configurations directly.
     */
    void set(const TensorView &view, DType value) {
        Tensor val(std::vector<size_t>{}, value);
        set(view, val);
    }

    struct TensorAssignment_ {
        Tensor &tensor;
        TensorView view;
        TensorAssignment_(Tensor &t, const TensorView &v) : tensor(t), view(v) {}
        
        Tensor operator=(const Tensor &val) {
            tensor.set(view, val);
            return tensor.getSlice(view);
        }
        
        Tensor operator=(DType value) {
            tensor.set(view, value);
            return tensor.getSlice(view);
        }
        
        operator Tensor() const { return tensor.getSlice(view); }
        operator DType() const { return tensor.getSlice(view).item(); }
        size_t size() const { return tensor.getSlice(view).size(); }
        size_t nDim() const { return tensor.getSlice(view).nDim(); }
        Shape8 shape() const { return tensor.getSlice(view).shape(); }
        DType item() const { return tensor.getSlice(view).item(); }
    };

    TensorAssignment_ operator[](const TensorView &view) {
        return TensorAssignment_(*this, view);
    }
    
    TensorAssignment_ operator[](const TensorView &view) const {
        return TensorAssignment_(const_cast<Tensor&>(*this), view);
    }

    /**
     * @brief Translates structural matrix arrays safely mimicking intrinsic parameters seamlessly.
     */
    Tensor reshape(
        const std::vector<size_t> &newShape /*!< @param newShape Complete vector capturing total volume dimension bounds */
    ) const {
        size_t newSize = 1;
        for (auto s : newShape) newSize *= s;
        
        if (newSize != size_) {
            throw std::runtime_error("Reshape size mismatch");
        }
        
        Tensor result = *this;
        result.shape_ = Shape8(newShape);
        return result;
    }

    /**
     * @brief Inverses strict geometric ordering rigorously passing heavily decoupled dimensions safely.
     */
    Tensor transpose(
        const std::vector<size_t> &perm /*!< @param perm Geometric order of sequence indices mapping new dimensions */
    ) const {
        if (perm.size() != nDim()) {
            throw std::runtime_error("Transpose perm mismatch");
        }
        
        std::vector<size_t> newShape(nDim());
        for (size_t i = 0; i < nDim(); i++) newShape[i] = shape_[perm[i]];
        
        Tensor result(newShape);
        transposeKernel<DType><<<(size_ + 255) / 256, 256>>>(
            data_.get(),
            result.mutableData(),
            shape_,
            Shape8(perm),
            nDim()
        );
        return result;
    }

    Tensor transpose(
        size_t axis1 /*!< @param axis1 Root axis */, 
        size_t axis2 /*!< @param axis2 Target axis */
    ) const {
        std::vector<size_t> p(nDim());
        for (size_t i = 0; i < nDim(); i++) {
            p[i] = i;
        }
        std::swap(p[axis1], p[axis2]);
        return transpose(p);
    }

    /**
     * @brief Automatically sweeps explicitly calculated geometric operators scaling universally identically.
     */
    Tensor unary(
        UnaryOp op /*!< @param op Single-parameter mathematical transformation map */
    ) const { 
        Tensor result(shape_.toVector());
        unaryOpKernel<DType><<<(size_ + 255) / 256, 256>>>(
            data_.get(),
            result.get(),
            size_,
            op
        );
        return result;
    }
    
    Tensor operator-() const { return unary(UnaryOp::NEG); }
    Tensor operator!() const { return unary(UnaryOp::NOT); }

    /**
     * @brief Broadly binds identically topological mappings converting intrinsic mathematical values safely.
     */
    Tensor binary(
        const Tensor &other /*!< @param other Operative geometric partner matrix */, 
        BinaryOp op /*!< @param op Map logic evaluation primitive */
    ) const {
        size_t n1 = nDim(), n2 = other.nDim(), n3 = std::max(n1, n2);
        std::vector<size_t> s3(n3); 
        Shape8 sa, sb, sc;
        
        for (size_t i = 0; i < n3; i++) {
            size_t d1 = (i < n1) ? shape_[n1 - 1 - i] : 1;
            size_t d2 = (i < n2) ? other.shape()[n2 - 1 - i] : 1;
            
            if (d1 != d2 && d1 != 1 && d2 != 1) {
                throw std::runtime_error("Broadcasting failed");
            }
            
            s3[n3 - 1 - i] = std::max(d1, d2);
            sa[n3 - 1 - i] = d1; 
            sb[n3 - 1 - i] = d2; 
            sc[n3 - 1 - i] = s3[n3 - 1 - i];
        }
        
        sa.nDim = sb.nDim = sc.nDim = n3;
        Tensor result(s3);
        
        binaryOpKernel<DType><<<(result.size() + 255) / 256, 256>>>(
            data_.get(), 
            other.data(), 
            result.mutableData(), 
            n3, sa, sb, sc, op, result.size()
        );
        return result;
    }

    /**
     * @brief Computes inherently safe logical representations casting purely to Boolean arrays independently limits.
     * @return Tensor<bool> Extracted logical parameter representation cleanly isolated from underlying values.
     */
    Tensor<bool> predicate(
        const Tensor &other /*!< @param other Partner evaluation constraint */, 
        BinaryOp op /*!< @param op Logical execution truth constraint parameter map */
    ) const {
        size_t n1 = nDim(), n2 = other.nDim(), n3 = std::max(n1, n2);
        std::vector<size_t> s3(n3); 
        Shape8 sa, sb, sc;
        
        for (size_t i = 0; i < n3; i++) {
            size_t d1 = (i < n1) ? shape_[n1 - 1 - i] : 1;
            size_t d2 = (i < n2) ? other.shape()[n2 - 1 - i] : 1;
            
            if (d1 != d2 && d1 != 1 && d2 != 1) {
                throw std::runtime_error("Broadcasting failed");
            }
            
            s3[n3 - 1 - i] = std::max(d1, d2);
            sa[n3 - 1 - i] = d1; 
            sb[n3 - 1 - i] = d2; 
            sc[n3 - 1 - i] = s3[n3 - 1 - i];
        }
        
        sa.nDim = sb.nDim = sc.nDim = n3;
        Tensor<bool> result(s3);
        
        binaryPredicateOpKernel<DType><<<(result.size() + 255) / 256, 256>>>(
            data_.get(), 
            other.data(), 
            result.mutableData(), 
            n3, sa, sb, sc, op, result.size()
        );
        return result;
    }

    /**
     * @brief Dynamically translates universally structured standalone values binding deeply efficiently.
     */
    Tensor binary(
        DType other /*!< @param other Explicit primitive scalar */, 
        BinaryOp op /*!< @param op Target logical primitive map */, 
        int scalar_on_left = 0 /*!< @param scalar_on_left Specifies if scalar is the left operand */
    ) const {
        Tensor result(shape_.toVector());
        binaryScalarOpKernel<DType><<<(size_ + 255) / 256, 256>>>(
            data_.get(), 
            other, 
            result.mutableData(), 
            size_, 
            op, 
            scalar_on_left
        );
        return result;
    }

    /**
     * @brief Applies explicitly evaluated generic scalar comparisons completely abstracting into truths correctly.
     * @return Tensor<bool> Generated output array specifically targeting strict logical Boolean matrices.
     */
    Tensor<bool> predicate(
        DType other /*!< @param other Value absolute truth constraint scalar */, 
        BinaryOp op /*!< @param op Parameter exact representation execution constraint natively */, 
        int scalar_on_left = 0 /*!< @param scalar_on_left Specifies if scalar is the left operand */
    ) const {
        Tensor<bool> result(shape_.toVector());
        binaryScalarPredicateOpKernel<DType><<<(size_ + 255) / 256, 256>>>(
            data_.get(), 
            other, 
            result.mutableData(), 
            size_, 
            op, 
            scalar_on_left
        );
        return result;
    }

    Tensor operator+(const Tensor &o) const { return binary(o, BinaryOp::ADD); }
    Tensor operator-(const Tensor &o) const { return binary(o, BinaryOp::SUB); }
    Tensor operator*(const Tensor &o) const { return binary(o, BinaryOp::MUL); }
    Tensor operator/(const Tensor &o) const { return binary(o, BinaryOp::DIV); }
    Tensor operator%(const Tensor &o) const { return binary(o, BinaryOp::MOD); }

    // Predicate assignments capturing cleanly unadulterated logical comparisons reliably isolated.
    Tensor<bool> operator==(const Tensor &o) const { return predicate(o, BinaryOp::EQ); }
    Tensor<bool> operator!=(const Tensor &o) const { return predicate(o, BinaryOp::NE); }
    Tensor<bool> operator>(const Tensor &o) const { return predicate(o, BinaryOp::GT); }
    Tensor<bool> operator>=(const Tensor &o) const { return predicate(o, BinaryOp::GE); }
    Tensor<bool> operator<(const Tensor &o) const { return predicate(o, BinaryOp::LT); }
    Tensor<bool> operator<=(const Tensor &o) const { return predicate(o, BinaryOp::LE); }

    Tensor operator+(DType o) const { return binary(o, BinaryOp::ADD); }
    Tensor operator-(DType o) const { return binary(o, BinaryOp::SUB); }
    Tensor operator*(DType o) const { return binary(o, BinaryOp::MUL); }
    Tensor operator/(DType o) const { return binary(o, BinaryOp::DIV); }

    Tensor<bool> operator==(DType o) const { return predicate(o, BinaryOp::EQ); }
    Tensor<bool> operator!=(DType o) const { return predicate(o, BinaryOp::NE); }
    Tensor<bool> operator>(DType o) const { return predicate(o, BinaryOp::GT); }
    Tensor<bool> operator>=(DType o) const { return predicate(o, BinaryOp::GE); }
    Tensor<bool> operator<(DType o) const { return predicate(o, BinaryOp::LT); }
    Tensor<bool> operator<=(DType o) const { return predicate(o, BinaryOp::LE); }

    /**
     * @brief Handles robust reduction architectures capturing exact statistical arrays implicitly.
     */
    Tensor reduce(
        const std::vector<size_t> &axes /*!< @param axes Set of positional axes to collapse logically */, 
        ReductionOp op /*!< @param op Collapsing statistical logic primitive map cleanly safely */
    ) const {
        std::vector<bool> is_red(nDim(), false); 
        std::vector<size_t> res_s; 
        Shape8 map;
        
        for (auto a : axes) is_red[a] = true;
        
        for (size_t i = 0; i < nDim(); i++) {
            if (!is_red[i]) {
                map[i] = res_s.size(); 
                res_s.push_back(shape_[i]);
            } else {
                map[i] = std::numeric_limits<size_t>::max(); 
            }
        }
        
        if (res_s.empty()) {
            res_s.push_back(1);
        }
        
        Tensor result(res_s);
        
        reduceKernel<DType><<<(result.size() + 255) / 256, 256>>>(
            data_.get(),
            result.get(),
            map,
            shape_,
            result.shape(),
            nDim(),
            result.nDim(),
            result.size(),
            op
        );
        return result;
    }

    std::vector<size_t> _all_axes() const {
        std::vector<size_t> axes(nDim());
        for (size_t i = 0; i < nDim(); i++) axes[i] = i;
        return axes;
    }

    Tensor sum(const std::vector<size_t> &a) const { return reduce(a, ReductionOp::SUM); }
    Tensor mean(const std::vector<size_t> &a) const { return reduce(a, ReductionOp::MEAN); }
    Tensor max(const std::vector<size_t> &a) const { return reduce(a, ReductionOp::MAX); }
    Tensor min(const std::vector<size_t> &a) const { return reduce(a, ReductionOp::MIN); }
    Tensor norm_l1(const std::vector<size_t> &a) const { return reduce(a, ReductionOp::NORM_L1); }
    Tensor norm_l2(const std::vector<size_t> &a) const { return reduce(a, ReductionOp::NORM_L2); }

    Tensor sum(int dim = -1) const { return reduce(dim < 0 ? _all_axes() : std::vector<size_t>{(size_t)dim}, ReductionOp::SUM); }
    Tensor mean(int dim = -1) const { return reduce(dim < 0 ? _all_axes() : std::vector<size_t>{(size_t)dim}, ReductionOp::MEAN); }
    Tensor max(int dim = -1) const { return reduce(dim < 0 ? _all_axes() : std::vector<size_t>{(size_t)dim}, ReductionOp::MAX); }
    Tensor min(int dim = -1) const { return reduce(dim < 0 ? _all_axes() : std::vector<size_t>{(size_t)dim}, ReductionOp::MIN); }
    Tensor norm_l1(int dim = -1) const { return reduce(dim < 0 ? _all_axes() : std::vector<size_t>{(size_t)dim}, ReductionOp::NORM_L1); }
    Tensor norm_l2(int dim = -1) const { return reduce(dim < 0 ? _all_axes() : std::vector<size_t>{(size_t)dim}, ReductionOp::NORM_L2); }

    /**
     * @brief Binds mathematically inherently expansive tensor relationships efficiently securely.
     */
    Tensor kron(
        const Tensor &other /*!< @param other Operative scaling geometry capturing Cartesian bounds safely systematically */
    ) const {
        if (nDim() != other.nDim()) throw std::runtime_error("Kron dim mismatch");
        
        std::vector<size_t> rs(nDim()); 
        Shape8 sa, sb, sc;
        
        for (size_t i = 0; i < nDim(); i++) {
            rs[i] = shape_[i] * other.shape()[i];
            sa[i] = shape_[i]; 
            sb[i] = other.shape()[i]; 
            sc[i] = rs[i];
        }
        
        sa.nDim = sb.nDim = sc.nDim = nDim();
        Tensor result(rs);
        
        kronKernel<DType><<<(result.size() + 255) / 256, 256>>>(
            data_.get(),
            other.data(),
            result.get(),
            sa,
            sb,
            sc,
            nDim(),
            result.size()
        );
        return result;
    }

    /**
     * @brief Computes generalized batched dimensional Matrix Multiplications safely isolating buffers natively.
     */
    Tensor matmul(
        const Tensor &o /*!< @param o Structural mapping vector array natively capturing explicit parameter limits flawlessly */
    ) const {
        if (nDim() < 2 || o.nDim() < 2) {
            throw std::runtime_error("Matmul failure: not enough dimensions");
        }
        size_t M = shape_[nDim() - 2];
        size_t K1 = shape_[nDim() - 1];
        size_t K2 = o.shape()[o.nDim() - 2];
        size_t N = o.shape()[o.nDim() - 1];
        
        if (K1 != K2) {
            throw std::runtime_error("Matmul mismatch");
        }
        
        std::vector<size_t> rs;
        for (size_t i = 0; i < nDim() - 2; i++) {
            rs.push_back(shape_[i]);
        }
        rs.push_back(M);
        rs.push_back(N);
        
        Tensor res(rs);
        size_t bs = size_ / (M * K1);
        size_t batchSizeB = o.size() / (K2 * N);
        
        matmulKernel<DType><<<(res.size() + 255) / 256, 256>>>(
            data_.get(),
            o.data(),
            res.get(),
            M,
            K1,
            N,
            bs,
            batchSizeB
        );
        return res;
    }
};

using BoolTensor = Tensor<bool>;
using ByteTensor = Tensor<signed char>;
using ShortTensor = Tensor<short>;
using IntTensor = Tensor<int>;
using LongTensor = Tensor<long long>;
using BFloatTensor = Tensor<bfloat>;
using HalfTensor = Tensor<half>;
using FloatTensor = Tensor<float>;
using DoubleTensor = Tensor<double>;

#endif // CLASS_TENSOR
