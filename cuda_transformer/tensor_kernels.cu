#include "headers.cu"

#ifndef TENSOR_KERNELS
#define TENSOR_KERNELS

/**
 * @brief Read or write an arbitrary N-D slice of a tensor using precomputed strides.
 *
 * Maps each 1-D thread index to an N-D position in the slice, then translates it
 * to a flat offset in the parent tensor using precomputed per-dimension strides
 * (avoiding expensive in-kernel modulo chains on the full tensor shape).
 *
 * When is_set == 0 (get):  slice_data[i]             = tensor_data[tensor_offset]
 * When is_set == 1 (set):  tensor_data[tensor_offset] = slice_data[slice_offset]
 *   The set path computes slice_offset from the slice's own actual_shape to handle
 *   broadcasting and stride-alignment between the source slice and the destination view.
 *
 * Grid:
 *   gridDim.x = (total_elements + blockDim.x - 1) / blockDim.x
 * Block: (blockDim.x)  —  typically 256
 *
 * @tparam DType Element data type.
 *
 * @param tensor_data       Device pointer to the parent tensor buffer; shape is arbitrary.
 * @param slice_data        Device pointer to the slice buffer; flat size = total_elements.
 * @param base_offset       Flat offset into tensor_data where the slice window begins.
 * @param slice_shape       Logical shape of the slice (up to 8 dims); used for index decomposition.
 * @param tensor_strides    Per-dimension strides of the parent tensor within the slice window;
 *                          shape [slice_nDim].
 * @param slice_actual_shape Actual shape of the slice_data buffer (may differ from slice_shape
 *                           when broadcasting); shape [actual_nDim].
 * @param slice_nDim        Number of dimensions in slice_shape / tensor_strides.
 * @param actual_nDim       Number of dimensions in slice_actual_shape.
 * @param total_elements    Total number of elements in the slice (product of slice_shape).
 * @param is_set            0 = read slice from tensor; 1 = write slice into tensor.
 */
template <typename DType>
__global__ void sliceKernelFast(
    DType *tensor_data,
    DType *slice_data,
    size_t base_offset,
    Shape8 slice_shape,
    Shape8 tensor_strides,
    Shape8 slice_actual_shape,
    size_t slice_nDim,
    size_t actual_nDim,
    size_t total_elements,
    int is_set
) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= total_elements) return;

    size_t temp_i = i;
    size_t tensor_offset = base_offset;
    size_t view_coords[8] = {0};

    // Fast mapping of 1D thread index to ND memory offset explicitly lacking dynamic divergence
    for (int j = (int)slice_nDim - 1; j >= 0; j--) {
        view_coords[j] = temp_i % slice_shape[j];
        temp_i /= slice_shape[j];
        tensor_offset += view_coords[j] * tensor_strides[j];
    }

    if (is_set) {
        size_t slice_offset = 0;
        if (actual_nDim > 0) {
            size_t stride = 1;
            for (int k = (int)actual_nDim - 1; k >= 0; k--) {
                size_t d_s = slice_actual_shape[k];
                int s_j = (int)slice_nDim - 1 - ((int)actual_nDim - 1 - k);
                
                if (s_j >= 0 && s_j < (int)slice_nDim && d_s > 1) {
                    slice_offset += view_coords[s_j] * stride;
                }
                stride *= d_s;
            }
        }
        tensor_data[tensor_offset] = slice_data[slice_offset];
    } else {
        slice_data[i] = tensor_data[tensor_offset];
    }
}

/**
 * @brief General N-D tensor transpose (axis permutation).
 *
 * Each thread handles one element.  Given the 1-D source index i, the kernel
 * decomposes it into N-D source coordinates, applies the axis permutation perm,
 * recomputes the destination flat offset in the permuted layout, and writes
 * src[i] → dst[dst_offset].
 *
 * For a 2-D matrix this is a standard transpose; for higher-rank tensors it is
 * a general axis permutation (equivalent to numpy.transpose or torch.permute).
 *
 * Grid:
 *   gridDim.x = (total_elements + blockDim.x - 1) / blockDim.x
 * Block: (blockDim.x)  —  typically 256
 *
 * @tparam DType Element data type.
 *
 * @param src   Device pointer (read);  flat layout following shape [shape[0], shape[1], ..., shape[nDim-1]].
 * @param dst   Device pointer (write); flat layout following the permuted shape
 *              [shape[perm[0]], shape[perm[1]], ..., shape[perm[nDim-1]]].
 * @param shape Original per-dimension sizes of src; length nDim.
 * @param perm  Permutation vector: dst axis j draws from src axis perm[j]; length nDim.
 * @param nDim  Number of tensor dimensions (≤ 8).
 */
template <typename DType>
__global__ void transposeKernel(
    const DType *src,
    DType *dst,
    Shape8 shape,
    Shape8 perm,
    size_t nDim
) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t total_size = 1;
    for (size_t k = 0; k < nDim; k++) total_size *= shape[k];

    if (i >= total_size) return;
    
    if (nDim == 0) { 
        dst[0] = src[0]; 
        return; 
    }

    size_t src_strides[8], dst_strides[8], dst_shape[8];
    for (size_t k = 0; k < nDim; k++) {
        dst_shape[k] = shape[perm[k]];
        src_strides[k] = 0;
        dst_strides[k] = 0;
    }

    src_strides[nDim - 1] = 1;
    dst_strides[nDim - 1] = 1;
    
    for (int k = (int)nDim - 2; k >= 0; k--) {
        src_strides[k] = src_strides[k + 1] * shape[k + 1];
        dst_strides[k] = dst_strides[k + 1] * dst_shape[k + 1];
    }

    size_t temp_i = i, src_idx[8];
    for (int j = (int)nDim - 1; j >= 0; j--) {
        src_idx[j] = temp_i % shape[j];
        temp_i /= shape[j];
    }

    size_t dst_offset = 0;
    for (size_t j = 0; j < nDim; j++) {
        dst_offset += src_idx[perm[j]] * dst_strides[j];
    }

    dst[dst_offset] = src[i];
}

/**
 * @brief Device helper: applies a unary operator to a single scalar value.
 *
 * Supported operations (UnaryOp enum):
 *   NEG  — negate (-a); for bool: logical NOT
 *   NOT  — bitwise NOT for integral types; logical NOT otherwise
 *   INV  — reciprocal (1/a)
 *   EXP  — exp(a)
 *   LOG  — log(a)
 *   SQR  — a*a
 *   SQRT — sqrt(a)
 *
 * @tparam DType Element data type.
 * @param a  Input scalar.
 * @param op Unary operation to apply.
 * @return   Result scalar of the same type.
 */
template <typename DType>
__device__ DType applyUnary(DType a, UnaryOp op) {
    if constexpr (std::is_same_v<DType, bool>) {
        if (op == UnaryOp::NEG || op == UnaryOp::NOT) return !a;
        return a;
    } else {
        switch (op) {
            case UnaryOp::NEG: return -a;
            case UnaryOp::NOT: 
                if constexpr (std::is_integral_v<DType>) return ~a; 
                else return !a;
            case UnaryOp::INV: return (DType)1 / a;
            case UnaryOp::EXP: return static_cast<DType>(exp(static_cast<ComputeType<DType>>(a)));
            case UnaryOp::LOG: return static_cast<DType>(log(static_cast<ComputeType<DType>>(a)));
            case UnaryOp::SQR: return a * a;
            case UnaryOp::SQRT: return static_cast<DType>(sqrt(static_cast<ComputeType<DType>>(a)));
            default: return a;
        }
    }
}

/**
 * @brief Element-wise unary operation over a flat tensor buffer.
 *
 * Applies applyUnary(a[i], op) for every element in a flat array.
 *
 * Grid:
 *   gridDim.x = (size + blockDim.x - 1) / blockDim.x
 * Block: (blockDim.x)  —  typically 256
 *
 * @tparam DType Element data type.
 *
 * @param a    Device pointer (read);  flat buffer of length size.
 * @param c    Device pointer (write); flat buffer of length size.
 * @param size Total number of elements.
 * @param op   Unary operation to apply (NEG, NOT, INV, EXP, LOG, SQR, SQRT).
 */
template <typename DType>
__global__ void unaryOpKernel(const DType *a, DType *c, size_t size, UnaryOp op) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = applyUnary(a[idx], op);
    }
}

/**
 * @brief Device helper: applies a binary arithmetic operator to two scalar values.
 *
 * Supported operations (BinaryOp enum):
 *   ADD — a + b
 *   SUB — a - b
 *   MUL — a * b
 *   DIV — a / b
 *   MOD — a % b (integral types only; returns 0 for floating-point)
 *   POW — pow(a, b)
 *   AND — a && b (logical)
 *   OR  — a || b (logical)
 *   XOR — a ^ b (integral) or (a != b) (floating-point)
 *
 * @tparam DType Element data type.
 * @param a  Left operand.
 * @param b  Right operand.
 * @param op Binary operation to apply.
 * @return   Result scalar of the same type.
 */
template <typename DType>
__device__ DType applyBinary(DType a, DType b, BinaryOp op) {
    switch (op) {
        case BinaryOp::ADD: return a + b;
        case BinaryOp::SUB: return a - b;
        case BinaryOp::MUL: return a * b;
        case BinaryOp::DIV: return a / b;
        case BinaryOp::MOD: 
            if constexpr (std::is_integral_v<DType>) return a % b; 
            else return static_cast<DType>(0);
        case BinaryOp::POW: 
            return static_cast<DType>(pow(static_cast<ComputeType<DType>>(a), static_cast<ComputeType<DType>>(b)));
        case BinaryOp::AND: 
            return static_cast<DType>(a && b);
        case BinaryOp::OR:  
            return static_cast<DType>(a || b);
        case BinaryOp::XOR: 
            if constexpr (std::is_integral_v<DType>) return (DType)(a ^ b); 
            else return (DType)(a != b);
        default: return (DType)0;
    }
}

/**
 * @brief Device helper: evaluates a comparison predicate on two scalar values.
 *
 * Supported operations (BinaryOp enum):
 *   EQ — a == b
 *   NE — a != b
 *   GT — a >  b
 *   GE — a >= b
 *   LT — a <  b
 *   LE — a <= b
 *
 * @tparam DType Element data type.
 * @param a  Left operand.
 * @param b  Right operand.
 * @param op Comparison operation to evaluate.
 * @return   bool result of the comparison.
 */
template <typename DType>
__device__ bool applyPredicate(DType a, DType b, BinaryOp op) {
    switch (op) {
        case BinaryOp::EQ: return a == b;
        case BinaryOp::NE: return a != b;
        case BinaryOp::GT: return a > b;
        case BinaryOp::GE: return a >= b;
        case BinaryOp::LT: return a < b;
        case BinaryOp::LE: return a <= b;
        default: return false;
    }
}

/**
 * @brief Element-wise binary predicate over two broadcastable N-D tensors → bool output.
 *
 * For each output element c[i], decomposes the flat index into N-D coordinates using
 * shapeC, then maps each coordinate back to flat offsets in a and b respecting
 * broadcasting (a dimension of size 1 in shapeA/shapeB is broadcast).
 *
 * Grid:
 *   gridDim.x = (sizeC + blockDim.x - 1) / blockDim.x
 * Block: (blockDim.x)  —  typically 256
 *
 * @tparam DType Input element data type.
 *
 * @param a      Device pointer (read);  flat buffer shaped according to shapeA.
 * @param b      Device pointer (read);  flat buffer shaped according to shapeB.
 * @param c      Device pointer (write); flat bool buffer of length sizeC.
 * @param nDim   Number of broadcast dimensions (≤ 8).
 * @param shapeA Per-dimension sizes of a; size-1 dims are broadcast.
 * @param shapeB Per-dimension sizes of b; size-1 dims are broadcast.
 * @param shapeC Per-dimension sizes of c (broadcast-resolved output shape).
 * @param op     Comparison operation (EQ, NE, GT, GE, LT, LE).
 * @param sizeC  Total number of output elements (product of shapeC).
 */
template <typename DType>
__global__ void binaryPredicateOpKernel(
    const DType *a,
    const DType *b,
    bool *c,
    size_t nDim,
    Shape8 shapeA,
    Shape8 shapeB,
    Shape8 shapeC,
    BinaryOp op,
    size_t sizeC
) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < sizeC) {
        size_t temp_idx = idx, offA = 0, offB = 0, strA = 1, strB = 1;
        
        for (int i = (int)nDim - 1; i >= 0; i--) {
            size_t i_val = temp_idx % shapeC[i];
            temp_idx /= shapeC[i];
            
            if (shapeA[i] > 1) offA += i_val * strA;
            if (shapeB[i] > 1) offB += i_val * strB;
            
            strA *= shapeA[i];
            strB *= shapeB[i];
        }
        
        c[idx] = applyPredicate(a[offA], b[offB], op);
    }
}

/**
 * @brief Element-wise binary arithmetic over two broadcastable N-D tensors.
 *
 * Same broadcasting logic as binaryPredicateOpKernel, but writes an arithmetic
 * result of type DType instead of a bool.
 *
 * Grid:
 *   gridDim.x = (sizeC + blockDim.x - 1) / blockDim.x
 * Block: (blockDim.x)  —  typically 256
 *
 * @tparam DType Element data type.
 *
 * @param a      Device pointer (read);  flat buffer shaped according to shapeA.
 * @param b      Device pointer (read);  flat buffer shaped according to shapeB.
 * @param c      Device pointer (write); flat buffer of length sizeC.
 * @param nDim   Number of broadcast dimensions (≤ 8).
 * @param shapeA Per-dimension sizes of a; size-1 dims are broadcast.
 * @param shapeB Per-dimension sizes of b; size-1 dims are broadcast.
 * @param shapeC Per-dimension sizes of c (broadcast-resolved output shape).
 * @param op     Binary arithmetic operation (ADD, SUB, MUL, DIV, MOD, POW, AND, OR, XOR).
 * @param sizeC  Total number of output elements (product of shapeC).
 */
template <typename DType>
__global__ void binaryOpKernel(
    const DType *a,
    const DType *b,
    DType *c,
    size_t nDim,
    Shape8 shapeA,
    Shape8 shapeB,
    Shape8 shapeC,
    BinaryOp op,
    size_t sizeC
) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < sizeC) {
        size_t temp_idx = idx, offA = 0, offB = 0, strA = 1, strB = 1;
        
        for (int i = (int)nDim - 1; i >= 0; i--) {
            size_t i_val = temp_idx % shapeC[i];
            temp_idx /= shapeC[i];
            
            if (shapeA[i] > 1) offA += i_val * strA;
            if (shapeB[i] > 1) offB += i_val * strB;
            
            strA *= shapeA[i];
            strB *= shapeB[i];
        }
        
        c[idx] = applyBinary(a[offA], b[offB], op);
    }
}

/**
 * @brief Element-wise binary arithmetic between a tensor and a scalar.
 *
 * Applies applyBinary(a[i], b, op) or applyBinary(b, a[i], op) depending on
 * scalar_on_left, allowing both left-scalar (b OP a[i]) and right-scalar (a[i] OP b).
 *
 * Grid:
 *   gridDim.x = (size + blockDim.x - 1) / blockDim.x
 * Block: (blockDim.x)  —  typically 256
 *
 * @tparam DType Element data type.
 *
 * @param a              Device pointer (read);  flat buffer of length size.
 * @param b              Scalar value.
 * @param c              Device pointer (write); flat buffer of length size.
 * @param size           Total number of elements.
 * @param op             Binary arithmetic operation.
 * @param scalar_on_left 1 → compute b OP a[i];  0 → compute a[i] OP b.
 */
template <typename DType>
__global__ void binaryScalarOpKernel(
    const DType *a, 
    DType b, 
    DType *c, 
    size_t size, 
    BinaryOp op, 
    int scalar_on_left
) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < size) {
        if (scalar_on_left) {
            c[idx] = applyBinary(b, a[idx], op);
        } else {
            c[idx] = applyBinary(a[idx], b, op);
        }
    }
}

/**
 * @brief Element-wise comparison predicate between a tensor and a scalar → bool output.
 *
 * Applies applyPredicate(a[i], b, op) or applyPredicate(b, a[i], op) depending on
 * scalar_on_left.
 *
 * Grid:
 *   gridDim.x = (size + blockDim.x - 1) / blockDim.x
 * Block: (blockDim.x)  —  typically 256
 *
 * @tparam DType Element data type.
 *
 * @param a              Device pointer (read);  flat buffer of length size.
 * @param b              Scalar value to compare against.
 * @param c              Device pointer (write); flat bool buffer of length size.
 * @param size           Total number of elements.
 * @param op             Comparison operation (EQ, NE, GT, GE, LT, LE).
 * @param scalar_on_left 1 → evaluate b OP a[i];  0 → evaluate a[i] OP b.
 */
template <typename DType>
__global__ void binaryScalarPredicateOpKernel(
    const DType *a, 
    DType b, 
    bool *c, 
    size_t size, 
    BinaryOp op, 
    int scalar_on_left
) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < size) {
        if (scalar_on_left) {
            c[idx] = applyPredicate(b, a[idx], op);
        } else {
            c[idx] = applyPredicate(a[idx], b, op);
        }
    }
}

/**
 * @brief Computes the Kronecker (tensor) product of two N-D tensors.
 *
 * For output index i, decomposes into N-D coordinates v using shapeC, then:
 *   offA += (v[dim] / shapeB[dim]) * strideA[dim]   — selects the block in A
 *   offB += (v[dim] % shapeB[dim]) * strideB[dim]   — selects the element within that block
 *   c[i] = a[offA] * b[offB]
 *
 * shapeC[dim] == shapeA[dim] * shapeB[dim] for every dimension.
 *
 * Grid:
 *   gridDim.x = (sizeC + blockDim.x - 1) / blockDim.x
 * Block: (blockDim.x)  —  typically 256
 *
 * @tparam DType Element data type.
 *
 * @param a      Device pointer (read);  shape described by shapeA; flat size = product(shapeA).
 * @param b      Device pointer (read);  shape described by shapeB; flat size = product(shapeB).
 * @param c      Device pointer (write); shape described by shapeC; flat size = sizeC.
 * @param shapeA Per-dimension sizes of a (length nDim).
 * @param shapeB Per-dimension sizes of b (length nDim).
 * @param shapeC Per-dimension sizes of c = shapeA * shapeB element-wise (length nDim).
 * @param nDim   Number of tensor dimensions (≤ 8).
 * @param sizeC  Total number of output elements (product of shapeC).
 */
template <typename DType>
__global__ void kronKernel(
    const DType *a,
    const DType *b,
    DType *c,
    Shape8 shapeA,
    Shape8 shapeB,
    Shape8 shapeC,
    size_t nDim,
    size_t sizeC
) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < sizeC) {
        size_t temp_i = idx, offA = 0, offB = 0, strA = 1, strB = 1;
        
        for (int i = (int)nDim - 1; i >= 0; i--) {
            size_t v = temp_i % shapeC[i];
            temp_i /= shapeC[i];
            
            offA += (v / shapeB[i]) * strA;
            offB += (v % shapeB[i]) * strB;
            
            strA *= shapeA[i];
            strB *= shapeB[i];
        }
        
        c[idx] = a[offA] * b[offB];
    }
}

/**
 * @brief General N-D reduction along one or more axes of a tensor.
 *
 * Each output element dst[i] is the reduction of all source elements whose N-D
 * indices match the N-D index of dst[i] on the non-reduced axes.  The mapping
 * array encodes which destination dimension each source dimension maps to
 * (mapping[srcDim] == -1 means that dimension is being reduced away).
 *
 * Supported reduction operations (ReductionOp enum):
 *   SUM     — Σ values
 *   MEAN    — Σ values / count
 *   MAX     — maximum value
 *   MIN     — minimum value
 *   NORM_L1 — Σ |value|
 *   NORM_L2 — √(Σ value²)
 *
 * Note: this is a simple reference kernel (one thread per output element, full
 * source scan per thread) and is not optimised for large reductions.
 *
 * Grid:
 *   gridDim.x = (sizeDst + blockDim.x - 1) / blockDim.x
 * Block: (blockDim.x)  —  typically 256
 *
 * @tparam DType Element data type.
 *
 * @param src      Device pointer (read);  flat buffer; logical shape shapeSrc.
 * @param dst      Device pointer (write); flat buffer of length sizeDst; logical shape shapeDst.
 * @param mapping  For each source dimension, the index of the corresponding destination
 *                 dimension, or (size_t)-1 if that source dimension is reduced.  Length nDimSrc.
 * @param shapeSrc Per-dimension sizes of src (length nDimSrc).
 * @param shapeDst Per-dimension sizes of dst (length nDimDst).
 * @param nDimSrc  Number of dimensions in src.
 * @param nDimDst  Number of dimensions in dst.
 * @param sizeDst  Total number of output elements (product of shapeDst).
 * @param op       Reduction operation to apply.
 */
template <typename DType>
__global__ void reduceKernel(
    const DType *src,
    DType *dst,
    Shape8 mapping,
    Shape8 shapeSrc,
    Shape8 shapeDst,
    size_t nDimSrc,
    size_t nDimDst,
    size_t sizeDst,
    ReductionOp op
) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < sizeDst) {
        size_t dst_idx[8] = {0};
        size_t tmp = idx;
        
        for (int i = (int)nDimDst - 1; i >= 0; i--) {
            dst_idx[i] = tmp % shapeDst[i];
            tmp /= shapeDst[i];
        }
        
        DType res;
        bool first = true;
        size_t count = 0, totalSrc = 1;
        
        for (size_t i = 0; i < nDimSrc; i++) {
            totalSrc *= shapeSrc[i];
        }
        
        for (size_t s = 0; s < totalSrc; s++) {
            size_t tmp_s = s; 
            bool match = true;
            
            for (int i = (int)nDimSrc - 1; i >= 0; i--) {
                size_t val = tmp_s % shapeSrc[i];
                tmp_s /= shapeSrc[i];
                if (mapping[i] != (size_t)-1 && val != dst_idx[mapping[i]]) {
                    match = false; 
                    break;
                }
            }
            
            if (!match) continue;
            
            DType val = src[s];
            if (first) {
                if (op == ReductionOp::NORM_L1) res = (val < static_cast<DType>(0)) ? -val : val;
                else if (op == ReductionOp::NORM_L2) res = val * val;
                else res = val;
                first = false;
            } else {
                switch (op) {
                    case ReductionOp::SUM: 
                    case ReductionOp::MEAN: res += val; break;
                    case ReductionOp::MAX: res = (val > res) ? val : res; break;
                    case ReductionOp::MIN: res = (val < res) ? val : res; break;
                    case ReductionOp::NORM_L1: res += (val < static_cast<DType>(0)) ? -val : val; break;
                    case ReductionOp::NORM_L2: res += val * val; break;
                }
            }
            count++;
        }
        
        if (op == ReductionOp::MEAN && count > 0) res /= (DType)count;
        if (op == ReductionOp::NORM_L2) {
            res = static_cast<DType>(sqrt(static_cast<ComputeType<DType>>(res)));
        }
        dst[idx] = res;
    }
}

/**
 * @brief Batched matrix multiplication: C[b] = A[b] × B[b] (or B[0] when batchSizeB == 1).
 *
 * Computes C[b, m, n] = Σ_k A[b, m, k] * B[b_real, k, n], where b_real = b when
 * batchSizeB > 1 or 0 when batchSizeB == 1 (broadcast the single B matrix over all
 * batch elements in A).
 *
 * Each thread handles one output element identified by its flat index into C.
 * This is a naïve (non-tiled) implementation; suitable for small matmuls or as a
 * fallback.
 *
 * Grid:
 *   gridDim.x = (batchSize * M * N + blockDim.x - 1) / blockDim.x
 * Block: (blockDim.x)  —  typically 256
 *
 * @tparam DType Element data type.
 *
 * @param a         Device pointer (read);  shape [batchSize, M, K].
 * @param b         Device pointer (read);  shape [batchSizeB, K, N].
 *                  When batchSizeB == 1 the single matrix is broadcast.
 * @param c         Device pointer (write); shape [batchSize, M, N].
 * @param M         Number of rows in each output matrix.
 * @param K         Shared (contracted) dimension.
 * @param N         Number of columns in each output matrix.
 * @param batchSize Number of matrices in a (and c).
 * @param batchSizeB Number of matrices in b; pass 1 to broadcast a single B.
 */
template <typename DType>
__global__ void matmulKernel(
    const DType *a,
    const DType *b,
    DType *c,
    size_t M,
    size_t K,
    size_t N,
    size_t batchSize,
    size_t batchSizeB
) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < batchSize * M * N) {
        size_t b_idx = idx / (M * N);
        size_t mn = idx % (M * N);
        size_t m = mn / N;
        size_t n = mn % N;
        
        size_t b_real_idx = (batchSizeB == 1) ? 0 : b_idx;
        DType sum = 0;
        
        for (size_t k = 0; k < K; k++) {
            sum += a[b_idx * M * K + m * K + k] * b[b_real_idx * K * N + k * N + n];
        }
        
        c[idx] = sum;
    }
}

#endif // TENSOR_KERNELS