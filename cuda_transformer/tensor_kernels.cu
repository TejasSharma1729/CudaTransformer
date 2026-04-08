#include "headers.cu"

#ifndef TENSOR_KERNELS
#define TENSOR_KERNELS

/**
 * @brief Fast memory-mapping kernel to access or update arbitrary Tensor slices.
 * 
 * Takes precomputed strides directly from the CPU Host to drastically avoid modulo-based 
 * divergence operations in thousands of CUDA threads.
 * 
 * @param tensor_data     The original tensor's massive buffer in global memory.
 * @param slice_data      The subset data buffer (to read from, or write to).
 * @param base_offset     The computed foundational offset mapping.
 * @param slice_shape     The geometric dimensions of the slice region itself.
 * @param tensor_strides  Precomputed bounds constraints on the active view region.
 * @param slice_actual_shape Fallback layout geometry handling data injection scaling. 
 * @param slice_nDim      Amount of dimensions directly constrained by the mathematical slicing.
 * @param actual_nDim     Amount of intrinsic topological dimensions establishing bounds.
 * @param total_elements  Absolute boundary scale used to aggressively govern out of bounds.
 * @param is_set          Boolean directing tensor data modification behavior (true implies push).
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
 * @brief Memory translocation mapping to arbitrarily route tensor dimensions backward or forward.
 * 
 * @param src   Device memory source of the raw original tensor structure.
 * @param dst   Allocated device destination for geometrically re-projected memory maps.
 * @param shape Unmodified topological bounds block describing the pristine source data.
 * @param perm  Geometric ordering translation key dictating final absolute alignment.
 * @param nDim  Dimension size tracking variable necessary to perfectly bound permutations.
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
 * @brief Executes scalar-to-scalar unary transformations instantly on natively driven GPU memory.
 */
template <typename DType>
/**
 * @brief Execute applyUnary operation.
 */
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
            case UnaryOp::EXP: return static_cast<DType>(exp(static_cast<double>(a)));
            case UnaryOp::LOG: return static_cast<DType>(log(static_cast<double>(a)));
            case UnaryOp::SQR: return a * a;
            case UnaryOp::SQRT: return static_cast<DType>(sqrt(static_cast<double>(a)));
            default: return a;
        }
    }
}

/**
 * @brief Grid wrapper accelerating the parallel evaluation of mathematically pure unary operations.
 */
template <typename DType>
/**
 * @brief Execute unaryOpKernel operation.
 */
__global__ void unaryOpKernel(const DType *a, DType *c, size_t size, UnaryOp op) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = applyUnary(a[idx], op);
    }
}

/**
 * @brief Evaluates an inherent mathematical application for binary operations linking variables.
 */
template <typename DType>
/**
 * @brief Execute applyBinary operation.
 */
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
            return static_cast<DType>(pow(static_cast<double>(a), static_cast<double>(b)));
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
 * @brief Performs strict boolean logical execution mapping relationships accurately via truths.
 * @return Structurally guaranteed true or false mapping identical representations.
 */
template <typename DType>
/**
 * @brief Execute applyPredicate operation.
 */
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
 * @brief Unifies logical predicate evaluation incorporating rigorous parallel execution geometries.
 * Guarantees result is injected purely into Boolean arrays regardless of original input DType.
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
 * @brief Instantiates unified arithmetic sweeping incorporating dynamic topological broadcasting.
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
 * @brief Sweeps a standalone scalar mathematical arithmetic operation universally across kernels.
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
 * @brief Scans standalone scalars evaluating logical comparison expressions inherently to bool logic.
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
 * @brief Calculates the Kronecker tensor product geometrically linking inherently huge hierarchies.
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
 * @brief Aggregation mappings rigorously capturing deep sums, means, and normal distributions.
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
            res = static_cast<DType>(sqrt(static_cast<double>(res)));
        }
        dst[idx] = res;
    }
}

/**
 * @brief Secure general-purpose Matrix Multiplication directly evaluating unbatched RHS flows. 
 * Formidably prevents memory out-of-bounds crashes during generic tensor vector mappings.
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