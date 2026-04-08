#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

typedef nv_bfloat16 bfloat16_t;
typedef nv_bfloat16 bfloat;
typedef half float16_t;

#ifndef HEADERS
#define HEADERS

/**
 * @brief Enumeration of the supported element-wise unary operations.
 */
enum class UnaryOp { NEG, NOT, INV, EXP, LOG, SQR, SQRT };

/**
 * @brief Enumeration of supported element-wise binary operations.
 */
enum class BinaryOp { ADD, SUB, MUL, DIV, MOD, POW, EQ, NE, GT, GE, LT, LE, AND, OR, XOR };

/**
 * @brief Enumeration of supported tensor reduction operations.
 */
enum class ReductionOp { SUM, MEAN, MAX, MIN, NORM_L1, NORM_L2 };

/**
 * @brief Fixed-size shape structure for CUDA kernels avoiding dynamic array passing.
 * Extensively used to pass dimensionality metadata up to 8 dimensions directly by value.
 */
struct Shape8 {
    size_t dims[8];
    size_t nDim;

    /**
     * @brief Instantiates a blank shape initialized inherently empty to dimensions.
     */
    __host__ __device__ Shape8() : nDim(0) {
        for (int i = 0; i < 8; ++i) dims[i] = 0;
    }

    /**
     * @brief Translates an explicit dynamic standard layout immediately to fixed arrays safely.
     * @param v Source vector holding sequential dimensional elements up to size 8.
     */
    __host__ Shape8(const std::vector<size_t>& v) {
        if (v.size() > 8) {
            throw std::invalid_argument("Tensor does not support more than 8 dimensions.");
        }
        nDim = v.size();
        for (size_t i = 0; i < 8; ++i) dims[i] = (i < nDim) ? v[i] : 0;
    }

    /**
     * @brief Quick access indexing evaluating instantly without bounds mapping natively.
     * @param i Zero-based raw geometric dimension offset lookup.
     * @return size_t Immediate corresponding bounds parameter at that scale.
     */
    __host__ __device__ size_t operator[](size_t i) const { return dims[i]; }
    
    /**
     * @brief Reference indexor accessing strictly immutable bounds for modifications natively.
     * @param i Specified bounds modification dimension.
     * @return size_t& Raw structural reference enabling bounds alteration.
     */
    __host__ __device__ size_t& operator[](size_t i) { return dims[i]; }
    
    /**
     * @brief Decouples explicitly back into dynamically scalable C++ standard vectors.
     * @return std::vector<size_t> Output extracted array replicating current bounds geometry.
     */
    __host__ std::vector<size_t> toVector() const {
        /**
         * @brief Execute v operation.
         */
        std::vector<size_t> v(nDim);
        for (size_t i = 0; i < nDim; ++i) v[i] = dims[i];
        return v;
    }
};

/**
 * @brief Represents a single dimension's index or range during slicing operations comprehensively.
 */
struct IndexRange {
    size_t start = 0;
    size_t end = 0;
    bool is_range = false;
    bool is_all = false;

    /**
     * @brief Explicit generation yielding an overarching wildcard dimension completely.
     */
    IndexRange() : is_range(true), is_all(true) {}
    
    /**
     * @brief Extensively restricts geometric representations collapsing to an exact scalar.
     * @param i Target parameter coordinate index fixing spatial bounds natively.
     */
    IndexRange(size_t i) : start(i), end(i + 1), is_range(false), is_all(false) {}
    
    /**
     * @brief Explicitly restricts an intersecting continuous span bounds isolating natively subsets.
     * @param s Foundational starting position bounds inclusive logically.
     * @param e Terminating terminating bounds bound exclusively efficiently.
     */
    IndexRange(size_t s, size_t e) : start(s), end(e), is_range(true), is_all(false) {}
};

using TensorView = std::vector<IndexRange>;
inline const IndexRange _ = IndexRange();

/**
 * @brief CUDA kernel to fill a tensor with a constant value natively.
 * 
 * @param data  Device pointer to the allocated tensor data.
 * @param size  Total size of the tensor memory to fill.
 * @param value The scalar value to fill the tensor with uniformly.
 */
template <typename DType>
/**
 * @brief Execute fillKernel operation.
 */
__global__ void fillKernel(DType *data, size_t size, DType value) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

/**
 * @brief CUDA kernel to cast a tensor to a different data type natively.
 * 
 * @param oldData  Device pointer to the allocated tensor data.
 * @param newData  Device pointer to the allocated tensor data.
 * @param size  Total size of the tensor memory to fill.
 */
template <typename DType, typename CastType>
/**
 * @brief Execute castKernel operation.
 */
__global__ void castKernel(const DType *oldData, CastType *newData, size_t size) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < size) {
        newData[idx] = static_cast<CastType>(static_cast<float>(oldData[idx]));
    }
}


/**
 * @brief Allocates a shared pointer to a CUDA-managed memory buffer dynamically.
 * 
 * @param size Total number of elements to allocate and fill.
 * @param val  Initial value for all elements in the allocated buffer.
 * @return std::shared_ptr<DType[]> Device-managed shared pointer ensuring lifespan.
 */
template <typename DType> 
std::shared_ptr<DType[]> cudaMakeShared(size_t size, DType val = 0) {
    if (size == 0) return nullptr;

    DType *raw_data = nullptr;
    cudaError_t err = cudaMalloc(&raw_data, size * sizeof(DType));
    
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
    }

    fillKernel<DType><<<(size + 255) / 256, 256>>>(raw_data, size, val);
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("fillKernel failed: ") + cudaGetErrorString(err));
    }
    
    return std::shared_ptr<DType[]>(raw_data, [](DType *ptr) { cudaFree(ptr); });
}

/**
 * @brief Creates a deep copy of a CUDA memory buffer natively mapped on the device.
 *
 * @param pointerToClone The source shared pointer capturing GPU mapped domains.
 * @param size Total number of elements to completely encapsulate into the newly cloned mapping.
 * @return std::shared_ptr<DType[]> Cloned device-managed shared pointer ensuring memory lifespan.
 */
template <typename DType>
std::shared_ptr<DType[]> cudaCloneShared(std::shared_ptr<DType[]> pointerToClone, size_t size) {
    if (size == 0) return nullptr;

    DType *raw_data = nullptr;
    cudaError_t err = cudaMalloc(&raw_data, size * sizeof(DType));

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
    }

    cudaMemcpy(raw_data, pointerToClone.get(), size * sizeof(DType), cudaMemcpyDeviceToDevice);
    return std::shared_ptr<DType[]>(raw_data, [](DType *ptr) { cudaFree(ptr); });
}

/**
 * @brief CUDA kernel for in-place SGD parameter update: params[i] -= lr * grads[i].
 *
 * @param params Device pointer to the parameter buffer to be updated.
 * @param grads  Device pointer to the gradient buffer.
 * @param lr     Learning rate scalar.
 * @param size   Total number of elements.
 */
template <typename DType>
/**
 * @brief Execute sgdUpdateKernel operation.
 */
__global__ void sgdUpdateKernel(DType *params, const DType *grads, DType lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= lr * grads[idx];
    }
}

/**
 * @brief Launches sgdUpdateKernel for a single parameter/gradient buffer pair.
 *
 * @param params Device shared pointer to the parameter buffer.
 * @param grads  Device shared pointer to the gradient buffer.
 * @param lr     Learning rate.
 * @param size   Number of elements.
 */
template <typename DType>
/**
 * @brief Execute runSgdUpdate operation.
 */
void runSgdUpdate(std::shared_ptr<DType[]> params, std::shared_ptr<DType[]> grads, DType lr, int size) {
    if (params == nullptr || grads == nullptr || size == 0) return;
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    sgdUpdateKernel<DType><<<gridSize, blockSize>>>(params.get(), grads.get(), lr, size);
}

#endif // HEADERS