#include "headers.cu"

#ifndef ADAM_KERNELS
#define ADAM_KERNELS


/**
 * @brief CUDA kernel for the Adam optimization algorithm.
 * This kernel updates the parameters based on the gradients, first moment (m), and second moment (v) estimates.
 * It also applies bias correction to m and v. The decay is applied to the gradients.
 *
 * @tparam DType Floating-point data type.
 * @param param Pointer to the parameter tensor.
 * @param grad Pointer to the gradient tensor.
 * @param m Pointer to the first moment tensor.
 * @param v Pointer to the second moment tensor.
 * @param size Size of the parameter tensor.
 * @param lr Learning rate.
 * @param beta1 Exponential decay rate for the first moment.
 * @param beta2 Exponential decay rate for the second moment.
 * @param epsilon Small constant for numerical stability.
 * @param decay Weight decay factor.
 * @param timeStep Current time step (iteration count) for bias correction.
 */
template <typename DType = float> __global__ void adamUpdateKernel(
    DType *param,
    DType *grad,
    DType *m,
    DType *v,
    int size,
    double lr,
    double beta1,
    double beta2,
    double epsilon,
    double decay,
    int timeStep
) {
    int paramIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (paramIdx >= size) {
        return;
    }

    using CT = ComputeType<DType>;
    CT g = (CT)grad[paramIdx] + (CT)decay * (CT)param[paramIdx];
    CT m_t = (CT)m[paramIdx] * (CT)beta1 + (CT)g * (CT)(1.0 - beta1);
    m_t /= ((CT)1.0 - pow((CT)beta1, (CT)timeStep)); // Bias correction for m
    CT v_t = (CT)v[paramIdx] * (CT)beta2 + (CT)g * (CT)g * (CT)(1.0 - (CT)beta2);
    v_t /= ((CT)1.0 - pow((CT)beta2, (CT)timeStep)); // Bias correction for v

    CT correction = (CT)lr * m_t / (sqrt(v_t) + (CT)epsilon);
    param[paramIdx] = (DType)((CT)param[paramIdx] - correction);
    m[paramIdx] = (DType)m_t;
    v[paramIdx] = (DType)v_t;
}


/**
 * @brief CUDA kernel for the AdamW optimization algorithm.
 * This kernel updates the parameters based on the gradients, first moment (m), and second moment (v) estimates.
 * It applies weight decay directly to the parameters and includes bias correction for m and v.
 * 
 * @tparam DType Floating-point data type.
 * @param param Pointer to the parameter tensor.
 * @param grad Pointer to the gradient tensor.
 * @param m Pointer to the first moment tensor.
 * @param v Pointer to the second moment tensor.
 * @param size Size of the parameter tensor.
 * @param lr Learning rate.
 * @param beta1 Exponential decay rate for the first moment.
 * @param beta2 Exponential decay rate for the second moment.
 * @param epsilon Small constant for numerical stability.
 * @param decay Weight decay factor.
 * @param timeStep Current time step (iteration count) for bias correction.
 */
template <typename DType = float> __global__ void adamWUpdateKernel(
    DType *param,
    DType *grad,
    DType *m,
    DType *v,
    int size,
    double lr,
    double beta1,
    double beta2,
    double epsilon,
    double decay,
    int timeStep
) {
    int paramIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (paramIdx >= size) {
        return;
    }

    using CT = ComputeType<DType>;
    CT g = (CT)grad[paramIdx];
    CT m_t = (CT)m[paramIdx] * (CT)beta1 + (CT)g * (CT)(1.0 - beta1);
    m_t /= ((CT)1.0 - pow((CT)beta1, (CT)timeStep)); // Bias correction for m
    CT v_t = (CT)v[paramIdx] * (CT)beta2 + (CT)g * (CT)g * (CT)(1.0 - (CT)beta2);
    v_t /= ((CT)1.0 - pow((CT)beta2, (CT)timeStep)); // Bias correction for v

    CT correction = (CT)lr * m_t / (sqrt(v_t) + (CT)epsilon);
    CT decayFactor = (CT)1.0 - (CT)decay * (CT)lr;
    param[paramIdx] = (DType)((CT)param[paramIdx] * decayFactor - correction);
    m[paramIdx] = (DType)m_t;
    v[paramIdx] = (DType)v_t;
}


#endif // ADAM_KERNELS