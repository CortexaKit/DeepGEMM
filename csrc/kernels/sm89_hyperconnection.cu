#include "sm89_hyperconnection.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace deep_gemm::sm89_hyperconnection {
namespace {

__global__ void tf32_hc_prenorm_gemm_kernel(const __nv_bfloat16* __restrict__ a,
                                            const float* __restrict__ b,
                                            float* __restrict__ d,
                                            int m,
                                            int n,
                                            int k,
                                            int split_idx,
                                            int num_splits,
                                            int64_t a_stride_m,
                                            int64_t a_stride_k,
                                            int64_t b_stride_n,
                                            int64_t b_stride_k,
                                            int64_t d_stride_split,
                                            int64_t d_stride_m,
                                            int64_t d_stride_n) {
    const int row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n) {
        return;
    }

    const int k_begin = (k * split_idx) / num_splits;
    const int k_end = (k * (split_idx + 1)) / num_splits;

    // Kahan-style compensated accumulation for split-k partial sums.
    float acc = 0.0f;
    float c = 0.0f;
    for (int kk = k_begin; kk < k_end; ++kk) {
        const float av = __bfloat162float(a[row * a_stride_m + kk * a_stride_k]);
        const float bv = b[col * b_stride_n + kk * b_stride_k];
        const float prod = av * bv;
        const float y = prod - c;
        const float t = acc + y;
        c = (t - acc) - y;
        acc = t;
    }

    d[split_idx * d_stride_split + row * d_stride_m + col * d_stride_n] = acc;
}

__global__ void tf32_hc_prenorm_sqrsum_kernel(const __nv_bfloat16* __restrict__ a,
                                              float* __restrict__ sqr_sum,
                                              int m,
                                              int k,
                                              int split_idx,
                                              int num_splits,
                                              int64_t a_stride_m,
                                              int64_t a_stride_k,
                                              int64_t sqr_stride_split,
                                              int64_t sqr_stride_m) {
    const int row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const int tid = threadIdx.x;
    const int k_begin = (k * split_idx) / num_splits;
    const int k_end = (k * (split_idx + 1)) / num_splits;

    float partial = 0.0f;
    for (int kk = k_begin + tid; kk < k_end; kk += blockDim.x) {
        const float av = __bfloat162float(a[row * a_stride_m + kk * a_stride_k]);
        partial += av * av;
    }

    __shared__ float red[256];
    red[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sqr_sum[split_idx * sqr_stride_split + row * sqr_stride_m] = red[0];
    }
}

}  // namespace

void launch_tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                 const torch::Tensor& b,
                                 const torch::Tensor& d,
                                 const torch::Tensor& sqr_sum,
                                 int num_splits) {
    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b.size(0);

    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    constexpr int kThreadsN = 128;
    const dim3 block_n(kThreadsN);
    const dim3 grid_n(static_cast<unsigned>(m),
                      static_cast<unsigned>((n + kThreadsN - 1) / kThreadsN));

    constexpr int kReduceThreads = 256;
    const dim3 block_reduce(kReduceThreads);
    const dim3 grid_reduce(static_cast<unsigned>(m));

    // Launch on current stream to keep producer/consumer ordering correct.
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* a_ptr = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr());
    const auto* b_ptr = b.data_ptr<float>();
    auto* d_ptr = d.data_ptr<float>();
    auto* sqr_ptr = sqr_sum.data_ptr<float>();

    const int64_t d_stride_split = (d.dim() == 3) ? d.stride(0) : 0;
    const int64_t d_stride_m = (d.dim() == 3) ? d.stride(1) : d.stride(0);
    const int64_t d_stride_n = (d.dim() == 3) ? d.stride(2) : d.stride(1);

    const int64_t sqr_stride_split = (sqr_sum.dim() == 2) ? sqr_sum.stride(0) : 0;
    const int64_t sqr_stride_m = (sqr_sum.dim() == 2) ? sqr_sum.stride(1) : sqr_sum.stride(0);

    for (int split_idx = 0; split_idx < num_splits; ++split_idx) {
        tf32_hc_prenorm_gemm_kernel<<<grid_n, block_n, 0, stream>>>(
            a_ptr,
            b_ptr,
            d_ptr,
            static_cast<int>(m),
            static_cast<int>(n),
            static_cast<int>(k),
            split_idx,
            num_splits,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            d_stride_split,
            d_stride_m,
            d_stride_n);

        tf32_hc_prenorm_sqrsum_kernel<<<grid_reduce, block_reduce, 0, stream>>>(
            a_ptr,
            sqr_ptr,
            static_cast<int>(m),
            static_cast<int>(k),
            split_idx,
            num_splits,
            a.stride(0),
            a.stride(1),
            sqr_stride_split,
            sqr_stride_m);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "sm89 tf32_hc_prenorm_gemm kernel launch failed");
}

}  // namespace deep_gemm::sm89_hyperconnection

