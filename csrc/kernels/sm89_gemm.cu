#include "sm89_gemm.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../utils/exception.hpp"

namespace deep_gemm::sm89_gemm {
namespace {

int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ int map_scale_row(int row, int rows, int scale_rows) {
    if (scale_rows == rows) {
        return row;
    }
    const int block_rows = (rows + scale_rows - 1) / scale_rows;
    const int idx = row / block_rows;
    return idx < scale_rows ? idx : (scale_rows - 1);
}

template <typename out_t>
__device__ __forceinline__ void store_out(out_t* ptr, float v);

template <>
__device__ __forceinline__ void store_out<float>(float* ptr, float v) {
    *ptr = v;
}

template <>
__device__ __forceinline__ void store_out<__nv_bfloat16>(__nv_bfloat16* ptr, float v) {
    *ptr = __float2bfloat16(v);
}

template <typename out_t, int TY, int TX, int BK>
__global__ void fp8_scaled_gemm_nt_tiled_kernel(const __nv_fp8_e4m3* __restrict__ a,
                                                const float* __restrict__ sfa,
                                                const __nv_fp8_e4m3* __restrict__ b,
                                                const float* __restrict__ sfb,
                                                const float* __restrict__ c,
                                                out_t* __restrict__ out,
                                                int64_t out_stride_m,
                                                int64_t out_stride_n,
                                                int m,
                                                int n,
                                                int k,
                                                int64_t a_stride_m,
                                                int64_t a_stride_k,
                                                int64_t b_stride_n,
                                                int64_t b_stride_k,
                                                int sfa_rows,
                                                int sfa_k_groups,
                                                int64_t sfa_stride_row,
                                                int64_t sfa_stride_k,
                                                int sfb_rows,
                                                int sfb_k_groups,
                                                int64_t sfb_stride_row,
                                                int64_t sfb_stride_k,
                                                int gran_k_a,
                                                int gran_k_b,
                                                bool has_c) {
    const int row = blockIdx.y * TY + threadIdx.y;
    const int col = blockIdx.x * TX + threadIdx.x;

    const int sfa_row = (row < m) ? map_scale_row(row, m, sfa_rows) : 0;
    const int sfb_row = (col < n) ? map_scale_row(col, n, sfb_rows) : 0;
    const float* sfa_row_ptr = sfa + static_cast<int64_t>(sfa_row) * sfa_stride_row;
    const float* sfb_row_ptr = sfb + static_cast<int64_t>(sfb_row) * sfb_stride_row;

    float acc = 0.0f;
    if (row < m && col < n) {
        for (int kk = 0; kk < k; ++kk) {
            const int ka = kk / gran_k_a;
            const int kb = kk / gran_k_b;
            const int ka_clamped = ka < sfa_k_groups ? ka : (sfa_k_groups - 1);
            const int kb_clamped = kb < sfb_k_groups ? kb : (sfb_k_groups - 1);
            const int64_t a_idx =
                static_cast<int64_t>(row) * a_stride_m + static_cast<int64_t>(kk) * a_stride_k;
            const int64_t b_idx =
                static_cast<int64_t>(col) * b_stride_n + static_cast<int64_t>(kk) * b_stride_k;
            const float av = static_cast<float>(a[a_idx]) *
                             sfa_row_ptr[static_cast<int64_t>(ka_clamped) * sfa_stride_k];
            const float bv = static_cast<float>(b[b_idx]) *
                             sfb_row_ptr[static_cast<int64_t>(kb_clamped) * sfb_stride_k];
            acc += av * bv;
        }
    }

    if (row < m && col < n) {
        const int64_t out_idx =
            static_cast<int64_t>(row) * out_stride_m + static_cast<int64_t>(col) * out_stride_n;
        if (has_c) {
            store_out(out + out_idx, acc + c[out_idx]);
        } else {
            store_out(out + out_idx, acc);
        }
    }
}

}  // namespace

void fp8_fp4_gemm_nt(const torch::Tensor& a,
                     const torch::Tensor& sfa,
                     const torch::Tensor& b,
                     const torch::Tensor& sfb,
                     const std::optional<torch::Tensor>& c,
                     const torch::Tensor& d,
                     int gran_k_a,
                     int gran_k_b) {
    DG_HOST_ASSERT(a.dim() == 2 and b.dim() == 2 and d.dim() == 2);
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat and sfb.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    const auto m = static_cast<int>(a.size(0));
    const auto k = static_cast<int>(a.size(1));
    const auto n = static_cast<int>(b.size(0));
    DG_HOST_ASSERT(static_cast<int>(b.size(1)) == k);
    DG_HOST_ASSERT(static_cast<int>(d.size(0)) == m and static_cast<int>(d.size(1)) == n);
    DG_HOST_ASSERT(gran_k_a > 0 and gran_k_b > 0);

    auto sfa_f = sfa.contiguous();
    auto sfb_f = sfb.contiguous();
    torch::Tensor c_f;
    if (c.has_value()) {
        DG_HOST_ASSERT(c.value().scalar_type() == torch::kFloat);
        c_f = c.value().contiguous();
    }

    DG_HOST_ASSERT(sfa_f.dim() == 2 or sfa_f.dim() == 1);
    DG_HOST_ASSERT(sfb_f.dim() == 2 or sfb_f.dim() == 1);
    if (sfa_f.dim() == 1) {
        const int k_groups_a = ceil_div_int(k, gran_k_a);
        DG_HOST_ASSERT(static_cast<int>(sfa_f.size(0)) % k_groups_a == 0);
        sfa_f = sfa_f.view({sfa_f.size(0) / k_groups_a, k_groups_a});
    }
    if (sfb_f.dim() == 1) {
        const int k_groups_b = ceil_div_int(k, gran_k_b);
        DG_HOST_ASSERT(static_cast<int>(sfb_f.size(0)) % k_groups_b == 0);
        sfb_f = sfb_f.view({sfb_f.size(0) / k_groups_b, k_groups_b});
    }
    const int sfa_rows = static_cast<int>(sfa_f.size(0));
    const int sfb_rows = static_cast<int>(sfb_f.size(0));
    const int sfa_k_groups = static_cast<int>(sfa_f.size(1));
    const int sfb_k_groups = static_cast<int>(sfb_f.size(1));
    DG_HOST_ASSERT(sfa_rows > 0 and sfb_rows > 0);
    DG_HOST_ASSERT(sfa_k_groups > 0 and sfb_k_groups > 0);

    constexpr int TX = 16;
    constexpr int TY = 16;
    constexpr int BK = 32;
    const dim3 block(TX, TY);
    const dim3 grid(ceil_div_int(n, TX), ceil_div_int(m, TY));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (d.scalar_type() == torch::kFloat) {
        fp8_scaled_gemm_nt_tiled_kernel<float, TY, TX, BK><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(a.data_ptr()),
        sfa_f.data_ptr<float>(),
        reinterpret_cast<const __nv_fp8_e4m3*>(b.data_ptr()),
        sfb_f.data_ptr<float>(),
        c.has_value() ? c_f.data_ptr<float>() : nullptr,
        d.data_ptr<float>(),
        d.stride(0),
        d.stride(1),
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        sfa_rows,
        sfa_k_groups,
        sfa_f.stride(0),
        sfa_f.stride(1),
        sfb_rows,
        sfb_k_groups,
        sfb_f.stride(0),
        sfb_f.stride(1),
        gran_k_a,
        gran_k_b,
        c.has_value());
    } else {
        fp8_scaled_gemm_nt_tiled_kernel<__nv_bfloat16, TY, TX, BK><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(a.data_ptr()),
        sfa_f.data_ptr<float>(),
        reinterpret_cast<const __nv_fp8_e4m3*>(b.data_ptr()),
        sfb_f.data_ptr<float>(),
        c.has_value() ? c_f.data_ptr<float>() : nullptr,
        reinterpret_cast<__nv_bfloat16*>(d.data_ptr<at::BFloat16>()),
        d.stride(0),
        d.stride(1),
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        sfa_rows,
        sfa_k_groups,
        sfa_f.stride(0),
        sfa_f.stride(1),
        sfb_rows,
        sfb_k_groups,
        sfb_f.stride(0),
        sfb_f.stride(1),
        gran_k_a,
        gran_k_b,
        c.has_value());
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

}  // namespace deep_gemm::sm89_gemm

