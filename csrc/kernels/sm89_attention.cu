#include "sm89_attention.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace deep_gemm::sm89_attention {

namespace {

template <typename T>
__device__ __forceinline__ void store_out(T* ptr, float v);

template <>
__device__ __forceinline__ void store_out<float>(float* ptr, float v) {
    *ptr = v;
}

template <>
__device__ __forceinline__ void store_out<__nv_bfloat16>(__nv_bfloat16* ptr, float v) {
    *ptr = __float2bfloat16(v);
}

template <typename out_t, int kHeadDim, int kThreads = 128, int kBlockN = 128>
__global__ void fp8_mqa_logits_kernel(const __nv_fp8_e4m3* __restrict__ q,
                                      const __nv_fp8_e4m3* __restrict__ kv,
                                      const float* __restrict__ kv_scales,
                                      const float* __restrict__ weights,
                                      const int* __restrict__ cu_start,
                                      const int* __restrict__ cu_end,
                                      out_t* __restrict__ logits,
                                      int seq_len,
                                      int seq_len_kv,
                                      int max_seqlen_k,
                                      int stride_logits,
                                      int num_heads,
                                      int q_stride_m,
                                      int q_stride_h,
                                      int kv_stride,
                                      int weight_stride) {
    const int m = blockIdx.x;
    if (m >= seq_len) return;

    const int logical_n = (max_seqlen_k > 0) ? max_seqlen_k : seq_len_kv;
    const int n = blockIdx.y * kBlockN + threadIdx.x;
    if (n >= logical_n) return;

    int kv_idx = n;
    if (max_seqlen_k > 0) {
        kv_idx = cu_start[m] + n;
        if (kv_idx < 0 || kv_idx >= cu_end[m] || kv_idx >= seq_len_kv) {
            store_out(logits + static_cast<int64_t>(m) * stride_logits + n, 0.0f);
            return;
        }
    } else if (kv_idx >= seq_len_kv) {
        store_out(logits + static_cast<int64_t>(m) * stride_logits + n, 0.0f);
        return;
    }

    const __nv_fp8_e4m3* q_row = q + static_cast<int64_t>(m) * q_stride_m;
    const __nv_fp8_e4m3* kv_row = kv + static_cast<int64_t>(kv_idx) * kv_stride;
    const float kv_scale = kv_scales[kv_idx];
    const float* w_row = weights + static_cast<int64_t>(m) * weight_stride;

    float acc = 0.0f;
    for (int h = 0; h < num_heads; ++h) {
        float dot = 0.0f;
        const int q_offset = h * q_stride_h;
        #pragma unroll
        for (int d = 0; d < kHeadDim; ++d) {
            dot += static_cast<float>(q_row[q_offset + d]) * (static_cast<float>(kv_row[d]) * kv_scale);
        }
        if (dot > 0.0f) acc += dot * w_row[h];
    }

    store_out(logits + static_cast<int64_t>(m) * stride_logits + n, acc);
}

template <typename out_t, int kHeadDim, int kThreads = 128, int kBlockN = 128>
__global__ void fp8_paged_mqa_logits_kernel(
    const __nv_fp8_e4m3* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ kv_cache,
    const float* __restrict__ kv_scales,
    const float* __restrict__ weights,
    const int* __restrict__ context_lens,
    const int* __restrict__ block_table,
    out_t* __restrict__ logits,
    int batch_size,
    int next_n,
    int num_heads,
    int block_kv,
    bool is_context_lens_2d,
    int logits_stride,
    int block_table_stride,
    int context_lens_stride,
    int q_stride_b,
    int q_stride_n,
    int q_stride_h,
    int kv_stride_block,
    int kv_stride_k,
    int kv_sf_stride_block,
    int weight_stride_row) {
    const int row = blockIdx.x;
    const int total_rows = batch_size * next_n;
    if (row >= total_rows) return;

    const int b = row / next_n;
    const int n_step = row - b * next_n;
    const int col = blockIdx.y * kBlockN + threadIdx.x;
    if (col >= logits_stride) return;

    const int ctx_len = is_context_lens_2d ? context_lens[b * context_lens_stride + n_step] : context_lens[b];
    if (col >= ctx_len) {
        store_out(logits + static_cast<int64_t>(row) * logits_stride + col, 0.0f);
        return;
    }

    const int kv_block_rank = col / block_kv;
    const int kv_inner = col - kv_block_rank * block_kv;
    const int kv_block_id = block_table[b * block_table_stride + kv_block_rank];
    const __nv_fp8_e4m3* kv_ptr =
        kv_cache + static_cast<int64_t>(kv_block_id) * kv_stride_block + static_cast<int64_t>(kv_inner) * kv_stride_k;
    const float kv_scale = kv_scales[static_cast<int64_t>(kv_block_id) * kv_sf_stride_block + kv_inner];

    const __nv_fp8_e4m3* q_row = q + static_cast<int64_t>(b) * q_stride_b + static_cast<int64_t>(n_step) * q_stride_n;
    const float* w_row = weights + static_cast<int64_t>(row) * weight_stride_row;

    float acc = 0.0f;
    for (int h = 0; h < num_heads; ++h) {
        float dot = 0.0f;
        const int q_offset = h * q_stride_h;
        #pragma unroll
        for (int d = 0; d < kHeadDim; ++d) {
            dot += static_cast<float>(q_row[q_offset + d]) * (static_cast<float>(kv_ptr[d]) * kv_scale);
        }
        if (dot > 0.0f) acc += dot * w_row[h];
    }

    store_out(logits + static_cast<int64_t>(row) * logits_stride + col, acc);
}

}  // namespace

void launch_fp8_mqa_logits(const torch::Tensor& q,
                           const torch::Tensor& kv,
                           const torch::Tensor& kv_scales,
                           const torch::Tensor& weights,
                           const torch::Tensor& cu_seq_len_k_start,
                           const torch::Tensor& cu_seq_len_k_end,
                           const torch::Tensor& logits,
                           int seq_len,
                           int seq_len_kv,
                           int max_seqlen_k,
                           int stride_logits,
                           int num_heads,
                           int head_dim) {
    constexpr int kThreads = 128;
    constexpr int kBlockN = 128;
    const int logical_n = (max_seqlen_k > 0) ? max_seqlen_k : seq_len_kv;
    dim3 grid(seq_len, (logical_n + kBlockN - 1) / kBlockN);
    dim3 block(kThreads);
    auto stream = at::cuda::getDefaultCUDAStream();

    const auto* q_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr());
    const auto* kv_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(kv.data_ptr());
    const auto* sf_ptr = kv_scales.data_ptr<float>();
    const auto* w_ptr = weights.data_ptr<float>();
    const auto* start_ptr = cu_seq_len_k_start.data_ptr<int>();
    const auto* end_ptr = cu_seq_len_k_end.data_ptr<int>();

    const bool is_fp32 = logits.scalar_type() == torch::kFloat32;
    #define LAUNCH_MQA(HD) \
        if (is_fp32) { \
            auto* out_ptr = logits.data_ptr<float>(); \
            fp8_mqa_logits_kernel<float, HD, kThreads, kBlockN><<<grid, block, 0, stream>>>( \
                q_ptr, kv_ptr, sf_ptr, w_ptr, start_ptr, end_ptr, out_ptr, seq_len, seq_len_kv, \
                max_seqlen_k, stride_logits, num_heads, static_cast<int>(q.stride(0)), \
                static_cast<int>(q.stride(1)), static_cast<int>(kv.stride(0)), static_cast<int>(weights.stride(0))); \
        } else { \
            auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()); \
            fp8_mqa_logits_kernel<__nv_bfloat16, HD, kThreads, kBlockN><<<grid, block, 0, stream>>>( \
                q_ptr, kv_ptr, sf_ptr, w_ptr, start_ptr, end_ptr, out_ptr, seq_len, seq_len_kv, \
                max_seqlen_k, stride_logits, num_heads, static_cast<int>(q.stride(0)), \
                static_cast<int>(q.stride(1)), static_cast<int>(kv.stride(0)), static_cast<int>(weights.stride(0))); \
        }
    if (head_dim == 128) {
        LAUNCH_MQA(128)
    } else if (head_dim == 64) {
        LAUNCH_MQA(64)
    } else {
        LAUNCH_MQA(32)
    }
    #undef LAUNCH_MQA
}

void launch_fp8_paged_mqa_logits(const torch::Tensor& q,
                                 const torch::Tensor& kv_cache,
                                 const torch::Tensor& kv_cache_scales,
                                 const torch::Tensor& weights,
                                 const torch::Tensor& context_lens,
                                 const torch::Tensor& logits,
                                 const torch::Tensor& block_table,
                                 int batch_size,
                                 int next_n,
                                 int num_heads,
                                 int head_dim,
                                 int block_kv,
                                 bool is_context_lens_2d,
                                 int logits_stride,
                                 int block_table_stride) {
    constexpr int kThreads = 128;
    constexpr int kBlockN = 128;
    const int total_rows = batch_size * next_n;
    dim3 grid(total_rows, (logits_stride + kBlockN - 1) / kBlockN);
    dim3 block(kThreads);
    auto stream = at::cuda::getDefaultCUDAStream();

    const auto* q_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr());
    const auto* kv_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(kv_cache.data_ptr());
    const auto* sf_ptr = kv_cache_scales.data_ptr<float>();
    const auto* w_ptr = weights.data_ptr<float>();
    const auto* ctx_ptr = context_lens.data_ptr<int>();
    const auto* bt_ptr = block_table.data_ptr<int>();

    const bool is_fp32 = logits.scalar_type() == torch::kFloat32;
    #define LAUNCH_PAGED(HD) \
        if (is_fp32) { \
            auto* out_ptr = logits.data_ptr<float>(); \
            fp8_paged_mqa_logits_kernel<float, HD, kThreads, kBlockN><<<grid, block, 0, stream>>>( \
                q_ptr, kv_ptr, sf_ptr, w_ptr, ctx_ptr, bt_ptr, out_ptr, batch_size, next_n, num_heads, \
                block_kv, is_context_lens_2d, logits_stride, block_table_stride, static_cast<int>(context_lens.stride(0)), \
                static_cast<int>(q.stride(0)), static_cast<int>(q.stride(1)), static_cast<int>(q.stride(2)), \
                static_cast<int>(kv_cache.stride(0)), static_cast<int>(kv_cache.stride(1)), \
                static_cast<int>(kv_cache_scales.stride(0)), static_cast<int>(weights.stride(0))); \
        } else { \
            auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()); \
            fp8_paged_mqa_logits_kernel<__nv_bfloat16, HD, kThreads, kBlockN><<<grid, block, 0, stream>>>( \
                q_ptr, kv_ptr, sf_ptr, w_ptr, ctx_ptr, bt_ptr, out_ptr, batch_size, next_n, num_heads, \
                block_kv, is_context_lens_2d, logits_stride, block_table_stride, static_cast<int>(context_lens.stride(0)), \
                static_cast<int>(q.stride(0)), static_cast<int>(q.stride(1)), static_cast<int>(q.stride(2)), \
                static_cast<int>(kv_cache.stride(0)), static_cast<int>(kv_cache.stride(1)), \
                static_cast<int>(kv_cache_scales.stride(0)), static_cast<int>(weights.stride(0))); \
        }
    if (head_dim == 128) {
        LAUNCH_PAGED(128)
    } else if (head_dim == 64) {
        LAUNCH_PAGED(64)
    } else {
        LAUNCH_PAGED(32)
    }
    #undef LAUNCH_PAGED
}

}  // namespace deep_gemm::sm89_attention
