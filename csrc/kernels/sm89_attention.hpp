#pragma once

#include <torch/extension.h>

namespace deep_gemm::sm89_attention {

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
                           int head_dim);

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
                                 int block_table_stride);

}  // namespace deep_gemm::sm89_attention
