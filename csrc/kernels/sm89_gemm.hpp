#pragma once

#include <optional>
#include <torch/extension.h>

namespace deep_gemm::sm89_gemm {

void fp8_fp4_gemm_nt(const torch::Tensor& a,
                     const torch::Tensor& sfa,
                     const torch::Tensor& b,
                     const torch::Tensor& sfb,
                     const std::optional<torch::Tensor>& c,
                     const torch::Tensor& d,
                     int gran_k_a,
                     int gran_k_b);

}  // namespace deep_gemm::sm89_gemm

