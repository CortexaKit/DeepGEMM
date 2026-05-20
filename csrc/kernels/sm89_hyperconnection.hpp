#pragma once

#include <optional>
#include <torch/extension.h>

namespace deep_gemm::sm89_hyperconnection {

void launch_tf32_hc_prenorm_gemm(const torch::Tensor& a,
                                 const torch::Tensor& b,
                                 const torch::Tensor& d,
                                 const torch::Tensor& sqr_sum,
                                 int num_splits);

}  // namespace deep_gemm::sm89_hyperconnection

