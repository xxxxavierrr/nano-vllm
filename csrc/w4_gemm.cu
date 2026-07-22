#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kInt4PerWord = 8;

template <typename scale_t, bool QuantizeActivation>
__global__ void packed_w4_gemm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ qweight,
    const scale_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ output,
    int m,
    int n,
    int k,
    int group_size) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || column >= n) {
    return;
  }

  float accumulator = 0.0f;
  for (int group_start = 0; group_start < k; group_start += group_size) {
    const int group = group_start / group_size;
    const float weight_scale = static_cast<float>(scales[group * n + column]);
    float activation_scale = 1.0f;
    if constexpr (QuantizeActivation) {
      float absmax = 0.0f;
      for (int offset = 0; offset < group_size; ++offset) {
        absmax = fmaxf(
            absmax,
            fabsf(__bfloat162float(x[row * k + group_start + offset])));
      }
      activation_scale = fmaxf(absmax / 127.0f, 1.0e-12f);
    }
    for (int offset = 0; offset < group_size; ++offset) {
      const int inner = group_start + offset;
      const uint32_t packed = static_cast<uint32_t>(
          qweight[(inner / kInt4PerWord) * n + column]);
      const int quantized_weight =
          static_cast<int>((packed >> ((inner % kInt4PerWord) * 4)) & 0xF) - 8;
      float activation = __bfloat162float(x[row * k + inner]);
      if constexpr (QuantizeActivation) {
        const int quantized_activation = __float2int_rn(activation / activation_scale);
        activation = static_cast<float>(max(-127, min(127, quantized_activation)))
            * activation_scale;
      }
      accumulator += activation * static_cast<float>(quantized_weight) * weight_scale;
    }
  }
  output[row * n + column] = __float2bfloat16(accumulator);
}

void validate_inputs(
    const torch::Tensor& x,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    int64_t group_size) {
  TORCH_CHECK(x.is_cuda() && qweight.is_cuda() && scales.is_cuda(),
              "native W4 tensors must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16,
              "native W4 activation must be BF16");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32,
              "native W4 qweight must be INT32");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16 ||
                  scales.scalar_type() == torch::kBFloat16,
              "native W4 scales must be FP16 or BF16");
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2 && scales.dim() == 2,
              "native W4 tensors must be rank-2");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous() && scales.is_contiguous(),
              "native W4 tensors must be contiguous");
  const int64_t k = qweight.size(0) * kInt4PerWord;
  TORCH_CHECK(x.size(1) == k, "native W4 activation K mismatch");
  TORCH_CHECK(qweight.size(1) == scales.size(1), "native W4 N mismatch");
  TORCH_CHECK(group_size == 128 && k % group_size == 0,
              "native W4 v1 requires group_size=128");
  TORCH_CHECK(scales.size(0) == k / group_size, "native W4 group mismatch");
  TORCH_CHECK(x.get_device() == qweight.get_device() &&
                  x.get_device() == scales.get_device(),
              "native W4 tensors must share a device");
}

template <bool QuantizeActivation>
torch::Tensor launch(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    int64_t group_size,
    dim3 threads) {
  validate_inputs(x, qweight, scales, group_size);
  c10::cuda::CUDAGuard guard(x.device());
  const int m = static_cast<int>(x.size(0));
  const int n = static_cast<int>(qweight.size(1));
  const int k = static_cast<int>(x.size(1));
  auto output = torch::empty({m, n}, x.options());
  const dim3 blocks((n + threads.x - 1) / threads.x,
                    (m + threads.y - 1) / threads.y);
  AT_DISPATCH_SWITCH(
      scales.scalar_type(), "packed_w4_gemm", AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
        packed_w4_gemm_kernel<scalar_t, QuantizeActivation><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            qweight.data_ptr<int32_t>(),
            scales.data_ptr<scalar_t>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            m, n, k, static_cast<int>(group_size));
      }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
        packed_w4_gemm_kernel<scalar_t, QuantizeActivation><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            qweight.data_ptr<int32_t>(),
            scales.data_ptr<scalar_t>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            m, n, k, static_cast<int>(group_size));
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

torch::Tensor w4a16_small_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    int64_t group_size) {
  return launch<false>(x, qweight, scales, group_size, dim3(32, 4));
}

torch::Tensor w4a16_large_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    int64_t group_size) {
  return launch<false>(x, qweight, scales, group_size, dim3(16, 16));
}

torch::Tensor w4a8_large_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    int64_t group_size) {
  return launch<true>(x, qweight, scales, group_size, dim3(16, 16));
}
