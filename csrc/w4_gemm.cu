#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <mma.h>
#include <cuda_runtime.h>

namespace {

constexpr int kInt4PerWord = 8;

template <typename scale_t, int TileM, int TileN, int TileK, int FragmentsPerWarp>
__global__ void packed_w4a16_wmma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ qweight,
    const scale_t* __restrict__ scales,
    const int32_t* __restrict__ input_perm,
    __nv_bfloat16* __restrict__ output,
    int m,
    int n,
    int k,
    int group_size) {
  constexpr int kNTiles = TileN / 16;
  __shared__ __nv_bfloat16 activation_tile[TileM * TileK];
  __shared__ __nv_bfloat16 weight_tile[TileK * TileN];
  __shared__ float output_tile[TileM * TileN];

  const int thread = threadIdx.x;
  const int warp = thread / 32;
  const int warp_m = warp / (kNTiles / FragmentsPerWarp);
  const int warp_n = warp % (kNTiles / FragmentsPerWarp);
  const int row_start = blockIdx.y * TileM;
  const int column_start = blockIdx.x * TileN;

  using namespace nvcuda;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>
      accumulators[FragmentsPerWarp];
#pragma unroll
  for (int fragment = 0; fragment < FragmentsPerWarp; ++fragment) {
    wmma::fill_fragment(accumulators[fragment], 0.0f);
  }

  for (int k_start = 0; k_start < k; k_start += TileK) {
    for (int index = thread; index < TileM * TileK; index += blockDim.x) {
      const int tile_row = index / TileK;
      const int tile_k = index % TileK;
      const int row = row_start + tile_row;
      const int inner = k_start + tile_k;
      const int activation_inner = input_perm[inner];
      activation_tile[index] =
          row < m ? x[row * k + activation_inner] : __float2bfloat16(0.0f);
    }
    for (int index = thread; index < (TileK / kInt4PerWord) * TileN;
         index += blockDim.x) {
      const int packed_k = index / TileN;
      const int tile_column = index % TileN;
      const int column = column_start + tile_column;
      uint32_t packed = 0;
      float scale = 0.0f;
      if (column < n) {
        packed = static_cast<uint32_t>(
            qweight[(k_start / kInt4PerWord + packed_k) * n + column]);
        scale = static_cast<float>(scales[(k_start / group_size) * n + column]);
      }
#pragma unroll
      for (int lane = 0; lane < kInt4PerWord; ++lane) {
        const int quantized = static_cast<int>((packed >> (lane * 4)) & 0xF) - 8;
        const int tile_k = packed_k * kInt4PerWord + lane;
        weight_tile[tile_k * TileN + tile_column] =
            __float2bfloat16(static_cast<float>(quantized) * scale);
      }
    }
    __syncthreads();

    for (int tile_k = 0; tile_k < TileK; tile_k += 16) {
      wmma::fragment<
          wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
          activation_fragment;
      wmma::load_matrix_sync(
          activation_fragment,
          activation_tile + warp_m * 16 * TileK + tile_k,
          TileK);
#pragma unroll
      for (int fragment = 0; fragment < FragmentsPerWarp; ++fragment) {
        wmma::fragment<
            wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major>
            weight_fragment;
        const int tile_n = warp_n * FragmentsPerWarp + fragment;
        wmma::load_matrix_sync(
            weight_fragment,
            weight_tile + tile_k * TileN + tile_n * 16,
            TileN);
        wmma::mma_sync(
            accumulators[fragment],
            activation_fragment,
            weight_fragment,
            accumulators[fragment]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int fragment = 0; fragment < FragmentsPerWarp; ++fragment) {
    const int tile_n = warp_n * FragmentsPerWarp + fragment;
    const int output_tile_id = warp_m * kNTiles + tile_n;
    wmma::store_matrix_sync(
        output_tile + output_tile_id * 16 * 16,
        accumulators[fragment],
        16,
        wmma::mem_row_major);
  }
  __syncthreads();
  for (int index = thread; index < TileM * TileN; index += blockDim.x) {
    const int tile_row = index / TileN;
    const int tile_column = index % TileN;
    const int row = row_start + tile_row;
    const int column = column_start + tile_column;
    if (row < m && column < n) {
      const int output_tile_id = (tile_row / 16) * kNTiles + tile_column / 16;
      const int warp_row = tile_row % 16;
      const int warp_column = tile_column % 16;
      output[row * n + column] = __float2bfloat16(
          output_tile[output_tile_id * 16 * 16 + warp_row * 16 + warp_column]);
    }
  }
}

template <typename scale_t, bool QuantizeActivation>
__global__ void packed_w4_gemm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ qweight,
    const scale_t* __restrict__ scales,
    const int32_t* __restrict__ input_perm,
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
        const int activation_inner = input_perm[group_start + offset];
        absmax = fmaxf(
            absmax,
            fabsf(__bfloat162float(x[row * k + activation_inner])));
      }
      activation_scale = fmaxf(absmax / 127.0f, 1.0e-12f);
    }
    for (int offset = 0; offset < group_size; ++offset) {
      const int inner = group_start + offset;
      const uint32_t packed = static_cast<uint32_t>(
          qweight[(inner / kInt4PerWord) * n + column]);
      const int quantized_weight =
          static_cast<int>((packed >> ((inner % kInt4PerWord) * 4)) & 0xF) - 8;
      float activation = __bfloat162float(x[row * k + input_perm[inner]]);
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
    const torch::Tensor& input_perm,
    int64_t group_size) {
  TORCH_CHECK(x.is_cuda() && qweight.is_cuda() && scales.is_cuda() && input_perm.is_cuda(),
              "native W4 tensors must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16,
              "native W4 activation must be BF16");
  TORCH_CHECK(qweight.scalar_type() == torch::kInt32,
              "native W4 qweight must be INT32");
  TORCH_CHECK(input_perm.scalar_type() == torch::kInt32,
              "native W4 input permutation must be INT32");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16 ||
                  scales.scalar_type() == torch::kBFloat16,
              "native W4 scales must be FP16 or BF16");
  TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2 && scales.dim() == 2 && input_perm.dim() == 1,
              "native W4 tensors must be rank-2");
  TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous() && scales.is_contiguous() && input_perm.is_contiguous(),
              "native W4 tensors must be contiguous");
  const int64_t k = qweight.size(0) * kInt4PerWord;
  TORCH_CHECK(x.size(1) == k, "native W4 activation K mismatch");
  TORCH_CHECK(qweight.size(1) == scales.size(1), "native W4 N mismatch");
  TORCH_CHECK(input_perm.numel() == k,
              "native W4 input permutation K mismatch");
  TORCH_CHECK(group_size == 128 && k % group_size == 0,
              "native W4 v1 requires group_size=128");
  TORCH_CHECK(scales.size(0) == k / group_size, "native W4 group mismatch");
  TORCH_CHECK(x.get_device() == qweight.get_device() &&
                  x.get_device() == scales.get_device() &&
                  x.get_device() == input_perm.get_device(),
              "native W4 tensors must share a device");
}

template <int TileM, int TileN, int TileK, int FragmentsPerWarp>
torch::Tensor launch_w4a16(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor input_perm,
    int64_t group_size) {
  validate_inputs(x, qweight, scales, input_perm, group_size);
  c10::cuda::CUDAGuard guard(x.device());
  const int m = static_cast<int>(x.size(0));
  const int n = static_cast<int>(qweight.size(1));
  const int k = static_cast<int>(x.size(1));
  auto output = torch::empty({m, n}, x.options());
  const dim3 threads(
      (TileM / 16) * (TileN / 16) / FragmentsPerWarp * 32);
  const dim3 blocks((n + TileN - 1) / TileN, (m + TileM - 1) / TileM);
  AT_DISPATCH_SWITCH(
      scales.scalar_type(), "packed_w4a16_wmma", AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
        packed_w4a16_wmma_kernel<
            scalar_t, TileM, TileN, TileK, FragmentsPerWarp><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            qweight.data_ptr<int32_t>(),
            scales.data_ptr<scalar_t>(),
            input_perm.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            m, n, k, static_cast<int>(group_size));
      }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
        packed_w4a16_wmma_kernel<
            scalar_t, TileM, TileN, TileK, FragmentsPerWarp><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            qweight.data_ptr<int32_t>(),
            scales.data_ptr<scalar_t>(),
            input_perm.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            m, n, k, static_cast<int>(group_size));
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <bool QuantizeActivation>
torch::Tensor launch(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor input_perm,
    int64_t group_size,
    dim3 threads) {
  validate_inputs(x, qweight, scales, input_perm, group_size);
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
            input_perm.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            m, n, k, static_cast<int>(group_size));
      }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
        packed_w4_gemm_kernel<scalar_t, QuantizeActivation><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            qweight.data_ptr<int32_t>(),
            scales.data_ptr<scalar_t>(),
            input_perm.data_ptr<int32_t>(),
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
    torch::Tensor input_perm,
    int64_t group_size) {
  return launch_w4a16<16, 64, 128, 1>(
      x, qweight, scales, input_perm, group_size);
}

torch::Tensor w4a16_large_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor input_perm,
    int64_t group_size) {
  return launch_w4a16<32, 128, 32, 2>(
      x, qweight, scales, input_perm, group_size);
}

torch::Tensor w4a8_large_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor input_perm,
    int64_t group_size) {
  return launch<true>(x, qweight, scales, input_perm, group_size, dim3(16, 16));
}
