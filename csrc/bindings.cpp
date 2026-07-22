#include <torch/extension.h>

torch::Tensor w4a16_small_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor input_perm,
    int64_t group_size);
torch::Tensor w4a16_large_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor input_perm,
    int64_t group_size);
torch::Tensor w4a8_large_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor input_perm,
    int64_t group_size);

TORCH_LIBRARY(nanovllm_native, module) {
  module.def(
      "w4a16_small(Tensor x, Tensor qweight, Tensor scales, Tensor input_perm, int group_size) -> Tensor");
  module.def(
      "w4a16_large(Tensor x, Tensor qweight, Tensor scales, Tensor input_perm, int group_size) -> Tensor");
  module.def(
      "w4a8_large(Tensor x, Tensor qweight, Tensor scales, Tensor input_perm, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(nanovllm_native, CUDA, module) {
  module.impl("w4a16_small", &w4a16_small_cuda);
  module.impl("w4a16_large", &w4a16_large_cuda);
  module.impl("w4a8_large", &w4a8_large_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("w4a16_small", &w4a16_small_cuda);
  module.def("w4a16_large", &w4a16_large_cuda);
  module.def("w4a8_large", &w4a8_large_cuda);
}
