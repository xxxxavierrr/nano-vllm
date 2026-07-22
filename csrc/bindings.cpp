#include <torch/extension.h>

torch::Tensor w4a16_small_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    int64_t group_size);
torch::Tensor w4a16_large_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    int64_t group_size);
torch::Tensor w4a8_large_cuda(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    int64_t group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("w4a16_small", &w4a16_small_cuda);
  module.def("w4a16_large", &w4a16_large_cuda);
  module.def("w4a8_large", &w4a8_large_cuda);
}
