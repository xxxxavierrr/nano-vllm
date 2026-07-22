import os

from setuptools import setup


ext_modules = []
cmdclass = {}
if os.getenv("NANOVLLM_BUILD_CUDA_EXT") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext_modules.append(
        CUDAExtension(
            "nanovllm._C",
            sources=["csrc/bindings.cpp", "csrc/w4_gemm.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_89,code=sm_89",
                ],
            },
        )
    )
    cmdclass["build_ext"] = BuildExtension


setup(ext_modules=ext_modules, cmdclass=cmdclass)
