from typing import TYPE_CHECKING

__all__ = ["LLM", "SamplingParams"]

if TYPE_CHECKING:
    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams


def __getattr__(name: str):
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    if name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams
        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
