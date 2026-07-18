from dataclasses import dataclass
from enum import Enum


class CUDAGraphMode(str, Enum):
    FULL_AND_PIECEWISE = "FULL_AND_PIECEWISE"
    FULL_DECODE_ONLY = "FULL_DECODE_ONLY"
    PIECEWISE = "PIECEWISE"
    NONE = "NONE"

    @classmethod
    def parse(cls, value: "CUDAGraphMode | str") -> "CUDAGraphMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(value.upper())
        except (AttributeError, ValueError) as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"cudagraph_mode must be one of: {choices}") from exc

    @property
    def uses_full(self) -> bool:
        return self in (self.FULL_AND_PIECEWISE, self.FULL_DECODE_ONLY)

    @property
    def uses_piecewise(self) -> bool:
        return self in (self.FULL_AND_PIECEWISE, self.PIECEWISE)


class ExecutionMode(str, Enum):
    FULL = "FULL"
    PIECEWISE = "PIECEWISE"
    EAGER = "EAGER"


@dataclass(frozen=True, slots=True)
class BatchDescriptor:
    num_tokens: int
    num_padded_tokens: int
    num_seqs: int
    uniform_decode: bool
    execution_mode: ExecutionMode


def make_piecewise_capture_sizes(max_tokens: int) -> list[int]:
    if max_tokens <= 0:
        return []
    sizes = [size for size in (1, 2, 4) if size <= max_tokens]
    sizes.extend(range(8, max_tokens + 1, 8))
    if not sizes or sizes[-1] != max_tokens:
        sizes.append(max_tokens)
    return sorted(set(sizes))


def make_full_capture_sizes(max_batch_size: int) -> list[int]:
    if max_batch_size <= 0:
        return []
    sizes = [size for size in (1, 2, 4, 8) if size <= max_batch_size]
    sizes.extend(range(16, max_batch_size + 1, 16))
    if not sizes or sizes[-1] != max_batch_size:
        sizes.append(max_batch_size)
    return sorted(set(sizes))


class CUDAGraphDispatcher:
    def __init__(
        self,
        mode: CUDAGraphMode | str,
        full_capture_sizes: list[int],
        piecewise_capture_sizes: list[int],
    ):
        self.mode = CUDAGraphMode.parse(mode)
        self.full_capture_sizes = sorted(set(full_capture_sizes))
        self.piecewise_capture_sizes = sorted(set(piecewise_capture_sizes))

    @staticmethod
    def _find_bucket(num_tokens: int, sizes: list[int]) -> int | None:
        return next((size for size in sizes if size >= num_tokens), None)

    def dispatch(
        self,
        num_tokens: int,
        num_seqs: int,
        uniform_decode: bool,
    ) -> BatchDescriptor:
        if num_tokens <= 0 or num_seqs <= 0:
            raise ValueError("a CUDA Graph batch must contain tokens and sequences")

        if uniform_decode and self.mode.uses_full:
            bucket = self._find_bucket(num_tokens, self.full_capture_sizes)
            if bucket is not None:
                return BatchDescriptor(
                    num_tokens,
                    bucket,
                    num_seqs,
                    uniform_decode,
                    ExecutionMode.FULL,
                )

        if self.mode.uses_piecewise:
            bucket = self._find_bucket(num_tokens, self.piecewise_capture_sizes)
            if bucket is not None:
                return BatchDescriptor(
                    num_tokens,
                    bucket,
                    num_seqs,
                    uniform_decode,
                    ExecutionMode.PIECEWISE,
                )

        return BatchDescriptor(
            num_tokens,
            num_tokens,
            num_seqs,
            uniform_decode,
            ExecutionMode.EAGER,
        )
