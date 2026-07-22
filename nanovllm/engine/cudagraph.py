from dataclasses import dataclass
from enum import Enum

from nanovllm.engine.batch import ExecutionSignature


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
    uniform_query_len: int | None
    execution_mode: ExecutionMode

    def __post_init__(self) -> None:
        ExecutionSignature(
            num_tokens=self.num_tokens,
            num_requests=self.num_seqs,
            num_padded_tokens=self.num_padded_tokens,
            uniform_query_len=self.uniform_query_len,
        )

    @property
    def signature(self) -> ExecutionSignature:
        return ExecutionSignature(
            num_tokens=self.num_tokens,
            num_requests=self.num_seqs,
            num_padded_tokens=self.num_padded_tokens,
            uniform_query_len=self.uniform_query_len,
        )

    @property
    def padded_requests(self) -> int:
        if self.uniform_query_len is None:
            raise ValueError("non-uniform execution has no request padding key")
        quotient, remainder = divmod(
            self.num_padded_tokens,
            self.uniform_query_len,
        )
        if remainder:
            raise ValueError("padded tokens are not divisible by query length")
        return quotient

    @property
    def full_key(self) -> tuple[int, int]:
        if self.execution_mode is not ExecutionMode.FULL:
            raise ValueError("only Full execution descriptors have a Full key")
        return self.uniform_query_len, self.padded_requests


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


def infer_piecewise_capture_limit(
    *,
    requested_max_tokens: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    speculative_tokens: int,
) -> int:
    if min(
        requested_max_tokens,
        max_num_batched_tokens,
        max_num_seqs,
    ) <= 0:
        raise ValueError("CUDA Graph capture limits must be positive")
    if speculative_tokens < 0:
        raise ValueError("speculative_tokens cannot be negative")
    return min(requested_max_tokens, max_num_batched_tokens, 512)


class CUDAGraphDispatcher:
    def __init__(
        self,
        mode: CUDAGraphMode | str,
        full_capture_sizes: list[int],
        piecewise_capture_sizes: list[int],
        full_query_lengths: list[int] | None = None,
    ):
        self.mode = CUDAGraphMode.parse(mode)
        self.full_capture_sizes = sorted(set(full_capture_sizes))
        self.piecewise_capture_sizes = sorted(set(piecewise_capture_sizes))
        self.full_query_lengths = sorted(set(full_query_lengths or [1]))
        if not self.full_query_lengths or self.full_query_lengths[0] <= 0:
            raise ValueError("Full Graph query lengths must be positive")

    @staticmethod
    def _find_bucket(num_tokens: int, sizes: list[int]) -> int | None:
        return next((size for size in sizes if size >= num_tokens), None)

    def dispatch(
        self,
        num_tokens: int,
        num_seqs: int,
        uniform_query_len: int | None,
    ) -> BatchDescriptor:
        if num_tokens <= 0 or num_seqs <= 0:
            raise ValueError("a CUDA Graph batch must contain tokens and sequences")

        if (
            uniform_query_len in self.full_query_lengths
            and self.mode.uses_full
        ):
            if num_tokens != num_seqs * uniform_query_len:
                raise ValueError(
                    "uniform query length does not match tokens and requests"
                )
            bucket = self._find_bucket(num_seqs, self.full_capture_sizes)
            if bucket is not None:
                return BatchDescriptor(
                    num_tokens,
                    bucket * uniform_query_len,
                    num_seqs,
                    uniform_query_len,
                    ExecutionMode.FULL,
                )

        if self.mode.uses_piecewise:
            bucket = self._find_bucket(num_tokens, self.piecewise_capture_sizes)
            if bucket is not None:
                return BatchDescriptor(
                    num_tokens,
                    bucket,
                    num_seqs,
                    uniform_query_len,
                    ExecutionMode.PIECEWISE,
                )

        return BatchDescriptor(
            num_tokens,
            num_tokens,
            num_seqs,
            uniform_query_len,
            ExecutionMode.EAGER,
        )
