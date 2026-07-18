import pytest

from nanovllm.config import Config
from nanovllm.engine.cudagraph import (
    CUDAGraphDispatcher,
    CUDAGraphMode,
    ExecutionMode,
    make_full_capture_sizes,
    make_piecewise_capture_sizes,
)


def make_dispatcher(mode=CUDAGraphMode.FULL_AND_PIECEWISE):
    return CUDAGraphDispatcher(
        mode,
        full_capture_sizes=make_full_capture_sizes(512),
        piecewise_capture_sizes=make_piecewise_capture_sizes(512),
    )


def test_full_and_piecewise_dispatches_by_batch_shape():
    dispatcher = make_dispatcher()

    decode = dispatcher.dispatch(num_tokens=7, num_seqs=7, uniform_decode=True)
    assert decode.execution_mode is ExecutionMode.FULL
    assert decode.num_padded_tokens == 8

    mixed = dispatcher.dispatch(num_tokens=19, num_seqs=2, uniform_decode=False)
    assert mixed.execution_mode is ExecutionMode.PIECEWISE
    assert mixed.num_padded_tokens == 24

    oversized = dispatcher.dispatch(num_tokens=513, num_seqs=2, uniform_decode=False)
    assert oversized.execution_mode is ExecutionMode.EAGER
    assert oversized.num_padded_tokens == 513


@pytest.mark.parametrize(
    ("mode", "decode_mode", "mixed_mode"),
    [
        (CUDAGraphMode.FULL_DECODE_ONLY, ExecutionMode.FULL, ExecutionMode.EAGER),
        (CUDAGraphMode.PIECEWISE, ExecutionMode.PIECEWISE, ExecutionMode.PIECEWISE),
        (CUDAGraphMode.NONE, ExecutionMode.EAGER, ExecutionMode.EAGER),
    ],
)
def test_explicit_cudagraph_modes(mode, decode_mode, mixed_mode):
    dispatcher = make_dispatcher(mode)
    assert dispatcher.dispatch(4, 4, True).execution_mode is decode_mode
    assert dispatcher.dispatch(4, 2, False).execution_mode is mixed_mode


def test_full_falls_back_to_piecewise_when_decode_exceeds_full_capture_range():
    dispatcher = CUDAGraphDispatcher(
        CUDAGraphMode.FULL_AND_PIECEWISE,
        full_capture_sizes=make_full_capture_sizes(8),
        piecewise_capture_sizes=make_piecewise_capture_sizes(16),
    )
    descriptor = dispatcher.dispatch(12, 12, True)
    assert descriptor.execution_mode is ExecutionMode.PIECEWISE
    assert descriptor.num_padded_tokens == 16


def test_capture_size_generation_keeps_requested_limit():
    assert make_piecewise_capture_sizes(19) == [1, 2, 4, 8, 16, 19]
    assert make_full_capture_sizes(10) == [1, 2, 4, 8, 10]
    assert make_piecewise_capture_sizes(512) == [1, 2, 4, *range(8, 513, 8)]


def test_enforce_eager_overrides_explicit_graph_mode(tmp_path, monkeypatch):
    class FakeHFConfig:
        max_position_embeddings = 4096

    monkeypatch.setattr(
        "nanovllm.config.AutoConfig.from_pretrained",
        lambda _: FakeHFConfig(),
    )
    config = Config(
        str(tmp_path),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        enforce_eager=True,
    )
    assert config.cudagraph_mode is CUDAGraphMode.NONE
