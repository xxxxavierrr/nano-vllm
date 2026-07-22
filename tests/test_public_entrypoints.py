import importlib


def test_public_engine_and_benchmark_entrypoints_import():
    nanovllm = importlib.import_module("nanovllm")
    benchmark = importlib.import_module("benchmarks.inprocess")

    assert nanovllm.LLM.__name__ == "LLM"
    assert nanovllm.SamplingParams.__name__ == "SamplingParams"
    assert callable(benchmark.run_inprocess)
