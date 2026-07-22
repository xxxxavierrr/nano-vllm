from docs.hooks.check_structure import FunctionSize, evaluate, inspect_source


def test_inspect_source_tracks_methods_and_async_functions():
    source = """
class Runner:
    def run(self):
        return 1

async def serve():
    return 2
"""
    functions = inspect_source("sample.py", source)
    assert [(item.name, item.line) for item in functions] == [
        ("Runner.run", 3),
        ("serve", 6),
    ]


def test_unknown_oversized_function_is_rejected():
    function = FunctionSize("sample.py", "orchestrate", 1, 81)
    [finding] = evaluate([function])
    assert finding.severity == "error"
    assert finding.limit == 80


def test_named_required_boundary_uses_stricter_limit():
    function = FunctionSize("nanovllm/engine/model_runner.py", "ModelRunner.run", 1, 61)
    [finding] = evaluate([function])
    assert finding.severity == "error"
    assert finding.limit == 60


def test_numerical_kernel_exception_is_explicit():
    function = FunctionSize(
        "nanovllm/layers/gptq_kernel.py",
        "_gptq_w4a16_kernel",
        1,
        200,
    )
    [finding] = evaluate([function])
    assert finding.severity == "exempt"
    assert finding.reason


def test_legacy_debt_fails_if_it_grows():
    at_ceiling = FunctionSize(
        "nanovllm/engine/scheduler.py", "Scheduler.schedule", 1, 85
    )
    over_ceiling = FunctionSize(
        "nanovllm/engine/scheduler.py", "Scheduler.schedule", 1, 86
    )
    assert evaluate([at_ceiling])[0].severity == "debt"
    assert evaluate([over_ceiling])[0].severity == "error"
