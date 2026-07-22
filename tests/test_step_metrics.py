from nanovllm.engine.metrics import (
    RunnerStepMetrics,
    RunnerStepOutput,
    SpeculativeStepMetrics,
)


def test_runner_step_output_keeps_result_and_metrics_together():
    speculative = SpeculativeStepMetrics(
        drafted=6,
        proposed=4,
        accepted=3,
        rejected=1,
        bonus=1,
        verification_rounds=2,
    )
    metrics = RunnerStepMetrics(
        execution_mode="FULL",
        real_tokens=8,
        padded_tokens=12,
        num_requests=2,
        speculative=speculative,
    )
    output = RunnerStepOutput(result=[[1, 2], [3]], metrics=metrics)

    assert output.result == [[1, 2], [3]]
    assert output.metrics.execution_mode == "FULL"
    assert output.metrics.speculative.accepted == 3


def test_runner_metrics_get_independent_zero_speculative_defaults():
    first = RunnerStepMetrics("EAGER", 1, 1, 1)
    second = RunnerStepMetrics("PIECEWISE", 7, 8, 2)

    assert first.speculative is not second.speculative
    assert first.speculative.drafted == 0
