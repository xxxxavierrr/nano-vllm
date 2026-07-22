# Design

Add immutable `SpeculativeStepMetrics` and `RunnerStepMetrics`, plus a
`RunnerStepOutput` envelope carrying the existing token result. `LLMEngine`
unwraps once, scheduler postprocessing remains unchanged, and both engine step
reporting and offline benchmark consume the returned metrics.
