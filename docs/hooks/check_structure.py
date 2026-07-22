from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_MAX_LINES = 80

# These are architectural acceptance boundaries, not a baseline allowlist.
REQUIRED_LIMITS = {
    ("bench.py", "main"): 80,
    ("benchmarks/metrics.py", "summarize"): 50,
    ("nanovllm/calibration/gptq_quantizer.py", "quantize_linear_gptq"): 40,
    ("nanovllm/engine/model_runner.py", "ModelRunner.prepare_inputs"): 60,
    ("nanovllm/engine/model_runner.py", "ModelRunner.run"): 60,
    ("nanovllm/engine/mtp_proposer.py", "MTPProposer.propose"): 60,
    ("nanovllm/layers/deltanet.py", "packed_causal_conv1d"): 60,
}

# Existing debt may not grow. Each entry has a removal ceiling and reason.
DEBT_CEILINGS = {
    ("benchmarks/backends/openai_chat.py", "OpenAIChatBackend.run"): (
        83,
        "legacy transport lifecycle; extract request/stream parsing",
    ),
    ("nanovllm/engine/model_runner.py", "ModelRunner.__init__"): (
        164,
        "legacy construction owner; remove through manager extraction",
    ),
    ("nanovllm/engine/model_runner.py", "ModelRunner.allocate_kv_cache"): (
        188,
        "legacy cache allocation; move to CacheManager",
    ),
    ("nanovllm/engine/model_runner.py", "ModelRunner.run_model"): (
        81,
        "legacy graph replay facade; move to GraphManager",
    ),
    ("nanovllm/engine/model_runner.py", "ModelRunner.capture_cudagraph"): (
        205,
        "legacy graph lifecycle; move to GraphManager",
    ),
    ("nanovllm/engine/scheduler.py", "Scheduler.schedule"): (
        85,
        "legacy scheduling pass; separate budget and batch construction",
    ),
    ("nanovllm/engine/scheduler.py", "Scheduler.postprocess"): (
        90,
        "legacy request transition pass; separate state transitions",
    ),
    ("nanovllm/layers/linear.py", "LinearBase.__init__"): (
        82,
        "legacy quantized parameter construction; move to quant methods",
    ),
    ("nanovllm/config.py", "Config.__post_init__"): (
        116,
        "legacy config resolution; split model, quantization, spec, and cache policies",
    ),
    ("nanovllm/serve/api_server.py", "create_app"): (
        116,
        "legacy FastAPI assembly; extract lifecycle and route builders",
    ),
    ("nanovllm/serve/api_server.py", "parse_args"): (
        86,
        "legacy declarative CLI; split server and engine option groups",
    ),
    ("nanovllm/serve/engine.py", "run_engine_proc"): (
        99,
        "legacy process lifecycle; extract startup, command loop, and shutdown",
    ),
}

# Numerical kernels are reviewed for one mathematical responsibility rather
# than Python orchestration line count. Wrappers are intentionally absent.
KERNEL_EXEMPTIONS = {
    ("nanovllm/layers/deltanet.py", "_gated_delta_recurrent_kernel"):
        "single recurrent-state scan kernel",
    ("nanovllm/layers/deltanet.py", "_packed_causal_conv_branch_state_kernel"):
        "single branch-state causal-convolution kernel",
    ("nanovllm/layers/deltanet.py", "_packed_causal_conv_state_kernel"):
        "single committed-state causal-convolution kernel",
    ("nanovllm/layers/deltanet_chunk.py", "_prepare_delta_chunk_kernel"):
        "single chunk preparation kernel",
    ("nanovllm/layers/deltanet_chunk.py", "_apply_delta_chunk_kernel"):
        "single chunk application kernel",
    ("nanovllm/layers/fp8_attention.py", "_fp8_paged_attention_kernel"):
        "single FP8 paged-attention kernel",
    ("nanovllm/layers/fp8_decode_attention.py", "_fp8_decode_split_kernel"):
        "single split-K FP8 decode kernel",
    ("nanovllm/layers/fp8_decode_attention.py", "_fp8_decode_reduce_kernel"):
        "single split-K reduction kernel",
    ("nanovllm/layers/gptq_kernel.py", "_gptq_w4a16_kernel"):
        "single packed W4A16 GEMM kernel",
}


@dataclass(frozen=True, slots=True)
class FunctionSize:
    path: str
    name: str
    line: int
    lines: int


@dataclass(frozen=True, slots=True)
class Finding:
    severity: str
    function: FunctionSize
    limit: int | None
    reason: str


class _FunctionVisitor(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path = path
        self.parents: list[str] = []
        self.functions: list[FunctionSize] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.parents.append(node.name)
        self.generic_visit(node)
        self.parents.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        name = ".".join((*self.parents, node.name))
        self.functions.append(
            FunctionSize(
                path=self.path,
                name=name,
                line=node.lineno,
                lines=node.end_lineno - node.lineno + 1,
            )
        )
        self.parents.append(node.name)
        self.generic_visit(node)
        self.parents.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function


def inspect_source(path: str, source: str) -> list[FunctionSize]:
    visitor = _FunctionVisitor(path)
    visitor.visit(ast.parse(source, filename=path))
    return visitor.functions


def evaluate(functions: Iterable[FunctionSize]) -> list[Finding]:
    findings: list[Finding] = []
    for function in functions:
        key = (function.path, function.name)
        if key in KERNEL_EXEMPTIONS:
            findings.append(
                Finding("exempt", function, None, KERNEL_EXEMPTIONS[key])
            )
            continue
        if key in REQUIRED_LIMITS:
            limit = REQUIRED_LIMITS[key]
            severity = "error" if function.lines > limit else "pass"
            findings.append(Finding(severity, function, limit, "required boundary"))
            continue
        if key in DEBT_CEILINGS:
            limit, reason = DEBT_CEILINGS[key]
            if function.lines > limit:
                severity = "error"
            else:
                severity = "pass" if function.lines <= DEFAULT_MAX_LINES else "debt"
            findings.append(Finding(severity, function, limit, reason))
            continue
        if function.lines > DEFAULT_MAX_LINES:
            findings.append(
                Finding(
                    "error",
                    function,
                    DEFAULT_MAX_LINES,
                    "unregistered oversized Python orchestration",
                )
            )
    return findings


def repository_functions(root: Path) -> list[FunctionSize]:
    paths = [root / "bench.py"]
    for directory in (root / "benchmarks", root / "nanovllm"):
        paths.extend(directory.rglob("*.py"))
    functions: list[FunctionSize] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        functions.extend(inspect_source(relative, path.read_text(encoding="utf-8")))
    return functions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Python ownership boundaries")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json", action="store_true", help="emit machine-readable findings")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="report violations without failing; intended for the pre-change baseline",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    findings = evaluate(repository_functions(args.root.resolve()))
    if args.json:
        print(json.dumps([asdict(finding) for finding in findings], indent=2))
    else:
        for finding in findings:
            function = finding.function
            limit = "exempt" if finding.limit is None else f"limit={finding.limit}"
            print(
                f"{finding.severity:6} {function.path}:{function.line} "
                f"{function.name} lines={function.lines} {limit} - {finding.reason}"
            )
    has_errors = any(finding.severity == "error" for finding in findings)
    return int(has_errors and not args.report_only)


if __name__ == "__main__":
    raise SystemExit(main())
