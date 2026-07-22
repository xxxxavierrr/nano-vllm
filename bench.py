import argparse
import math
from pathlib import Path

from benchmarks.inprocess import run_inprocess
from nanovllm.engine.cudagraph import CUDAGraphMode


DEFAULT_MODEL = "/root/autodl-tmp/huggingface/Qwen3-0.6B"


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--label", default="nano-vllm")
    parser.add_argument("--quantization", choices=["fp8", "gptq"])
    parser.add_argument(
        "--gptq-kernel-backend", choices=["auto", "triton", "marlin"],
        default="auto",
    )
    parser.add_argument("--kv-cache-dtype", choices=["auto", "fp8_e4m3"], default="auto")
    parser.add_argument("--delta-state-dtype", choices=["auto", "fp8_e4m3"], default="auto")
    parser.add_argument("--speculative-method", choices=["none", "mtp"], default="none")
    parser.add_argument("--num-speculative-tokens", type=int, default=2)
    parser.add_argument("--mtp-model")


def _add_workload_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--range-ratio", type=float, default=0.0)
    parser.add_argument("--shared-prefix-len", type=int, default=0)
    parser.add_argument("--request-rate", type=float, default=math.inf)
    parser.add_argument("--max-concurrency", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-output-len", type=int, default=8)
    parser.add_argument("--warmup-num-requests", type=int, default=1)


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=2333)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--cudagraph-mode", choices=[mode.value for mode in CUDAGraphMode],
        default=CUDAGraphMode.FULL_AND_PIECEWISE.value,
    )
    parser.add_argument("--piecewise-max-tokens", type=int, default=512)
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--e2e-slo-ms", type=float)
    parser.add_argument("--output-json")
    parser.add_argument("--request-details", action="store_true")


def _validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.num_requests <= 0 or args.input_len <= 0 or args.output_len <= 0:
        parser.error("request count and token lengths must be positive")
    if not 0 <= args.range_ratio < 1:
        parser.error("--range-ratio must be in [0, 1)")
    min_input = max(1, int(args.input_len * (1 - args.range_ratio)))
    if not 0 <= args.shared_prefix_len <= min_input:
        parser.error("--shared-prefix-len cannot exceed the minimum sampled input length")
    if args.input_len + args.output_len > args.max_model_len:
        parser.error("input_len + output_len cannot exceed --max-model-len")
    if args.request_rate <= 0 or args.max_concurrency < 0:
        parser.error("request rate must be positive and concurrency non-negative")
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be between 1 and 65535")
    if args.piecewise_max_tokens <= 0 or args.warmup_num_requests <= 0:
        parser.error("piecewise and warmup sizes must be positive")
    if args.speculative_method == "mtp":
        if args.num_speculative_tokens not in (1, 2, 3):
            parser.error("--num-speculative-tokens must be 1, 2, or 3")
        if args.temperature != 0:
            parser.error("current MTP milestone requires --temperature 0")
        if args.mtp_model is not None and not Path(args.mtp_model).is_dir():
            parser.error(f"MTP model directory does not exist: {args.mtp_model}")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Comprehensive in-process nano-vLLM benchmark"
    )
    _add_model_options(parser)
    _add_workload_options(parser)
    _add_runtime_options(parser)
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def main(argv: list[str] | None = None):
    return run_inprocess(parse_args(argv))


if __name__ == "__main__":
    main()
