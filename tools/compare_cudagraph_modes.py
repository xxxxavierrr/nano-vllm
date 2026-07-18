import argparse
import gc
import json

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.cudagraph import CUDAGraphMode


def run_mode(args, mode: CUDAGraphMode, master_port: int):
    torch.manual_seed(args.seed)
    llm = LLM(
        args.model,
        cudagraph_mode=mode.value,
        piecewise_max_tokens=args.piecewise_max_tokens,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=len(args.prompt_lengths),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        master_port=master_port,
    )
    try:
        prompts = [
            [((index + offset) % (llm.model_runner.config.hf_config.vocab_size - 1)) + 1
             for index in range(length)]
            for offset, length in enumerate(args.prompt_lengths)
        ]
        torch.manual_seed(args.seed)
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=args.temperature,
                max_tokens=args.output_len,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
        return [output["token_ids"] for output in outputs]
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare eager and FULL_AND_PIECEWISE token outputs on a GPU",
    )
    parser.add_argument("model")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[20, 20])
    parser.add_argument("--output-len", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=2360)
    parser.add_argument("--piecewise-max-tokens", type=int, default=24)
    parser.add_argument("--max-num-batched-tokens", type=int, default=24)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    if any(length <= 0 for length in args.prompt_lengths):
        parser.error("prompt lengths must be positive")
    if sum(args.prompt_lengths) <= args.max_num_batched_tokens:
        parser.error(
            "sum(prompt lengths) must exceed max_num_batched_tokens to exercise a mixed step"
        )
    return args


def main():
    args = parse_args()
    eager = run_mode(args, CUDAGraphMode.NONE, args.master_port)
    full_and_piecewise = run_mode(
        args,
        CUDAGraphMode.FULL_AND_PIECEWISE,
        args.master_port + 1,
    )
    result = {
        "eager": eager,
        "full_and_piecewise": full_and_piecewise,
        "match": eager == full_and_piecewise,
    }
    print(json.dumps(result, indent=2))
    if not result["match"]:
        raise SystemExit("eager and FULL_AND_PIECEWISE generated different token IDs")


if __name__ == "__main__":
    main()
