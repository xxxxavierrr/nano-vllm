import argparse
import os
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--quantization", choices=["fp8"])
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(0)
    started = perf_counter()
    llm = LLM(
        os.path.expanduser(args.model),
        quantization=args.quantization,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    init_seconds = perf_counter() - started

    model = llm.model_runner.model
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    config = llm.model_runner.config

    torch.manual_seed(0)
    started = perf_counter()
    prompt = llm.tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Explain why low-precision inference can be faster in three sentences.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.6, max_tokens=args.max_tokens),
        use_tqdm=False,
    )
    generation_seconds = perf_counter() - started

    print(f"quantization={args.quantization or 'none'}")
    print(f"init_seconds={init_seconds:.3f}")
    print(f"model_storage_mib={(parameter_bytes + buffer_bytes) / 2**20:.2f}")
    print(f"kv_cache_blocks={config.num_kvcache_blocks}")
    print(f"kv_cache_token_capacity={config.num_kvcache_blocks * config.kvcache_block_size}")
    print(f"generation_seconds={generation_seconds:.3f}")
    print(f"output_tokens={len(outputs[0]['token_ids'])}")
    print(f"output={outputs[0]['text']!r}")


if __name__ == "__main__":
    main()
