import argparse
import copy
import math

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer as HFQwen3_5DecoderLayer,
    Qwen3_5RMSNorm as HFQwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

from nanovllm.engine.batch import (
    AttentionMetadata,
    ExecutionSignature,
    PreparedBatch,
    SamplingMetadata,
)
from nanovllm.models.qwen3_5_mtp import Qwen3_5MTP
from nanovllm.utils.context import forward_context
from nanovllm.utils.mtp_loader import load_mtp_model


class TransformersMTPReference(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = copy.deepcopy(config)
        config.layer_types[0] = "full_attention"
        config._attn_implementation = "flash_attention_2"
        self.config = config
        self.fc = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        self.layers = nn.ModuleList([HFQwen3_5DecoderLayer(config, 0)])
        self.norm = HFQwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = HFQwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = HFQwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)

    def forward(self, positions, target_hidden_states, token_embeddings):
        hidden_states = torch.cat(
            (
                self.pre_fc_norm_embedding(token_embeddings),
                self.pre_fc_norm_hidden(target_hidden_states),
            ),
            dim=-1,
        )
        hidden_states = self.fc(hidden_states)
        position_ids = positions.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        attention_mask = None
        hidden_states = self.layers[0](
            hidden_states.unsqueeze(0),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )[0]
        return self.norm(hidden_states)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare nano-vLLM Qwen3.6 MTP with Transformers."
    )
    parser.add_argument(
        "--target-model",
        default="/root/autodl-tmp/huggingface/Qwen3.6-27b-gptq-int4",
    )
    parser.add_argument(
        "--mtp-model",
        default="/root/autodl-tmp/huggingface/Qwen3.6-27B-mtp",
    )
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=2441)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    if args.tokens < 1:
        raise ValueError("--tokens must be positive")
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{args.master_port}",
        rank=0,
        world_size=1,
    )
    old_dtype = torch.get_default_dtype()
    old_device = torch.get_default_device()
    try:
        config = AutoConfig.from_pretrained(args.target_model).text_config
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda")
        nano_model = Qwen3_5MTP(config)
        load_mtp_model(nano_model, args.mtp_model, config)

        reference = TransformersMTPReference(config)
        raw = load_file(
            f"{args.mtp_model}/mtp.safetensors", device="cpu"
        )
        state_dict = {
            name.removeprefix("mtp."): tensor
            for name, tensor in raw.items()
        }
        reference.load_state_dict(state_dict, strict=True)
        del raw, state_dict

        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        target_hidden = torch.randn(
            args.tokens,
            config.hidden_size,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        token_embeddings = torch.randn(
            args.tokens,
            config.hidden_size,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        positions = torch.arange(args.tokens, device="cuda", dtype=torch.long)
        cu_seqlens = torch.tensor(
            [0, args.tokens], device="cuda", dtype=torch.int32
        )
        prepared = PreparedBatch(
            input_ids=torch.zeros(
                args.tokens, device="cuda", dtype=torch.long
            ),
            positions=positions,
            signature=ExecutionSignature(
                args.tokens, 1, args.tokens, None
            ),
            attention=AttentionMetadata(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=args.tokens,
                max_seqlen_k=args.tokens,
                use_kv_cache=False,
            ),
            sampling=SamplingMetadata(),
        )
        with forward_context(prepared):
            actual = nano_model(positions, target_hidden, token_embeddings)
        expected = reference(positions, target_hidden, token_embeddings)

        actual_float = actual.float()
        expected_float = expected.float()
        delta = actual_float - expected_float
        cosine = torch.nn.functional.cosine_similarity(
            actual_float.flatten(), expected_float.flatten(), dim=0
        )
        relative_l2 = delta.norm() / expected_float.norm().clamp_min(1e-12)
        print(f"tokens={args.tokens}")
        print(f"max_abs={delta.abs().max().item():.8f}")
        print(f"mean_abs={delta.abs().mean().item():.8f}")
        print(f"relative_l2={relative_l2.item():.8f}")
        print(f"cosine_similarity={cosine.item():.8f}")
        if relative_l2.item() > 1e-2 or cosine.item() < 0.9999:
            raise AssertionError(
                "nano-vLLM MTP exceeds the BF16 reference tolerance"
            )
    finally:
        torch.set_default_device(old_device)
        torch.set_default_dtype(old_dtype)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
