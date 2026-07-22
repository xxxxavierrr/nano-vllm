import torch

import nanovllm.layers.attention as attention_module
from nanovllm.engine.batch import (
    AttentionMetadata,
    ExecutionSignature,
    PreparedBatch,
    SamplingMetadata,
)
from nanovllm.layers.attention import Attention
from nanovllm.utils.context import forward_context


def test_attention_only_consumes_real_tokens_and_restores_padding(monkeypatch):
    stored_tokens = []

    def fake_store(key, value, k_cache, v_cache, slot_mapping):
        stored_tokens.append((key.size(0), value.size(0), slot_mapping.numel()))

    def fake_flash(q, k, v, **kwargs):
        return q

    monkeypatch.setattr(attention_module, "store_kvcache", fake_store)
    monkeypatch.setattr(attention_module, "flash_attn_varlen_func", fake_flash)

    attention = Attention(num_heads=1, head_dim=2, scale=1.0, num_kv_heads=1)
    attention.k_cache.tensor = torch.ones(1)
    attention.v_cache.tensor = torch.ones(1)
    q = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    k = q + 10
    v = q + 20
    prepared = PreparedBatch(
        input_ids=torch.zeros(4, dtype=torch.long),
        positions=torch.zeros(4, dtype=torch.long),
        signature=ExecutionSignature(3, 1, 4, None),
        attention=AttentionMetadata(
            cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
            max_seqlen_q=3,
            max_seqlen_k=3,
        ),
        sampling=SamplingMetadata(),
    )
    with forward_context(prepared):
        output = attention(q, k, v)

    assert stored_tokens == [(3, 3, 3)]
    assert output.shape == q.shape
    torch.testing.assert_close(output[:3], q[:3])
    torch.testing.assert_close(output[3], torch.zeros_like(output[3]))
