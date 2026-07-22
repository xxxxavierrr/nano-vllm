# Research

The implementation starts from existing uncommitted GDN/model/test changes.
Those changes are preserved and inspected before patching. The migration will
inventory every `set_context` call and every model-layer fallback dependency so
no warmup, MTP, replay, FP8 KV, or Graph path loses required metadata.
