# Command log

Run from the repository root on 2026-07-22.

```powershell
Invoke-WebRequest -UseBasicParsing -Uri <pinned official raw GitHub source> -OutFile $env:TEMP\<source-file>
rg -n "forward_cuda|_forward_core|qwen_gdn_attention_core|get_forward_context|attn_metadata|state|chunk" $env:TEMP\vllm_*.py
rg -n "qwen_gdn_core|gated_delta_packed|state_slots|chunk|forward" nanovllm\layers nanovllm\models\qwen3_5.py nanovllm\engine\model_runner.py
```

The official files were downloaded only to the OS temporary directory for
line-accurate inspection. The upstream commit was resolved through the GitHub
commits API. No third-party code was copied into the repository.

