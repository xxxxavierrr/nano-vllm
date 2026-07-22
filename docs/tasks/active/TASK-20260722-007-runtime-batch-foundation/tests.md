# Test evidence

| Area | Planned validation | Status |
| --- | --- | --- |
| Static | compile all changed Python files | passed |
| Context | nested/scoped reset and typed metadata access | passed: 2 tests |
| Dispatcher | semantic uniform query lengths and existing modes | passed: 9 tests |
| GDN structure | no model-layer metadata builder | passed by source inspection; packed convolution loop remains active GDN work |
| CPU regression | context, dispatcher, speculative scheduler, abort, capacity, GPTQ reference/config | passed: 53, skipped: 1 |
| Local attention/model collection | tests requiring `flash_attn` | unavailable: local environment has no `flash_attn` |
| GPU numerics/state/Graph | server validation | unavailable locally |
