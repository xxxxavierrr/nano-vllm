# Test evidence

Planned local evidence: descriptor/key validation, q=1 and q=1+k dispatch,
padding, unsupported-shape fallback, syntax, and CPU regressions.

Required server evidence: capture/replay logs, eager parity for q=1/2/3/4,
GDN working-state commit/replay, MTP token parity, and latency/memory metrics.

## Local result

- Static compilation: passed.
- Dispatcher/key, MTP, context, and state suite: 31 passed.
- Verified distinct `(query_len, padded_requests)` keys, request padding for
  q=1+k, and Piecewise fallback for uncaptured uniform query lengths.
- Attention/Piecewise collection is unavailable locally because `flash_attn`
  is not installed. No environment or dependency change was made.
