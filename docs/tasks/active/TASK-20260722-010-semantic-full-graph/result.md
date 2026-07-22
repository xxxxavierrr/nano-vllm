# Result

Local implementation is complete. Full dispatch buckets requests, uses
`(uniform_query_len, padded_requests)` keys, and only selects finite query
lengths captured for the active configuration. Capture and replay now size
token-major and request-major buffers separately and include q=1+k for MTP.

The task remains active pending CUDA capture/replay, token/state parity, and
performance evidence on the GPU server.
