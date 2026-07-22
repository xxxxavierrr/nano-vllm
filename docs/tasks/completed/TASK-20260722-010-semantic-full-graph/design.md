# Design

`BatchDescriptor` continues to carry real/padded token counts and real request
count. Its Full key is `(uniform_query_len, padded_requests)`, where
`padded_requests = num_padded_tokens / uniform_query_len`.

The dispatcher buckets requests for Full and tokens for Piecewise. The runner
captures only the finite query lengths supported by the active configuration:
`1` and, when MTP is enabled, `1 + num_speculative_tokens`. Stable token-major
and request-major buffers are sized separately. GDN keeps its current declared
one-request Full capability until packed multi-request replay is proven.
