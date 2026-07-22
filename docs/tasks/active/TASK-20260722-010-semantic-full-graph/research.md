# Research

The existing dispatcher carries `uniform_query_len`, but Full selection is
hard-coded to length one and graphs are keyed only by padded token count. The
runner also copies Full replay block tables and GDN slots using token count;
that happens to work only because decode currently has one token per request.

For a fixed MTP depth `k`, verification has uniform query length `1+k`.
Capture identity therefore needs both query length and padded request count.
Token count alone is ambiguous and cannot size request-major metadata.
