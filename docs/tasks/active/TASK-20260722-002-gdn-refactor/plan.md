# Plan and process

- [x] Research vLLM's GDN forward and compiler boundary.
- [x] Inventory APIs and kernel responsibilities.
- [x] Collapse production, tests, and benchmark onto one packed API.
- [x] Add one stateful GDN custom op.
- [x] Fuse projections and update loader mapping.
- [x] Update tests and benchmark call sites.
- [x] Apply source-study alignment: remove layer metadata orchestration, remove
  the production Python request loop, and add explicit pre-cache chunk warmup.
- [>] Validate CUDA numerics, state, loader, and Graph behavior on the server.
- [ ] Measure kernel crossover and end-to-end online performance.
- [ ] Fill the result, promote durable findings, and archive.

Resume at the GPU validation item using the existing dirty refactor.
