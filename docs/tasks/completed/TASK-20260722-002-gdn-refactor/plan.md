# Plan and process

- [x] Research vLLM's GDN forward and compiler boundary.
- [x] Inventory APIs and kernel responsibilities.
- [x] Collapse production, tests, and benchmark onto one packed API.
- [x] Add one stateful GDN custom op.
- [x] Fuse projections and update loader mapping.
- [x] Update tests and benchmark call sites.
- [x] Apply source-study alignment: remove layer metadata orchestration, remove
  the production Python request loop, and add explicit pre-cache chunk warmup.
- [x] Validate CUDA numerics, state, and loader integration on the server.
- [!] Transfer dedicated Graph replay, kernel crossover, and online goodput to
  TASK-20260722-012.
- [x] Fill the result, promote durable findings, and archive as superseded.
