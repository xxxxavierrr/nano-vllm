# Design

Introduce typed attention, GDN, sampling, and execution metadata inside one
prepared batch. Retain a compatibility-aware forward-context access point for
custom ops, but populate it with a scoped context manager. Move GDN partition
construction to runner-side helpers and make warmup/tests use the same contract.

Numerical kernel signatures and scheduler behavior remain unchanged in this
stage.
