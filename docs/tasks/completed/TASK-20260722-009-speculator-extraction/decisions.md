# Decisions

- The proposer does not own scheduler request state or GDN commit/rollback.
- Target embeddings/logit head may be shared by reference, but ownership stays
  explicit in the proposer constructor.
- Keep greedy-only behavior; probabilistic acceptance is future work.
