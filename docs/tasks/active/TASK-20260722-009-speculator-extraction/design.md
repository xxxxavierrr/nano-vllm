# Design

`MTPProposer` owns the draft model forward lifecycle and returns draft chains.
`GreedyAcceptance` owns exact prefix verification. Both consume the typed batch
and scoped context contracts. Scheduler state and GDN state transaction remain
outside the proposer.
