# Test evidence

Planned: dataclass mapping, engine consumer tests, benchmark static checks, and
existing scheduler/speculation regression. GPU reporting remains required.

Local result: static compilation passed and 53 focused CPU tests passed. The
suite covers independent immutable metric defaults, result/metric association,
Graph dispatch, speculative acceptance/scheduling, state transactions, abort,
and capacity behavior.
