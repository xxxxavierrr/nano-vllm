# Research

`ModelRunner.run` currently mutates two fields. `LLMEngine.step` and the offline
benchmark read them later, which couples correctness to call ordering and makes
future concurrent or asynchronous execution unsafe. Rank-local calls already
return picklable Python objects, so metrics can travel with the batch result.
