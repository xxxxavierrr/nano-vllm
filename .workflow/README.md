# nano-vLLM engineering workflow

This directory is the repository-level operating contract for humans and
coding agents. Read this file before acting on a request.

## Mandatory lifecycle

Every engineering request, investigation, benchmark, bug fix, refactor, or
optimization must follow this lifecycle:

1. **Create or revise the owning spec before implementation**
   - Specs are durable capability/module contracts, not task records. Name
     them by subject, for example `docs/specs/gated-deltanet.md`.
   - Every request must add its requirement to an existing owning spec or
     create a new subject spec when no suitable one exists.
   - Record motivation, scope, non-goals, requirements, acceptance criteria,
     constraints, unresolved questions, and a dated change-log entry.
   - Never split a spec merely because execution is split into multiple tasks.
   - Read-only discovery needed to write an accurate spec is allowed first.
2. **Create an active task**
   - Allocate `TASK-YYYYMMDD-NNN-short-slug` for the execution unit.
   - Create `docs/tasks/active/<TASK-ID>/` with all required task files:
     `README.md`, `research.md`, `design.md`, `plan.md`, `commands.md`,
     `tests.md`, `decisions.md`, and `result.md`.
   - Link it to the spec and add it to `docs/TASKS.md`.
3. **Research and design**
   - Put references and repository findings in `research.md`.
   - Put architecture, interfaces, state ownership, alternatives, risks, and
     compatibility choices in `design.md`.
4. **Plan**
   - Put verifiable steps and the test matrix in `plan.md`.
   - Mark exactly one step `[>]` while work is in progress.
   - For implementation work, list the affected ownership boundaries and the
     methods/modules that must become smaller or must not grow.
5. **Implement within the spec**
   - Amend the spec/design before materially changing scope or behavior.
   - Preserve unrelated user changes and secrets.
   - Orchestration code delegates policy, state mutation, numerical work, and
     reporting to their declared owners. Passing tests do not justify adding a
     new responsibility to an already broad method.
6. **Record execution**
   - Log meaningful build, test, benchmark, migration, server, and Git commands
     in `commands.md`, with environment, outcome, and artifact paths.
   - Never log tokens, credentials, private URLs, or raw secrets.
7. **Validate**
   - Record planned and actual evidence in `tests.md`.
   - Never call skipped or unavailable tests passed.
   - Separate static/CPU, GPU correctness, graph, online, accuracy, benchmark,
     and regression evidence.
   - Run the repository structure check for touched Python orchestration code.
     New or expanded review-trigger functions require decomposition or a
     task-local exception with a concrete reason and owner.
8. **Record decisions and knowledge**
   - Put task tradeoffs in `decisions.md`.
   - Promote only stable cross-task facts to `docs/knowledge/`.
9. **Complete and archive**
   - Fill `result.md` with delivery, evidence, limitations, and follow-ups.
   - Set the spec to `completed`, `cancelled`, or `superseded`.
   - Move the task from `docs/tasks/active/` to `docs/tasks/completed/`.
   - Update `docs/TASKS.md`, `docs/README.md`, and affected architecture,
     knowledge, benchmark, and public-interface docs.
   - If required validation is missing, keep the task active and state why.

## Status and plan markers

Statuses are `draft`, `active`, `blocked`, `completed`, `cancelled`, and
`superseded`. Plan markers are `[ ]` pending, `[>]` in progress, `[x]`
completed, and `[!]` blocked/failed with a reason.

## Source of truth

- The spec owns **what and why**.
- `design.md` owns **how and why this design**.
- `plan.md` owns **execution order and current progress**.
- `commands.md` and `tests.md` own **evidence**.
- `result.md` owns **delivered behavior and remaining gaps**.
- `docs/knowledge/` owns durable cross-task knowledge.

## Required content

A spec records its subject, status, dates, owner, motivation, requirements,
scope, non-goals, acceptance criteria, constraints, open questions, and change
log. It must not contain per-task progress, command logs, or test transcripts.
A task README links the owning spec(s) and summarizes its goal, status, current
gate, and record files. Supporting task files must contain real task data; do
not create placeholder-only records.

## Architecture and maintainability gate

Before implementation, `design.md` must name for every affected component:

- what it owns;
- what it delegates;
- mutable state and lifecycle it controls;
- responsibilities explicitly forbidden from entering it.

The checked-in hooks are mandatory:

1. Before implementation, run
   `python docs/hooks/check_structure.py --report-only` and record the baseline.
2. After implementation and before every commit, run
   `python docs/hooks/run_required_checks.py` and record the result.

The structure checker is a review trigger, not a substitute for design review:

- Python orchestration functions over 80 lines are rejected by default;
- token-hot-path orchestration functions listed by the architecture spec use a
  stricter 60-line budget;
- numerical Triton/CUDA kernels, generated code, and compatibility shims may
  be exempt only in the checked-in policy, with a reason;
- an existing over-budget function is technical debt, not permission to grow;
  a task touching its behavior must reduce it or record a blocking dependency.

Function length is only the mechanical floor. A shorter function still fails
review when it owns unrelated policy, state, numerical, serialization, and
reporting responsibilities. Structural acceptance evidence belongs in
`tests.md`; exceptions and their removal condition belong in `decisions.md`.

## Resume protocol

Read `.workflow/README.md`, the documentation indexes, the active task and its
spec/evidence, then relevant knowledge. Resume from the single `[>]` item; do
not repeat completed work.
