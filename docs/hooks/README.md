# Engineering hooks

These scripts are the executable part of [the repository workflow](../../.workflow/README.md).
They are intentionally stored with the engineering contract instead of hidden
in a developer-specific Git configuration.

Required lifecycle:

1. Before implementation, run `python docs/hooks/check_structure.py --report-only`
   and record the relevant baseline in the active task.
2. After implementation and before a commit, run
   `python docs/hooks/run_required_checks.py`.
3. Record both commands and outcomes in the task's `commands.md`/`tests.md`.

The scripts do not install a local `.git/hooks` entry. That would be easy to
bypass and would not run consistently for agents, Windows, and the GPU server.
The workflow makes the checked-in commands mandatory instead.
