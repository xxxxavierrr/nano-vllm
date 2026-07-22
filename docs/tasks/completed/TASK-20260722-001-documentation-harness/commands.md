# Command log

Run from the repository root on 2026-07-22.

```powershell
rg --files -g "*.md" -g ".workflow/**" -g "AGENTS.md"
Get-ChildItem -Recurse -File docs | Select-Object -ExpandProperty FullName
git diff --check
```

These inventory required artifacts and validate patch formatting. Final
outcomes are recorded in `tests.md`; secrets and routine inspections are not
logged.

