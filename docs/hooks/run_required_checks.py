from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path

from check_structure import main as check_structure


ROOT = Path(__file__).resolve().parents[2]


def _git_diff_check() -> int:
    completed = subprocess.run(
        ["git", "diff", "--check"],
        cwd=ROOT,
        check=False,
    )
    return completed.returncode


def _compile_python() -> bool:
    targets = (
        ROOT / "bench.py",
        ROOT / "benchmarks",
        ROOT / "nanovllm",
        ROOT / "docs" / "hooks",
    )
    return all(
        compileall.compile_file(str(path), quiet=1)
        if path.is_file()
        else compileall.compile_dir(str(path), quiet=1)
        for path in targets
    )


def main() -> int:
    checks = {
        "git diff --check": _git_diff_check() == 0,
        "python compileall": _compile_python(),
        "structure policy": check_structure(["--root", str(ROOT)]) == 0,
    }
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    return int(not all(checks.values()))


if __name__ == "__main__":
    sys.exit(main())
