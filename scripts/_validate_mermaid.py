# SPDX-License-Identifier: AGPL-3.0-or-later
"""Extract every ```mermaid block from the docs and validate each with
mermaid-cli (mmdc via npx). Prints a PASS/FAIL line per block and exits
non-zero if any block fails to parse/render.

Usage: python scripts/_validate_mermaid.py [file1.md file2.md ...]
       (defaults to docs/*.md)
"""
from __future__ import annotations

import glob
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)


def blocks(md: Path):
    text = md.read_text()
    # line number of each match start
    for m in FENCE.finditer(text):
        line = text[: m.start()].count("\n") + 1
        yield line, m.group(1)


def validate(code: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "g.mmd"
        out = Path(d) / "g.svg"
        src.write_text(code)
        try:
            r = subprocess.run(
                ["npx", "-y", "@mermaid-js/mermaid-cli@latest",
                 "-i", str(src), "-o", str(out)],
                capture_output=True, text=True, timeout=240,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if r.returncode == 0 and out.exists():
            return True, ""
        return False, (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else "unknown error"


def main() -> int:
    args = sys.argv[1:]
    files = [Path(a).resolve() for a in args] if args else sorted((ROOT / "docs").glob("*.md"))
    total = fails = 0
    for f in files:
        bs = list(blocks(f))
        if not bs:
            continue
        print(f"\n=== {f.relative_to(ROOT)} ({len(bs)} mermaid block(s)) ===")
        for line, code in bs:
            total += 1
            ok, err = validate(code)
            if ok:
                print(f"  [PASS] block @ line {line}")
            else:
                fails += 1
                print(f"  [FAIL] block @ line {line}: {err}")
    print(f"\nRESULT: {total - fails}/{total} mermaid blocks valid, {fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
