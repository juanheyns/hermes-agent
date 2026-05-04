#!/usr/bin/env python3
"""Bisect the --system-prompt of a claude_cli reproducer to find the trigger.

Given a ``*.repro.sh`` file emitted by the claude_cli adapter, this script
extracts the ``--system-prompt`` argument, progressively shrinks it, and
re-runs the command until it succeeds. It then narrows back down to find
the smallest excerpt that flips success → failure.

Strategy
--------
1. **Halving from the end.** The Hermes system prompt is roughly:
   [envelope instructions] + [tool catalog] + [caller-supplied prompt].
   We try keeping only the first N% of the prompt and see when the call
   starts succeeding. This finds the section CONTAINING the trigger.
2. **Linear narrow within the offending section.** Once we know the
   trigger sits between byte X and byte Y, we slide a window to find the
   smallest still-failing prefix.

Outputs:
* Prints success/failure for each tested prompt size.
* Prints the smallest prompt suffix that, when REMOVED, makes the call
  succeed — i.e. the suspected trigger.
* Writes a clean copy of the trigger to ``<repro>.trigger.txt``.

Run:
    python scripts/claude_cli_bisect_prompt.py <path/to/repro.sh>

Caveats
-------
* Each test spawn burns one quota unit. The script stops as soon as it
  has a tight bound (typically 8–12 spawns).
* Session ids are replaced with fresh uuidgen calls so CC won't reject
  reuse.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


def parse_repro(path: Path) -> tuple[list[str], str, str, str]:
    """Return (argv, system_prompt, cwd, stdin) extracted from the repro.sh."""
    text = path.read_text()

    cwd_match = re.search(r"^cd (.+)$", text, re.MULTILINE)
    cwd = cwd_match.group(1).strip() if cwd_match else ""
    if cwd.startswith("'") and cwd.endswith("'"):
        cwd = cwd[1:-1]

    # Find the line that starts the claude command (begins with /, ends in heredoc).
    # The whole command spans multiple lines because of embedded newlines in --system-prompt.
    cmd_start = text.index("\n\n", text.index("set -e")) + 2
    heredoc_start = text.index(f"<<'__HERMES_STDIN__'", cmd_start)
    cmd_blob = text[cmd_start:heredoc_start].rstrip()

    argv = shlex.split(cmd_blob)
    try:
        sp_idx = argv.index("--system-prompt")
    except ValueError:
        raise SystemExit("repro file has no --system-prompt argument")
    system_prompt = argv[sp_idx + 1]

    # Stdin lines from the heredoc.
    stdin_start = text.index("\n", heredoc_start) + 1
    stdin_end = text.index("\n__HERMES_STDIN__", stdin_start)
    stdin = text[stdin_start:stdin_end]

    return argv, system_prompt, cwd, stdin


def replace_session_id(argv: list[str]) -> list[str]:
    """Swap --session-id <fixed> for a fresh uuid so CC accepts the reuse."""
    out = list(argv)
    try:
        idx = out.index("--session-id")
        out[idx + 1] = str(uuid.uuid4())
    except ValueError:
        pass
    return out


def replace_system_prompt(argv: list[str], new_prompt: str) -> list[str]:
    out = list(argv)
    idx = out.index("--system-prompt")
    out[idx + 1] = new_prompt
    return out


def run_once(argv: list[str], cwd: str, stdin: str, timeout_s: float = 60.0) -> tuple[bool, str]:
    """Run the command. Returns (success, error_excerpt)."""
    try:
        proc = subprocess.run(
            argv,
            input=stdin,
            cwd=cwd or None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, "<timeout>"

    # Walk the stream-json output for a result event with is_error=True.
    err_excerpt = ""
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "result":
            if evt.get("is_error"):
                err_excerpt = (evt.get("result") or "").strip()[:240]
                return False, err_excerpt
            return True, ""
    # No result event → likely a spawn or stderr crash.
    return False, (proc.stderr or "<no result event>").strip()[:240]


def bisect(prompt: str, argv: list[str], cwd: str, stdin: str) -> tuple[int, int]:
    """Binary-search for the smallest prefix length that still fails.

    Returns (failing_len, succeeding_len) bracket — failing_len is the
    SHORTEST prefix that still produces the error.
    """
    n = len(prompt)
    print(f"[bisect] full prompt length: {n} chars\n")

    # Sanity check: the full prompt should fail.
    print("[step] full prompt:", end=" ", flush=True)
    ok, msg = run_once(replace_system_prompt(replace_session_id(argv), prompt), cwd, stdin)
    if ok:
        print("succeeded — repro doesn't fail right now (account state may have shifted)")
        sys.exit(2)
    print(f"FAIL — {msg[:100]}")

    # Sanity check: empty prompt should succeed.
    print("[step] empty prompt:", end=" ", flush=True)
    ok, msg = run_once(replace_system_prompt(replace_session_id(argv), ""), cwd, stdin)
    print("OK" if ok else f"FAIL — {msg[:100]}")
    if not ok:
        print("[bisect] even an empty prompt fails — trigger isn't in the prompt")
        sys.exit(3)

    # Binary search: lo always fails, hi always succeeds.
    lo, hi = n, 0
    while lo - hi > 16:
        mid = (lo + hi) // 2
        candidate = prompt[:mid]
        print(f"[step] prefix[:{mid}]:", end=" ", flush=True)
        ok, msg = run_once(replace_system_prompt(replace_session_id(argv), candidate), cwd, stdin)
        if ok:
            hi = mid
            print("OK")
        else:
            lo = mid
            print(f"FAIL — {msg[:80]}")
        # CC has a per-window rate cap; small breathing room helps.
        time.sleep(1.0)

    return lo, hi


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path/to/repro.sh>", file=sys.stderr)
        return 1
    repro = Path(sys.argv[1]).expanduser().resolve()
    if not repro.exists():
        print(f"not found: {repro}", file=sys.stderr)
        return 1

    argv, prompt, cwd, stdin = parse_repro(repro)
    print(f"[parse] argv has {len(argv)} elements")
    print(f"[parse] system_prompt: {len(prompt)} chars")
    print(f"[parse] stdin: {len(stdin)} chars")
    print(f"[parse] cwd: {cwd}\n")

    lo, hi = bisect(prompt, argv, cwd, stdin)
    print(f"\n[bisect] failing length ≤ {lo}, succeeding length ≤ {hi}")
    print(f"[bisect] trigger sits between byte {hi} and byte {lo}\n")

    suspect = prompt[hi:lo]
    out_path = repro.with_suffix(".trigger.txt")
    out_path.write_text(suspect)
    print(f"[bisect] suspected trigger ({len(suspect)} chars) written to {out_path}")
    print("[bisect] preview (first 500 chars):")
    print("---")
    print(suspect[:500])
    print("---")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
