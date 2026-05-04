#!/usr/bin/env python3
"""Dump EVERY stream-json event from a real `claude -p` turn.

Used to inventory what metrics & usage fields CC actually exposes, so we
can decide which ones to plumb through the adapter instead of blindly
accepting "no token accounting".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "sonnet"
    session_id = str(uuid.uuid4())
    cmd = [
        "claude", "-p",
        "--verbose",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--tools", "",
        "--strict-mcp-config",
        "--permission-mode", "bypassPermissions",
        "--system-prompt", "Answer in one short sentence.",
        "--model", model,
        "--session-id", session_id,
    ]
    print(f"[probe] model={model} session={session_id}\n", file=sys.stderr)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def dump_stderr():
        for line in proc.stderr:
            if line.strip():
                print(f"[stderr] {line.rstrip()}", file=sys.stderr)

    threading.Thread(target=dump_stderr, daemon=True).start()

    # Send one user message.
    proc.stdin.write(
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "What is 2+2?"},
            "session_id": session_id,
        }) + "\n"
    )
    proc.stdin.flush()
    proc.stdin.close()

    # Collect every stdout line.
    events: list[dict] = []
    t0 = time.monotonic()
    for line in proc.stdout:
        if not line.strip():
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            print(f"[non-json] {line.rstrip()[:200]}")
            continue
        events.append(evt)
        if evt.get("type") == "result":
            break

    proc.wait(timeout=5)
    print(f"\n[probe] captured {len(events)} events in {time.monotonic() - t0:.2f}s\n")

    # Index events by type.
    by_type: dict[str, list[dict]] = {}
    for e in events:
        by_type.setdefault(e.get("type", "<no-type>"), []).append(e)

    print("[probe] event type histogram:")
    for t, lst in sorted(by_type.items()):
        print(f"  {t}: {len(lst)}")

    # Print one representative sample per type (full JSON).
    print("\n[probe] one sample per event type:\n")
    seen: set[str] = set()
    for e in events:
        t = e.get("type")
        if t in seen:
            continue
        seen.add(t)
        print(f"─── {t} ───")
        print(json.dumps(e, indent=2)[:2000])
        print()

    # Specifically hunt for usage/cost/token fields anywhere in the payload.
    print("\n[probe] usage-field hunt:\n")

    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key_lower = str(k).lower()
                if any(marker in key_lower for marker in (
                    "token", "usage", "cost", "cache", "duration",
                    "num_turn", "input_", "output_",
                )):
                    print(f"  {path}.{k} = {v!r}")
                walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")

    for i, e in enumerate(events):
        walk(e, f"event[{i}:{e.get('type')}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
