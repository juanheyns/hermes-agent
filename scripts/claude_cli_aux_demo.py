#!/usr/bin/env python3
"""Live demo: the claude_cli auxiliary-client branch against the real `claude`.

Runs one aux call through ``resolve_provider_client("claude-cli")`` to
verify the OpenAI-shape adapter end-to-end. Requires an installed and
authenticated ``claude`` CLI.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent.auxiliary_client import resolve_provider_client  # noqa: E402


def main() -> int:
    if not shutil.which("claude"):
        print("claude CLI not on PATH — skipping", file=sys.stderr)
        return 2

    client, model = resolve_provider_client("claude-cli", model="sonnet")
    if client is None:
        print("resolve_provider_client returned no client", file=sys.stderr)
        return 1

    try:
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Answer in one short sentence."},
                {"role": "user", "content": "What is the capital of Japan?"},
            ],
            max_tokens=128,
        )
        dt = time.monotonic() - t0
        msg = response.choices[0].message
        content = msg.content
        if isinstance(content, list):
            text = " ".join(
                (b.get("text", "") if isinstance(b, dict) else str(getattr(b, "text", "")))
                for b in content
            )
        else:
            text = str(content or "")
        print(f"finish_reason={response.choices[0].finish_reason} duration={dt:.2f}s")
        print(f"text: {text!r}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
