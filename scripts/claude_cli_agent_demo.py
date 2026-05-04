#!/usr/bin/env python3
"""Live demo: AIAgent with provider="claude-cli" against the real `claude`.

Instantiates the full AIAgent in claude_cli mode and runs a trivial
messages.create end-to-end. Sanity-checks that the PR 3 wiring is correct
when driven through the real (not mocked) stack.

Usage:
    python scripts/claude_cli_agent_demo.py [sonnet|opus|haiku]
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


def main() -> int:
    if not shutil.which("claude"):
        print("claude CLI not on PATH — skipping", file=sys.stderr)
        return 2

    model = sys.argv[1] if len(sys.argv) > 1 else "sonnet"

    from run_agent import AIAgent

    agent = AIAgent(
        provider="claude-cli",
        model=model,
        quiet_mode=False,
        skip_context_files=True,
        skip_memory=True,
        persist_session=False,
    )
    try:
        print(f"[demo] api_mode={agent.api_mode}  provider={agent.provider}  "
              f"client={type(agent._anthropic_client).__name__}")

        t0 = time.monotonic()
        response = agent._anthropic_messages_create(
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Reply with exactly one short sentence about the color of the sky."}
                ],
                "system": "Be terse.",
                "max_tokens": 256,
            }
        )
        dt = time.monotonic() - t0
        print(f"[demo] stop_reason={response.stop_reason}  duration={dt:.2f}s")
        for b in response.content:
            if getattr(b, "type", None) == "text":
                print(f"[demo] text: {b.text!r}")
        print("[demo] done")
    finally:
        try:
            agent._anthropic_client.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
