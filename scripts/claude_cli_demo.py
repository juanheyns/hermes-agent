#!/usr/bin/env python3
"""End-to-end demo of the claude_cli adapter (the real one, not the mock).

Mirrors ``scripts/claude_cli_spike.py`` but drives the actual
``agent.claude_cli.ClaudeCliClient`` so you can exercise the Anthropic-shaped
surface the way ``run_agent.py`` will.

Covers:

  1. A plain text turn via ``client.messages.create``.
  2. Streaming deltas via ``client.messages.stream`` with ``get_final_message``.
  3. A tool-envelope round-trip: the model is instructed to emit
     ``<hermes:tool>`` envelopes; we parse the tool_use block, run the command
     (safelisted), feed a ``tool_result`` back, and check continuation.

Run:
    python scripts/claude_cli_demo.py [sonnet|opus|haiku]

Requires a working ``claude`` CLI with valid auth. Skips cleanly if not.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Make the repo root importable when run directly.
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent.claude_cli import build_claude_cli_client  # noqa: E402


SAFE_COMMANDS = ("ls", "pwd", "date", "echo", "whoami", "uname")


def safe_run(cmd: str) -> str:
    head = (cmd.strip().split() or [""])[0]
    if head not in SAFE_COMMANDS:
        return f"[demo-sandboxed] refused to run '{head}'. Pretend output: ok."
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=5
        )
        return (r.stdout + r.stderr)[:1500]
    except subprocess.TimeoutExpired:
        return "[demo] command timed out"


def demo_create(client, model: str) -> None:
    print(f"\n=== 1. create() with model={model} ===")
    t0 = time.monotonic()
    msg = client.messages.create(
        model=model,
        system="You are a terse assistant.",
        messages=[{"role": "user", "content": "Say hi in 3 words."}],
    )
    dt = time.monotonic() - t0
    print(f"  stop_reason={msg.stop_reason} duration={dt:.2f}s")
    for b in msg.content:
        print(f"  [{b.type}] {getattr(b, 'text', '')!r}")


def demo_stream(client, model: str) -> None:
    print(f"\n=== 2. stream() with model={model} ===")
    t0 = time.monotonic()
    first_token_at = None
    with client.messages.stream(
        model=model,
        system="You are a terse assistant.",
        messages=[{"role": "user", "content": "Count 1 to 5 with spaces."}],
    ) as stream:
        for event in stream:
            if first_token_at is None and event.type == "content_block_delta":
                delta_type = getattr(event.delta, "type", None)
                if delta_type == "text_delta":
                    first_token_at = time.monotonic() - t0
        final = stream.get_final_message()
    print(f"  first-text-token latency: {first_token_at!s}")
    print(f"  stop_reason={final.stop_reason}")
    for b in final.content:
        if getattr(b, "type", None) == "text":
            print(f"  [text] {b.text!r}")


def demo_tool_roundtrip(client, model: str) -> None:
    print(f"\n=== 3. tool-envelope round-trip with model={model} ===")
    history = [
        {
            "role": "user",
            "content": "What is the current working directory? Use the <hermes:tool> envelope.",
        }
    ]
    msg = client.messages.create(
        model=model,
        system="",  # adapter prepends the envelope instructions automatically
        messages=history,
    )
    print(f"  turn 1 stop_reason={msg.stop_reason}")
    tool_blocks = [b for b in msg.content if getattr(b, "type", None) == "tool_use"]
    if not tool_blocks:
        print("  !! model did not emit a tool envelope")
        for b in msg.content:
            if getattr(b, "type", None) == "text":
                print(f"     text was: {b.text!r}")
        return
    tb = tool_blocks[0]
    print(f"  parsed tool: name={tb.name} id={tb.id} input={tb.input}")

    cmd = tb.input.get("command", "")
    output = safe_run(cmd)
    print(f"  tool output: {output!r}")

    history.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tb.id,
                    "name": tb.name,
                    "input": tb.input,
                }
            ],
        }
    )
    history.append(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tb.id, "content": output}
            ],
        }
    )
    msg2 = client.messages.create(
        model=model,
        system="",
        messages=history,
    )
    print(f"  turn 2 stop_reason={msg2.stop_reason}")
    for b in msg2.content:
        if getattr(b, "type", None) == "text":
            print(f"  [text] {b.text!r}")


def main() -> int:
    if not shutil.which("claude"):
        print("claude CLI not on PATH — skipping demo", file=sys.stderr)
        return 2

    model = sys.argv[1] if len(sys.argv) > 1 else "sonnet"
    client = build_claude_cli_client(model_default=model, cwd=os.getcwd())
    try:
        demo_create(client, model)
        demo_stream(client, model)
        demo_tool_roundtrip(client, model)
    finally:
        client.close()
    print("\n[demo] done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
