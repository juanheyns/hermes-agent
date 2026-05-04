#!/usr/bin/env python3
"""Mock ``claude`` binary for testing the claude_cli adapter.

Reads stream-json lines from stdin. On each user message, emits a scripted
sequence of stream-json events on stdout. Behavior is driven by env vars so
each test can shape the mock without editing the file.

Env vars
--------
CCCLI_FAKE_MODE:
  * ``plain`` (default) — respond with a short text block and a success result.
  * ``tool``             — respond with a <hermes:tool> envelope then stop.
  * ``tool_then_plain``  — if input contains ``<hermes:tool_result>`` respond
                           plain, else respond with a tool envelope. Useful for
                           testing the full tool round-trip.
  * ``crash``            — emit a partial text delta then exit rc=2.
  * ``auth_fail``        — write an OAuth-expired line to stderr then exit rc=1.
  * ``echo``             — echo the user text back so we can assert on stdin.

CCCLI_FAKE_TEXT:
  Text to emit in ``plain``/``echo`` mode. Default ``"Hello"``.

CCCLI_FAKE_TOOL_NAME / CCCLI_FAKE_TOOL_ID / CCCLI_FAKE_TOOL_ARG:
  Envelope fields for ``tool`` mode. Defaults: ``bash`` / ``t_01`` / ``pwd``.
"""

from __future__ import annotations

import json
import os
import sys
import uuid


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def handle_plain(user_text: str) -> None:
    emit({"type": "system", "subtype": "init"})
    text = os.environ.get("CCCLI_FAKE_TEXT", "Hello")
    # Emit a plausible message_start with initial usage numbers — mirrors
    # what a real `claude -p` stream-json output looks like (see
    # scripts/claude_cli_event_probe.py output).
    emit(
        {
            "type": "stream_event",
            "event": {
                "type": "message_start",
                "message": {
                    "id": "msg_fake_abc123",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        "input_tokens": 42,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "output_tokens": 1,
                        "service_tier": "standard",
                    },
                },
            },
            "ttft_ms": 123,
        }
    )
    for chunk in [text[: len(text) // 2], text[len(text) // 2 :]]:
        if chunk:
            emit(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": chunk},
                    },
                }
            )
    emit(
        {
            "type": "assistant",
            "message": {
                "id": "msg_fake_abc123",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": text}],
                "usage": {
                    "input_tokens": 42,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 7,
                    "service_tier": "standard",
                },
            },
        }
    )
    emit(
        {
            "type": "result",
            "subtype": "success",
            "duration_ms": 12,
            "duration_api_ms": 9,
            "num_turns": 1,
            "total_cost_usd": 0.00042,
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 42,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 7,
                "service_tier": "standard",
            },
            "modelUsage": {
                "claude-sonnet-4-6": {
                    "inputTokens": 42,
                    "outputTokens": 7,
                    "costUSD": 0.00042,
                }
            },
        }
    )


def handle_echo(user_text: str) -> None:
    emit({"type": "system", "subtype": "init"})
    reply = f"echo: {user_text}"
    emit(
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": reply},
            },
        }
    )
    emit({"type": "result", "subtype": "success", "duration_ms": 3, "num_turns": 1})


def handle_tool(user_text: str) -> None:
    emit({"type": "system", "subtype": "init"})
    name = os.environ.get("CCCLI_FAKE_TOOL_NAME", "bash")
    tid = os.environ.get("CCCLI_FAKE_TOOL_ID", "t_01")
    arg = os.environ.get("CCCLI_FAKE_TOOL_ARG", "pwd")
    envelope = (
        f'<hermes:tool name="{name}" id="{tid}">'
        f'<arg name="command">{arg}</arg>'
        f"</hermes:tool>"
    )
    # Chunk the envelope so the split-across-deltas path is exercised.
    for chunk in (envelope[: len(envelope) // 2], envelope[len(envelope) // 2 :]):
        emit(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": chunk},
                },
            }
        )
    emit({"type": "result", "subtype": "success", "duration_ms": 20, "num_turns": 1})


def handle_crash(user_text: str) -> None:
    emit({"type": "system", "subtype": "init"})
    emit(
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "partial"},
            },
        }
    )
    sys.stdout.flush()
    sys.exit(2)


def handle_auth_fail(user_text: str) -> None:
    sys.stderr.write("Error 401: OAuth token expired. Please run claude auth.\n")
    sys.stderr.flush()
    sys.exit(1)


def handle_tool_then_plain(user_text: str) -> None:
    if "<hermes:tool_result" in user_text:
        handle_plain(user_text)
    else:
        handle_tool(user_text)


def main() -> int:
    mode = os.environ.get("CCCLI_FAKE_MODE", "plain")
    handler = {
        "plain": handle_plain,
        "echo": handle_echo,
        "tool": handle_tool,
        "tool_then_plain": handle_tool_then_plain,
        "crash": handle_crash,
        "auth_fail": handle_auth_fail,
    }.get(mode, handle_plain)

    # Also record the exact argv and env into a file for test inspection.
    trace_path = os.environ.get("CCCLI_FAKE_TRACE")
    if trace_path:
        with open(trace_path, "a") as f:
            f.write(json.dumps({"argv": sys.argv, "pid": os.getpid()}) + "\n")

    # Optional: capture each stdin line (parsed JSON) to a file for tests.
    stdin_capture_path = os.environ.get("CCCLI_FAKE_STDIN_CAPTURE")

    # Read stdin line-by-line; respond after each line per mode.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if stdin_capture_path:
            with open(stdin_capture_path, "a") as f:
                f.write(json.dumps(msg) + "\n")
        content = msg.get("message", {}).get("content", "")
        user_text = content if isinstance(content, str) else json.dumps(content)
        handler(user_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
