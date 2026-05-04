"""End-to-end tests for ClaudeCliClient against a mock `claude` binary.

Uses ``tests/agent/claude_cli_fake.py`` as the subprocess so these tests don't
need a real Anthropic subscription. Each test sets ``CCCLI_FAKE_MODE`` to
script the mock's response shape.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import pytest

from agent.claude_cli.client import (
    ClaudeCliClient,
    SystemPromptPolicy,
    build_claude_cli_client,
)
from agent.claude_cli.errors import ClientBusy, OAuthExpired, SubprocessCrashed

FAKE_PATH = Path(__file__).parent / "claude_cli_fake.py"


@pytest.fixture
def fake_bin(tmp_path):
    """A shell stub that execs the fake claude script via the current python."""
    stub = tmp_path / "claude"
    stub.write_text(f'#!{sys.executable}\nimport runpy, sys\nsys.argv[0] = "{FAKE_PATH}"\nrunpy.run_path("{FAKE_PATH}", run_name="__main__")\n')
    stub.chmod(0o755)
    return str(stub)


@pytest.fixture
def trace_file(tmp_path):
    p = tmp_path / "trace.jsonl"
    return p


def _read_trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ─────────────────────────── create() ───────────────────────────


def test_create_plain_turn(fake_bin, monkeypatch):
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    monkeypatch.setenv("CCCLI_FAKE_TEXT", "Hello world")
    client = build_claude_cli_client(
        model_default="sonnet", claude_bin=fake_bin, turn_budget_s=15
    )
    try:
        msg = client.messages.create(
            model="sonnet",
            system="You are terse.",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert msg.role == "assistant"
        assert msg.stop_reason == "end_turn"
        # One text content block with the full reply.
        texts = [b for b in msg.content if getattr(b, "type", None) == "text"]
        assert texts
        assert texts[0].text == "Hello world"
    finally:
        client.close()


def test_create_tool_halt_produces_tool_use_block(fake_bin, monkeypatch):
    monkeypatch.setenv("CCCLI_FAKE_MODE", "tool")
    monkeypatch.setenv("CCCLI_FAKE_TOOL_NAME", "bash")
    monkeypatch.setenv("CCCLI_FAKE_TOOL_ID", "t_42")
    monkeypatch.setenv("CCCLI_FAKE_TOOL_ARG", "pwd")
    client = build_claude_cli_client(
        model_default="sonnet", claude_bin=fake_bin, turn_budget_s=15
    )
    try:
        msg = client.messages.create(
            messages=[{"role": "user", "content": "do something"}],
            system="sys",
        )
        assert msg.stop_reason == "tool_use"
        tool_blocks = [b for b in msg.content if getattr(b, "type", None) == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "bash"
        assert tool_blocks[0].id == "t_42"
        assert tool_blocks[0].input == {"command": "pwd"}
    finally:
        client.close()


def test_create_auth_fail_raises_oauth_expired(fake_bin, monkeypatch):
    monkeypatch.setenv("CCCLI_FAKE_MODE", "auth_fail")
    client = build_claude_cli_client(
        model_default="sonnet", claude_bin=fake_bin, turn_budget_s=10
    )
    try:
        with pytest.raises(OAuthExpired):
            client.messages.create(messages=[{"role": "user", "content": "hi"}])
    finally:
        client.close()


def test_create_crash_raises_subprocess_crashed(fake_bin, monkeypatch):
    monkeypatch.setenv("CCCLI_FAKE_MODE", "crash")
    client = build_claude_cli_client(
        model_default="sonnet", claude_bin=fake_bin, turn_budget_s=10
    )
    try:
        with pytest.raises(SubprocessCrashed):
            client.messages.create(messages=[{"role": "user", "content": "hi"}])
    finally:
        client.close()


# ─────────────────────────── streaming ───────────────────────────


def test_stream_yields_deltas_and_final_message(fake_bin, monkeypatch):
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    monkeypatch.setenv("CCCLI_FAKE_TEXT", "abcdef")
    client = build_claude_cli_client(
        model_default="sonnet", claude_bin=fake_bin, turn_budget_s=15
    )
    try:
        with client.messages.stream(
            messages=[{"role": "user", "content": "hi"}], system=""
        ) as stream:
            event_types = []
            for event in stream:
                event_types.append(event.type)
            final = stream.get_final_message()
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "message_stop" in event_types
        texts = [b for b in final.content if getattr(b, "type", None) == "text"]
        assert texts and texts[0].text == "abcdef"
    finally:
        client.close()


# ─────────────────────────── flag plumbing ───────────────────────────


def test_unsafe_flags_are_never_emitted(fake_bin, monkeypatch, trace_file):
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    monkeypatch.setenv("CCCLI_FAKE_TRACE", str(trace_file))
    client = build_claude_cli_client(
        model_default="opus", claude_bin=fake_bin, turn_budget_s=10
    )
    try:
        client.messages.create(messages=[{"role": "user", "content": "hi"}])
    finally:
        client.close()
    traces = _read_trace(trace_file)
    assert traces, "fake claude never ran"
    argv = traces[0]["argv"]
    argv_joined = " ".join(argv)
    assert "--dangerously-skip-permissions" not in argv_joined
    assert "--allow-dangerously-skip-permissions" not in argv_joined
    assert "--bare" not in argv_joined


def test_required_flags_are_emitted(fake_bin, monkeypatch, trace_file):
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    monkeypatch.setenv("CCCLI_FAKE_TRACE", str(trace_file))
    client = build_claude_cli_client(
        model_default="opus", claude_bin=fake_bin, turn_budget_s=10
    )
    try:
        client.messages.create(messages=[{"role": "user", "content": "hi"}])
    finally:
        client.close()
    traces = _read_trace(trace_file)
    argv = traces[0]["argv"]
    # argv[0] gets overridden to FAKE_PATH by the stub; the flags follow.
    assert "-p" in argv
    assert "--verbose" in argv
    assert "--input-format" in argv
    assert "--output-format" in argv
    # --tools "" is two args: "--tools" then an empty string.
    idx = argv.index("--tools")
    assert argv[idx + 1] == ""
    assert "--permission-mode" in argv
    pm_idx = argv.index("--permission-mode")
    assert argv[pm_idx + 1] == "bypassPermissions"


def test_resume_flag_used_on_respawn(fake_bin, monkeypatch, trace_file):
    """After a tool halt, the next call should respawn with --resume."""
    monkeypatch.setenv("CCCLI_FAKE_TRACE", str(trace_file))
    monkeypatch.setenv("CCCLI_FAKE_MODE", "tool_then_plain")
    monkeypatch.setenv("CCCLI_FAKE_TOOL_ID", "t_resume")
    monkeypatch.setenv("CCCLI_FAKE_TEXT", "done")
    client = build_claude_cli_client(
        model_default="sonnet", claude_bin=fake_bin, turn_budget_s=10
    )
    try:
        msg1 = client.messages.create(
            messages=[{"role": "user", "content": "first"}], system="sys"
        )
        assert msg1.stop_reason == "tool_use"
        # Now act like Hermes: send a tool_result back as next user message.
        msg2 = client.messages.create(
            system="sys",
            messages=[
                {"role": "user", "content": "first"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t_resume",
                            "name": "bash",
                            "input": {"command": "pwd"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t_resume",
                            "content": "/tmp",
                        }
                    ],
                },
            ],
        )
        assert msg2.stop_reason == "end_turn"
    finally:
        client.close()
    traces = _read_trace(trace_file)
    # First spawn uses --session-id, respawn after halt uses --resume.
    argvs = [t["argv"] for t in traces]
    joined = [" ".join(a) for a in argvs]
    assert any("--session-id" in a for a in joined)
    assert any("--resume" in a for a in joined)


# ─────────────────────────── concurrency ───────────────────────────


def test_second_concurrent_create_raises_client_busy(fake_bin, monkeypatch):
    """Two streams in flight at once should raise ClientBusy."""
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    monkeypatch.setenv("CCCLI_FAKE_TEXT", "foo")
    client = build_claude_cli_client(
        model_default="sonnet", claude_bin=fake_bin, turn_budget_s=10
    )
    try:
        with client.messages.stream(
            messages=[{"role": "user", "content": "hi"}]
        ) as _stream:
            # Don't drain; try to start another turn — should fail.
            with pytest.raises(ClientBusy):
                client.messages.create(messages=[{"role": "user", "content": "x"}])
    finally:
        client.close()
