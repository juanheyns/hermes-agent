"""Tests for token/cost/duration metrics plumbing.

Covers the EventTranslator's capture of Anthropic-shape usage fields plus
the CC-specific extras (cost_usd, durations, ttft_ms, modelUsage). These
numbers flow into the final Message and are read by Hermes's existing
``normalize_usage`` helper for cost tracking.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.claude_cli.event_translator import EventTranslator

FAKE_PATH = Path(__file__).parent / "claude_cli_fake.py"


# ───────────────────────── EventTranslator metrics ─────────────────────────


def _stream_message_start(usage: dict, *, message_id="msg_01", model="claude-sonnet-4-6", ttft_ms=None):
    evt = {
        "type": "stream_event",
        "event": {
            "type": "message_start",
            "message": {
                "id": message_id,
                "model": model,
                "role": "assistant",
                "usage": usage,
            },
        },
    }
    if ttft_ms is not None:
        evt["ttft_ms"] = ttft_ms
    return evt


def _stream_text_delta(text: str):
    return {
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": text},
        },
    }


def _result(usage: dict, **extra):
    return {"type": "result", "subtype": "success", "usage": usage, **extra}


def test_usage_captured_from_message_start():
    t = EventTranslator(model="sonnet")
    for _ in t.consume(_stream_message_start(
        {"input_tokens": 100, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    )):
        pass
    assert t._usage_input_tokens == 100
    assert t._usage_output_tokens == 1


def test_usage_updates_as_tokens_grow():
    """Running output_tokens should be picked up from message_delta-style events."""
    t = EventTranslator(model="sonnet")
    # Initial
    for _ in t.consume(_stream_message_start(
        {"input_tokens": 100, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    )):
        pass
    # message_delta with more output tokens
    for _ in t.consume({
        "type": "stream_event",
        "event": {
            "type": "message_delta",
            "usage": {"input_tokens": 100, "output_tokens": 25, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        },
    }):
        pass
    assert t._usage_output_tokens == 25


def test_result_event_populates_cost_duration_and_final_usage():
    t = EventTranslator(model="sonnet")
    # Seed some initial numbers
    for _ in t.consume(_stream_message_start(
        {"input_tokens": 100, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        ttft_ms=420,
    )):
        pass
    for _ in t.consume(_stream_text_delta("hi")):
        pass
    # Authoritative result
    for _ in t.consume(_result(
        {"input_tokens": 138, "output_tokens": 12, "cache_creation_input_tokens": 5, "cache_read_input_tokens": 3, "service_tier": "standard"},
        duration_ms=1660,
        duration_api_ms=2482,
        num_turns=1,
        total_cost_usd=0.000998,
        stop_reason="end_turn",
        modelUsage={"claude-sonnet-4-6": {"inputTokens": 138, "outputTokens": 12, "costUSD": 0.000594}},
    )):
        pass
    for _ in t.finalize():
        pass
    msg = t.build_final_message()
    # Anthropic-shape fields (consumed by normalize_usage downstream)
    assert msg.usage.input_tokens == 138
    assert msg.usage.output_tokens == 12
    assert msg.usage.cache_creation_input_tokens == 5
    assert msg.usage.cache_read_input_tokens == 3
    # CC-specific extras
    assert msg.ccli_cost_usd == pytest.approx(0.000998)
    assert msg.ccli_duration_ms == 1660
    assert msg.ccli_duration_api_ms == 2482
    assert msg.ccli_ttft_ms == 420
    assert msg.ccli_num_turns == 1
    assert msg.ccli_service_tier == "standard"
    assert msg.ccli_model_usage == {
        "claude-sonnet-4-6": {"inputTokens": 138, "outputTokens": 12, "costUSD": 0.000594}
    }


def test_real_anthropic_message_id_and_model_replace_placeholders():
    """The synthetic msg_<uuid> id and caller's alias should be replaced by
    the canonical Anthropic msg id and concrete model slug when available."""
    t = EventTranslator(model="sonnet")  # alias — expect concrete slug to win
    for _ in t.consume(_stream_message_start(
        {"input_tokens": 50, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        message_id="msg_01ABCxyz",
        model="claude-sonnet-4-6",
    )):
        pass
    for _ in t.consume(_result(
        {"input_tokens": 50, "output_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    )):
        pass
    for _ in t.finalize():
        pass
    msg = t.build_final_message()
    assert msg.id == "msg_01ABCxyz"
    assert msg.model == "claude-sonnet-4-6"


def test_message_delta_output_tokens_reflects_final_count():
    t = EventTranslator(model="sonnet")
    for _ in t.consume(_stream_message_start(
        {"input_tokens": 10, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    )):
        pass
    for _ in t.consume(_result(
        {"input_tokens": 10, "output_tokens": 99, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    )):
        pass
    events = list(t.finalize())
    md = [e for e in events if e.type == "message_delta"]
    assert md
    assert md[0].usage.output_tokens == 99


def test_absorb_usage_ignores_non_dict_and_missing_keys():
    t = EventTranslator(model="sonnet")
    # Non-dict: no crash, no change.
    t._absorb_usage(None)
    t._absorb_usage("garbage")
    t._absorb_usage(42)
    assert t._usage_input_tokens == 0
    # Partial dict: only sets what it can find.
    t._absorb_usage({"output_tokens": 5})
    assert t._usage_output_tokens == 5
    assert t._usage_input_tokens == 0


def test_largest_seen_wins_monotonic_counters():
    """A late 'result' event with slightly stale counts must not regress."""
    t = EventTranslator(model="sonnet")
    t._absorb_usage({"output_tokens": 10})
    t._absorb_usage({"output_tokens": 5})   # stale; should not overwrite
    assert t._usage_output_tokens == 10


# ───────────────── end-to-end via mock claude through full adapter ─────────────────


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    stub = tmp_path / "claude"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import runpy, sys\n"
        f'sys.argv[0] = "{FAKE_PATH}"\n'
        f'runpy.run_path("{FAKE_PATH}", run_name="__main__")\n'
    )
    stub.chmod(0o755)
    monkeypatch.setenv("CCCLI_CLAUDE_BIN", str(stub))
    return str(stub)


def test_end_to_end_usage_cost_and_duration_populated(fake_bin, monkeypatch):
    """Response from ClaudeCliClient.messages.create carries the real
    usage numbers and CC cost from the mock's scripted output."""
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    monkeypatch.setenv("CCCLI_FAKE_TEXT", "Hello world")

    from agent.claude_cli import build_claude_cli_client

    client = build_claude_cli_client(
        model_default="sonnet", claude_bin=fake_bin, turn_budget_s=15
    )
    try:
        response = client.messages.create(
            messages=[{"role": "user", "content": "hi"}],
            system="",
        )
        # Anthropic-shape usage — values come from the mock's scripted output.
        assert response.usage.input_tokens == 42
        assert response.usage.output_tokens == 7
        # CC-specific extras.
        assert response.ccli_cost_usd == pytest.approx(0.00042)
        assert response.ccli_duration_ms == 12
        assert response.ccli_duration_api_ms == 9
        assert response.ccli_ttft_ms == 123
        assert response.ccli_num_turns == 1
        # Canonical msg id and concrete model carried through.
        assert response.id == "msg_fake_abc123"
        assert response.model == "claude-sonnet-4-6"
    finally:
        client.close()


def test_normalize_usage_reads_claude_cli_response_correctly(fake_bin, monkeypatch):
    """Hermes's own normalize_usage helper must produce correct canonical
    counts when fed a claude_cli response (drives cost tracking)."""
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    monkeypatch.setenv("CCCLI_FAKE_TEXT", "testing")

    from agent.claude_cli import build_claude_cli_client
    from agent.usage_pricing import normalize_usage

    client = build_claude_cli_client(
        model_default="sonnet", claude_bin=fake_bin, turn_budget_s=15
    )
    try:
        response = client.messages.create(
            messages=[{"role": "user", "content": "hi"}],
            system="",
        )
        canonical = normalize_usage(
            response.usage,
            provider="claude-cli",
            api_mode="anthropic_messages",
        )
        assert canonical.input_tokens == 42
        assert canonical.output_tokens == 7
        assert canonical.cache_read_tokens == 0
        assert canonical.cache_write_tokens == 0
    finally:
        client.close()
