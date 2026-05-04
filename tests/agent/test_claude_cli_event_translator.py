"""Unit tests for agent.claude_cli.event_translator."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent.claude_cli.errors import OAuthExpired, SubprocessCrashed
from agent.claude_cli.event_translator import EventTranslator


def _delta_text(text: str) -> dict[str, Any]:
    return {
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": text},
        },
    }


def _delta_thinking(text: str) -> dict[str, Any]:
    return {
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": text},
        },
    }


def _result(subtype: str = "success", **extra) -> dict[str, Any]:
    return {"type": "result", "subtype": subtype, **extra}


def _system() -> dict[str, Any]:
    return {"type": "system", "subtype": "init"}


def _drive(translator: EventTranslator, cc_events: list[dict[str, Any]]) -> list[SimpleNamespace]:
    out: list[SimpleNamespace] = []
    for cc in cc_events:
        out.extend(translator.consume(cc))
        if translator.halted_for_tool or translator.pending_error is not None:
            break
    out.extend(translator.finalize())
    return out


# ─────────────────────────── plain text turn ────────────────────────────


def test_plain_text_turn_emits_expected_events():
    t = EventTranslator(model="sonnet")
    cc_events = [
        _system(),
        _delta_text("Hello"),
        _delta_text(" there"),
        _result(),
    ]
    events = _drive(t, cc_events)
    types = [e.type for e in events]
    # Must have at least one content_block_start/stop pair, text_deltas, and message_stop.
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert "message_delta" in types
    assert "message_stop" in types

    final = t.build_final_message()
    assert final.stop_reason == "end_turn"
    assert len(final.content) == 1
    assert final.content[0].type == "text"
    assert "Hello" in final.content[0].text
    assert "there" in final.content[0].text


def test_text_deltas_are_forwarded_after_lookback():
    """Text should be forwarded with a small lookback buffer to catch openers split across deltas."""
    t = EventTranslator(model="sonnet")
    # Feed enough text that lookback boundary is crossed (> 24 chars).
    events = []
    events.extend(
        t.consume(_delta_text("The quick brown fox jumps over the lazy dog."))
    )
    # No halt yet; at least some text_delta events should have been produced
    # (minus up to _SAFE_FORWARD_LOOKBACK trailing chars).
    text_events = [e for e in events if e.type == "content_block_delta"]
    assert text_events, "expected text_delta events to be forwarded"


# ──────────────────────────── tool halt ─────────────────────────────


def test_tool_envelope_triggers_halt_and_tool_use_block():
    t = EventTranslator(model="sonnet")
    envelope = (
        '<hermes:tool name="bash" id="t_01">'
        '<arg name="command">pwd</arg>'
        "</hermes:tool>"
    )
    events = _drive(t, [_delta_text(envelope)])
    # Halted.
    assert t.halted_for_tool
    assert t.parsed_tool is not None
    assert t.parsed_tool.name == "bash"
    assert t.parsed_tool.tool_id == "t_01"

    # Message has a tool_use block as last content.
    final = t.build_final_message()
    assert final.stop_reason == "tool_use"
    tool_blocks = [b for b in final.content if getattr(b, "type", None) == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].name == "bash"
    assert tool_blocks[0].id == "t_01"
    assert tool_blocks[0].input == {"command": "pwd"}


def test_text_before_envelope_preserved_in_final_message():
    t = EventTranslator(model="sonnet")
    cc_events = [
        _delta_text("Let me check the directory. "),
        _delta_text(
            '<hermes:tool name="bash" id="t_02"><arg name="command">pwd</arg></hermes:tool>'
        ),
    ]
    _drive(t, cc_events)
    final = t.build_final_message()
    text_blocks = [b for b in final.content if getattr(b, "type", None) == "text"]
    tool_blocks = [b for b in final.content if getattr(b, "type", None) == "tool_use"]
    assert tool_blocks and "bash" == tool_blocks[0].name
    assert text_blocks
    assert text_blocks[0].text.startswith("Let me check the directory.")
    # The envelope characters must not leak into the text block.
    assert "<hermes:tool" not in text_blocks[0].text


def test_envelope_split_across_deltas_still_halts():
    t = EventTranslator(model="sonnet")
    parts = [
        '<hermes:tool name="b',
        'ash" id="t_03"><arg name="command">pwd</arg></her',
        "mes:tool>",
    ]
    events = []
    for p in parts:
        events.extend(t.consume(_delta_text(p)))
        if t.halted_for_tool:
            break
    events.extend(t.finalize())
    assert t.halted_for_tool
    final = t.build_final_message()
    assert any(getattr(b, "type", None) == "tool_use" for b in final.content)


def test_hallucinated_tool_result_after_close_is_discarded():
    t = EventTranslator(model="sonnet")
    # Spike behavior: model emits envelope, then hallucinates a tool_result.
    tainted = (
        '<hermes:tool name="bash" id="t_04"><arg name="command">pwd</arg></hermes:tool>'
        '<hermes:tool_result id="t_04">FAKE</hermes:tool_result>'
        "then more text"
    )
    _drive(t, [_delta_text(tainted)])
    final = t.build_final_message()
    # Only the real tool_use block survives.
    tool_blocks = [b for b in final.content if getattr(b, "type", None) == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].input == {"command": "pwd"}


# ─────────────────────────── thinking blocks ────────────────────────────


def test_thinking_deltas_produce_thinking_block():
    t = EventTranslator(model="sonnet")
    cc_events = [
        _delta_thinking("Let me think... "),
        _delta_thinking("the answer is 42."),
        _delta_text("Hello"),
        _result(),
    ]
    events = _drive(t, cc_events)
    # thinking_delta events must appear.
    thinking_deltas = [
        e
        for e in events
        if e.type == "content_block_delta"
        and getattr(e.delta, "type", None) == "thinking_delta"
    ]
    assert thinking_deltas
    # Final message has a thinking block and a text block.
    final = t.build_final_message()
    types = [getattr(b, "type", None) for b in final.content]
    assert "thinking" in types
    assert "text" in types


# ────────────────────────── error paths ────────────────────────────


def test_result_subtype_failure_sets_pending_error():
    t = EventTranslator(model="sonnet")
    for ev in t.consume(_delta_text("partial")):
        pass
    for ev in t.consume(_result(subtype="error_during_execution")):
        pass
    assert isinstance(t.pending_error, SubprocessCrashed)


def test_oauth_expired_marker_in_stderr_sets_pending_error():
    t = EventTranslator(model="sonnet")
    t.on_subprocess_stderr("Error 401: OAuth token expired. Please run claude auth.")
    assert isinstance(t.pending_error, OAuthExpired)


def test_subprocess_exit_without_result_sets_pending_error():
    t = EventTranslator(model="sonnet")
    t.on_subprocess_exit(returncode=139, stderr_tail="segfault")
    assert isinstance(t.pending_error, SubprocessCrashed)


# ─────────────────────────── misc hygiene ────────────────────────────


def test_assistant_event_does_not_double_text():
    """CC emits both streaming deltas AND a final 'assistant' event with full text.

    Regression guard: the translator must NOT forward text from the
    'assistant' event (would double-count).
    """
    t = EventTranslator(model="sonnet")
    full_text = "Hello world"
    cc_events = [
        _delta_text(full_text),
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": full_text}]}},
        _result(),
    ]
    _drive(t, cc_events)
    final = t.build_final_message()
    text_blocks = [b for b in final.content if getattr(b, "type", None) == "text"]
    assert len(text_blocks) == 1
    assert text_blocks[0].text == full_text  # NOT doubled
