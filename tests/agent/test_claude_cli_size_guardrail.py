"""Tests for the prompt-size guardrail.

When the composed system prompt would exceed the hard limit (default 18KB),
the adapter truncates the tool catalog rather than letting the request hit
Anthropic's third-party-app classifier and fail.
"""

from __future__ import annotations

import logging

import pytest

from agent.claude_cli.client import (
    _compose_system_prompt,
    SystemPromptPolicy,
)


def _make_tools(n: int, description_chars: int = 80) -> list[dict]:
    """Build N synthetic tool defs with descriptions of the given size."""
    desc = "x" * description_chars
    return [
        {
            "name": f"tool_{i}",
            "description": desc,
            "input_schema": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                "required": ["a"],
            },
        }
        for i in range(n)
    ]


# ──────────────────────────── soft / hard limits ────────────────────────────


def test_small_prompt_no_warning(caplog):
    """Composed prompt well under the soft limit should not log anything."""
    caplog.set_level(logging.INFO, logger="agent.claude_cli.client")
    sp, ap = _compose_system_prompt(
        caller_system="be concise",
        tools=_make_tools(5),
        policy=SystemPromptPolicy.REPLACE,
    )
    assert sp is not None
    # Length should be modest — no warnings expected.
    assert not any(
        "claude_cli composed system prompt" in r.message
        or "third-party-app classifier" in r.message
        for r in caplog.records
    )


def test_medium_prompt_emits_soft_warning(caplog, monkeypatch):
    """Between soft and hard, log INFO but don't truncate."""
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PROMPT_SOFT", "200")
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PROMPT_HARD", "10000")
    caplog.set_level(logging.INFO, logger="agent.claude_cli.client")
    sp, _ = _compose_system_prompt(
        caller_system="x" * 500,
        tools=_make_tools(20),
        policy=SystemPromptPolicy.REPLACE,
    )
    # Catalog should NOT be truncated.
    assert "[…tool catalog truncated" not in (sp or "")
    # An INFO log line should be present.
    assert any(
        "soft warn threshold" in r.message
        for r in caplog.records
    )


def test_oversize_prompt_truncates_catalog(caplog, monkeypatch):
    """Above the hard limit, the catalog is truncated automatically."""
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PROMPT_SOFT", "100")
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PROMPT_HARD", "1000")
    caplog.set_level(logging.WARNING, logger="agent.claude_cli.client")
    sp, _ = _compose_system_prompt(
        caller_system="x" * 100,
        tools=_make_tools(50, description_chars=120),
        policy=SystemPromptPolicy.REPLACE,
    )
    assert sp is not None
    assert "[…tool catalog truncated" in sp
    # Hard limit allows some slack for envelope + caller text + truncation marker.
    assert len(sp) <= 1100
    assert any(
        "truncated tool catalog from" in r.message
        for r in caplog.records
    )


def test_truncation_preserves_envelope_and_caller_text(monkeypatch):
    """Even when severely truncated, the Hermes envelope + caller content stay intact."""
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PROMPT_HARD", "1500")
    caller = "Caller-specific instructions that must be kept verbatim."
    sp, _ = _compose_system_prompt(
        caller_system=caller,
        tools=_make_tools(80, description_chars=100),
        policy=SystemPromptPolicy.REPLACE,
    )
    assert sp is not None
    assert "<hermes:tool" in sp  # envelope contract preserved
    assert caller in sp           # caller text preserved
    assert "[…tool catalog truncated" in sp


def test_env_var_can_disable_truncation(monkeypatch):
    """A very high hard limit means nothing gets truncated."""
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PROMPT_HARD", "10000000")
    sp, _ = _compose_system_prompt(
        caller_system="x" * 100,
        tools=_make_tools(50, description_chars=200),
        policy=SystemPromptPolicy.REPLACE,
    )
    assert sp is not None
    assert "[…tool catalog truncated" not in sp


def test_lower_hard_wins_over_default_soft(monkeypatch):
    """If only hard is set lower than the default soft, hard wins.

    Budget must leave room for the envelope contract (~600 chars) plus
    caller text plus framing — otherwise the truncation function correctly
    returns an empty catalog with no marker.
    """
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PROMPT_HARD", "1200")
    sp, _ = _compose_system_prompt(
        caller_system="x" * 50,
        tools=_make_tools(20, description_chars=80),
        policy=SystemPromptPolicy.REPLACE,
    )
    assert sp is not None
    assert "[…tool catalog truncated" in sp
    # Hard limit + small overhead for the truncation marker line.
    assert len(sp) <= 1300


def test_env_var_disables_compact_catalog(monkeypatch):
    """HERMES_CLAUDE_CLI_CATALOG_COMPACT=0 restores verbose schema dumps."""
    monkeypatch.setenv("HERMES_CLAUDE_CLI_CATALOG_COMPACT", "0")
    # Use a generous hard limit so we don't truncate before observing the form.
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PROMPT_HARD", "100000")
    sp, _ = _compose_system_prompt(
        caller_system="",
        tools=_make_tools(3),
        policy=SystemPromptPolicy.REPLACE,
    )
    assert sp is not None
    assert '"properties"' in sp  # verbose schema marker present


def test_env_var_truncates_descriptions(monkeypatch):
    """HERMES_CLAUDE_CLI_MAX_DESC_CHARS narrows tool descriptions."""
    monkeypatch.setenv("HERMES_CLAUDE_CLI_MAX_DESC_CHARS", "20")
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PROMPT_HARD", "100000")
    long_desc = "alpha beta gamma delta epsilon zeta eta theta iota"
    sp, _ = _compose_system_prompt(
        caller_system="",
        tools=[{"name": "x", "description": long_desc, "input_schema": {"type": "object", "properties": {}}}],
        policy=SystemPromptPolicy.REPLACE,
    )
    assert sp is not None
    # Description gets cropped to 20 chars + ellipsis.
    assert "alpha beta gamma " in sp
    assert "epsilon" not in sp


def test_extreme_low_hard_drops_catalog_entirely(monkeypatch):
    """When hard is so low that even envelope+caller can't fit, the catalog
    drops cleanly to empty (no half-rendered line, no marker)."""
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PROMPT_HARD", "300")
    sp, _ = _compose_system_prompt(
        caller_system="x" * 50,
        tools=_make_tools(10),
        policy=SystemPromptPolicy.REPLACE,
    )
    assert sp is not None
    # Catalog should be entirely absent — no leftover JSON or marker fragment.
    assert "Available tools:" not in sp
    assert "tool_0" not in sp
