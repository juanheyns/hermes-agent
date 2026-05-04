"""Tests for the claude_cli branch of agent.auxiliary_client.

Uses the same mock `claude` binary as the adapter tests
(``claude_cli_fake.py``) so no real subscription is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent.auxiliary_client import (
    AsyncClaudeCliAuxiliaryClient,
    ClaudeCliAuxiliaryClient,
    _normalize_aux_provider,
    _resolve_auto,
    _try_claude_cli,
    resolve_provider_client,
)

FAKE_PATH = Path(__file__).parent / "claude_cli_fake.py"


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    """Shell stub that execs the fake claude script via the current python."""
    stub = tmp_path / "claude"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import runpy, sys\n"
        f'sys.argv[0] = "{FAKE_PATH}"\n'
        f'runpy.run_path("{FAKE_PATH}", run_name="__main__")\n'
    )
    stub.chmod(0o755)
    # Override the resolver so _try_claude_cli finds our stub.
    monkeypatch.setenv("CCCLI_CLAUDE_BIN", str(stub))
    return str(stub)


# ────────────────────────── alias normalization ──────────────────────────


def test_aliases_normalize_to_canonical():
    assert _normalize_aux_provider("claude-cli") == "claude-cli"
    assert _normalize_aux_provider("claude_cli") == "claude-cli"
    assert _normalize_aux_provider("Claude_CLI") == "claude-cli"
    assert _normalize_aux_provider("cc-cli") == "claude-cli"
    assert _normalize_aux_provider("claudecli") == "claude-cli"
    # Must not stomp the distinct "claude" alias (which maps to anthropic).
    assert _normalize_aux_provider("claude") == "anthropic"


# ────────────────────────── _try_claude_cli ──────────────────────────


def test_try_claude_cli_missing_binary_returns_none(monkeypatch, tmp_path):
    # Scrub every source resolve_claude_bin checks.
    monkeypatch.delenv("CCCLI_CLAUDE_BIN", raising=False)
    monkeypatch.delenv("CLAUDE_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))  # empty dir — claude not on PATH
    client, model = _try_claude_cli()
    assert client is None
    assert model is None


def test_try_claude_cli_returns_wrapper_when_binary_present(
    fake_bin, monkeypatch
):
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    client, model = _try_claude_cli()
    try:
        assert client is not None
        assert isinstance(client, ClaudeCliAuxiliaryClient)
        assert model  # non-empty default
        assert client.api_key == "claude-cli"
    finally:
        if client is not None:
            client.close()


# ────────────────────────── resolve_provider_client ──────────────────────────


def test_resolve_provider_claude_cli_uses_the_wrapper(fake_bin, monkeypatch):
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    client, model = resolve_provider_client("claude-cli")
    try:
        assert isinstance(client, ClaudeCliAuxiliaryClient)
        assert model
    finally:
        if client is not None:
            client.close()


def test_resolve_provider_claude_cli_respects_model_override(fake_bin, monkeypatch):
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    client, model = resolve_provider_client("claude-cli", model="opus")
    try:
        assert model == "opus"
    finally:
        if client is not None:
            client.close()


def test_resolve_provider_claude_cli_missing_binary_warns(monkeypatch, tmp_path, caplog):
    monkeypatch.delenv("CCCLI_CLAUDE_BIN", raising=False)
    monkeypatch.delenv("CLAUDE_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with caplog.at_level("WARNING"):
        client, model = resolve_provider_client("claude-cli")
    assert client is None
    assert any("claude-cli" in rec.message for rec in caplog.records)


def test_resolve_provider_async_mode_wraps_in_async_client(fake_bin, monkeypatch):
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    client, model = resolve_provider_client("claude-cli", async_mode=True)
    try:
        assert isinstance(client, AsyncClaudeCliAuxiliaryClient)
        assert model
    finally:
        if client is not None:
            # AsyncClaudeCliAuxiliaryClient has no close(); close the sync wrapper.
            # Pull the sync adapter out through the chain and close it.
            sync_adapter = client.chat.completions._sync
            sync_real = sync_adapter._client  # inner ClaudeCliClient
            sync_real.close()


# ────────────────────────── auto chain ignores claude-cli ──────────────────────────


def test_auto_chain_never_includes_claude_cli(fake_bin, monkeypatch):
    """Even with the claude binary present, auto resolution must not pick
    claude-cli — it's opt-in only via explicit provider selection."""
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    # Strip anything that would make another provider succeed; we expect
    # _resolve_auto to return (None, None) without touching claude-cli.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    # Prevent _read_main_provider / _read_main_model from succeeding.
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_provider", lambda: ""
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_model", lambda: ""
    )
    client, model = _resolve_auto()
    assert client is None
    assert model is None


# ────────────────────────── end-to-end aux call ──────────────────────────


def test_aux_call_returns_openai_shape(fake_bin, monkeypatch):
    """The aux client must expose .chat.completions.create returning an
    OpenAI-shaped response (.choices[0].message.content)."""
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    monkeypatch.setenv("CCCLI_FAKE_TEXT", "aux reply")
    client, model = resolve_provider_client("claude-cli")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
            max_tokens=256,
        )
        assert hasattr(response, "choices")
        assert len(response.choices) == 1
        msg = response.choices[0].message
        # content is a list of OpenAI-shape content blocks from the transport.
        # For plain text it should surface the text somehow.
        content = msg.content
        if isinstance(content, list):
            text = " ".join(
                str(b.get("text", "")) if isinstance(b, dict) else str(getattr(b, "text", ""))
                for b in content
            )
        else:
            text = str(content or "")
        assert "aux reply" in text
    finally:
        if client is not None:
            client.close()
