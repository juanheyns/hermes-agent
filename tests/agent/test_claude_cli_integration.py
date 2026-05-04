"""Integration tests: AIAgent driven in claude_cli mode against the mock binary.

Verifies the PR 3 plumbing: provider="claude-cli" triggers the embedded
subprocess adapter, and the rest of the anthropic_messages dispatch chain
works unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

FAKE_PATH = Path(__file__).parent / "claude_cli_fake.py"


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


def test_provider_claude_cli_maps_to_anthropic_messages(fake_bin, monkeypatch):
    """provider="claude-cli" should set api_mode="anthropic_messages" and
    wire up a ClaudeCliClient as self._anthropic_client."""
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    monkeypatch.setenv("CCCLI_FAKE_TEXT", "agent says hi")

    from agent.claude_cli import ClaudeCliClient
    from run_agent import AIAgent

    agent = AIAgent(
        provider="claude-cli",
        model="sonnet",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        persist_session=False,
    )
    try:
        assert agent.provider == "claude-cli"
        assert agent.api_mode == "anthropic_messages"
        assert agent._is_anthropic_oauth is False
        assert agent._anthropic_api_key == "claude-cli"
        assert isinstance(agent._anthropic_client, ClaudeCliClient)
    finally:
        try:
            agent._anthropic_client.close()
        except Exception:
            pass


def test_oauth_refresh_is_bypassed_for_claude_cli(fake_bin, monkeypatch):
    """`_try_refresh_anthropic_client_credentials` must be a no-op when
    provider is claude-cli — the method already guards on provider=='anthropic',
    but verifying here catches regressions if that guard ever loosens."""
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")

    from run_agent import AIAgent

    agent = AIAgent(
        provider="claude-cli",
        model="sonnet",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        persist_session=False,
    )
    try:
        # Returns False without touching the client when provider != "anthropic".
        assert agent._try_refresh_anthropic_client_credentials() is False
    finally:
        try:
            agent._anthropic_client.close()
        except Exception:
            pass


def test_anthropic_messages_create_dispatches_through_claude_cli_client(fake_bin, monkeypatch):
    """The existing `_anthropic_messages_create` entry point must drive the
    ClaudeCliClient.messages.create without error and return an
    Anthropic-shaped Message."""
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    monkeypatch.setenv("CCCLI_FAKE_TEXT", "roger that")

    from run_agent import AIAgent

    agent = AIAgent(
        provider="claude-cli",
        model="sonnet",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        persist_session=False,
    )
    try:
        response = agent._anthropic_messages_create(
            {
                "model": "sonnet",
                "messages": [{"role": "user", "content": "hi"}],
                "system": "be terse",
                "max_tokens": 128,
            }
        )
        # SDK-shaped Message.
        assert response.role == "assistant"
        assert response.stop_reason == "end_turn"
        texts = [b for b in response.content if getattr(b, "type", None) == "text"]
        assert texts
        assert "roger that" in texts[0].text
    finally:
        try:
            agent._anthropic_client.close()
        except Exception:
            pass


def test_tool_use_turn_returns_tool_use_stop_reason(fake_bin, monkeypatch):
    """A tool envelope should surface as stop_reason='tool_use' with a
    properly-shaped tool_use content block — consumable by the existing
    anthropic_messages tool-calling logic."""
    monkeypatch.setenv("CCCLI_FAKE_MODE", "tool")
    monkeypatch.setenv("CCCLI_FAKE_TOOL_NAME", "bash")
    monkeypatch.setenv("CCCLI_FAKE_TOOL_ID", "t_999")
    monkeypatch.setenv("CCCLI_FAKE_TOOL_ARG", "ls /tmp")

    from run_agent import AIAgent

    agent = AIAgent(
        provider="claude-cli",
        model="sonnet",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        persist_session=False,
    )
    try:
        response = agent._anthropic_messages_create(
            {
                "model": "sonnet",
                "messages": [{"role": "user", "content": "list /tmp"}],
                "system": "use the envelope",
                "max_tokens": 256,
            }
        )
        assert response.stop_reason == "tool_use"
        tools = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        assert len(tools) == 1
        assert tools[0].name == "bash"
        assert tools[0].id == "t_999"
        assert tools[0].input == {"command": "ls /tmp"}
    finally:
        try:
            agent._anthropic_client.close()
        except Exception:
            pass
