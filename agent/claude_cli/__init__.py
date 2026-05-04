"""Embedded `claude -p` subprocess adapter.

Exposes :class:`ClaudeCliClient`, an Anthropic-SDK-shaped client that drives a
persistent ``claude -p --input-format stream-json --output-format stream-json``
subprocess for model calls. Used when ``api_mode == "claude_cli"``.

Public surface is intentionally narrow — only :class:`ClaudeCliClient` and the
:func:`build_claude_cli_client` factory — so the adapter stays drop-in for the
existing Anthropic SDK call sites in ``run_agent.py``.

See ``plans/claude-cli-adapter-spec.md`` for the full design.
"""

from agent.claude_cli.client import (
    ClaudeCliClient,
    SystemPromptPolicy,
    build_claude_cli_client,
)
from agent.claude_cli.errors import (
    ApiError,
    ClaudeCliError,
    ClientBusy,
    OAuthExpired,
    QuotaExceeded,
    SpawnFailed,
    SubprocessCrashed,
    ToolParseError,
    TurnTimeout,
)

__all__ = [
    "ClaudeCliClient",
    "SystemPromptPolicy",
    "build_claude_cli_client",
    "ApiError",
    "ClaudeCliError",
    "ClientBusy",
    "OAuthExpired",
    "QuotaExceeded",
    "SpawnFailed",
    "SubprocessCrashed",
    "ToolParseError",
    "TurnTimeout",
]
