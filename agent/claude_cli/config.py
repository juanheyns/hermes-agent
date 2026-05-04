"""Small helpers for claude_cli adapter configuration.

Kept deliberately tiny: the full config surface belongs in
``hermes_cli/config.py`` once PR 3 wires the adapter in. This module only
resolves the binary path and holds sensible defaults used during module
construction.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Optional

from agent.claude_cli.errors import SpawnFailed


@dataclass(frozen=True)
class ClaudeCliConfig:
    claude_bin: str
    cwd: Optional[str] = None
    request_timeout_s: float = 600.0
    turn_budget_s: float = 300.0
    kill_grace_s: float = 0.5
    stderr_ring_bytes: int = 4000
    max_text_buffer_bytes: int = 4 * 1024 * 1024
    malformed_tool_scan_chars: int = 1024


def resolve_claude_bin(explicit: Optional[str] = None) -> str:
    """Resolve the path to the ``claude`` binary.

    Precedence: explicit arg → ``CCCLI_CLAUDE_BIN`` env → ``CLAUDE_BIN`` env →
    ``shutil.which("claude")``. Raises :class:`SpawnFailed` if not found.
    """
    candidate = (
        explicit
        or os.environ.get("CCCLI_CLAUDE_BIN")
        or os.environ.get("CLAUDE_BIN")
        or shutil.which("claude")
    )
    if not candidate:
        raise SpawnFailed(
            "The 'claude' CLI is not installed or not on PATH. "
            "Install it with: npm install -g @anthropic-ai/claude-code"
        )
    return candidate
