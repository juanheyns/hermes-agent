"""Reconcile Hermes's conversation state with Claude Code's session state.

Hermes passes the full ``messages`` list to every ``messages.create/stream``
call. Claude Code owns its own conversation (driven by ``--session-id`` and
``--resume``) and wants only the incremental user turn per call. This module
bridges the two worlds.

Strategy
--------

* Track an ``acknowledged_cursor`` — how many messages of the caller-supplied
  ``messages`` list have been delivered to CC so far.
* On each call, verify the prefix ``messages[:cursor]`` matches the digests we
  previously recorded. If so, send the tail ``messages[cursor:]`` and bump.
* If the prefix diverges (Hermes compressed or rewrote history), rotate: kill
  the current subprocess, start a fresh CC session, replay the entire
  ``messages`` list as stdin turns, and reset digests.
* Assistant messages in the caller-supplied history are not sent to CC via
  stdin (CC owns its assistant turns); they are only digested.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.claude_cli.errors import ClientBusy, SubprocessCrashed
from agent.claude_cli.subprocess_ import ClaudeSubprocess, LaunchSpec
from agent.claude_cli.tool_protocol import encode_tool_result

logger = logging.getLogger(__name__)


def _digest_message(msg: dict[str, Any]) -> str:
    """Hash the salient shape of a message so prefix divergence is detectable.

    Uses role + a JSON-normalized content projection. Not cryptographic —
    just collision-resistant enough to spot edits to the conversation prefix.
    """
    role = msg.get("role", "")
    content = msg.get("content")
    try:
        payload = json.dumps(
            {"role": role, "content": content}, sort_keys=True, default=str
        )
    except TypeError:
        payload = f"{role}:{content!r}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class SessionState:
    session_id: str
    spec: LaunchSpec
    subprocess: Optional[ClaudeSubprocess] = None
    digests: list[str] = field(default_factory=list)
    pending_tool_id: Optional[str] = None
    in_flight: bool = False


class SessionManager:
    """Owns the single active subprocess and reconciles history deltas."""

    def __init__(
        self,
        *,
        claude_bin: str,
        cwd: Optional[str],
        stderr_ring_bytes: int = 4000,
        kill_grace_s: float = 0.5,
    ) -> None:
        self._claude_bin = claude_bin
        self._cwd = cwd
        self._stderr_ring_bytes = stderr_ring_bytes
        self._kill_grace_s = kill_grace_s
        self._state: Optional[SessionState] = None

    # ─────────────────────── public API ───────────────────────

    def ensure(
        self,
        *,
        model: Optional[str],
        system_prompt: Optional[str],
        effort: Optional[str],
        messages: list[dict[str, Any]],
        append_system_prompt: Optional[str] = None,
        json_schema: Optional[str] = None,
        stderr_observer=None,
    ) -> SessionState:
        """Ensure a subprocess exists and is in sync with ``messages``.

        Returns the current :class:`SessionState`. On first call spawns fresh.
        On subsequent calls, verifies prefix digests and either appends new
        messages via stdin or rotates the session if the prefix diverged.
        """
        if self._state is not None and self._state.in_flight:
            raise ClientBusy("another turn is already in flight on this client")

        if self._state is None:
            self._spawn_new(
                model=model,
                system_prompt=system_prompt,
                append_system_prompt=append_system_prompt,
                effort=effort,
                json_schema=json_schema,
                messages=messages,
                stderr_observer=stderr_observer,
            )
            return self._state

        state = self._state

        # Check whether model / system_prompt / effort / json_schema changed —
        # if they did we rotate (CC freezes these at spawn time, so a change
        # requires a fresh process).
        if (
            state.spec.model != model
            or state.spec.system_prompt != system_prompt
            or state.spec.append_system_prompt != append_system_prompt
            or state.spec.effort != effort
            or state.spec.json_schema != json_schema
        ):
            logger.info("rotating claude session: spawn args changed")
            self._rotate(
                model=model,
                system_prompt=system_prompt,
                append_system_prompt=append_system_prompt,
                effort=effort,
                json_schema=json_schema,
                messages=messages,
                stderr_observer=stderr_observer,
            )
            return self._state

        # Prefix integrity check.
        n = len(state.digests)
        if len(messages) < n or any(
            _digest_message(messages[i]) != state.digests[i] for i in range(n)
        ):
            logger.info(
                "rotating claude session: history prefix diverged (had=%d, now=%d)",
                n,
                len(messages),
            )
            self._rotate(
                model=model,
                system_prompt=system_prompt,
                effort=effort,
                messages=messages,
                stderr_observer=stderr_observer,
            )
            return self._state

        # Subprocess may have died while idle — resurrect with --resume.
        if state.subprocess is None or not state.subprocess.is_alive():
            self._respawn_with_resume(stderr_observer=stderr_observer)

        # Send the tail.
        tail = messages[n:]
        self._send_messages(state, tail)
        return state

    def mark_in_flight(self, state: SessionState, value: bool = True) -> None:
        state.in_flight = value

    def interrupt_and_resume(self, stderr_observer=None) -> None:
        """Kill the current subprocess and respawn with ``--resume``."""
        if self._state is None:
            return
        if self._state.subprocess is not None:
            self._state.subprocess.close(kill=True)
            self._state.subprocess = None
        self._respawn_with_resume(stderr_observer=stderr_observer)

    def send_tool_result(
        self,
        *,
        tool_id: str,
        output: str,
        is_error: bool = False,
    ) -> None:
        """Send a ``<hermes:tool_result>`` envelope on stdin for the current session."""
        if self._state is None or self._state.subprocess is None:
            raise SubprocessCrashed("no active session to send tool_result to")
        envelope = encode_tool_result(tool_id, output, is_error=is_error)
        self._state.subprocess.send_user_line(
            {
                "type": "user",
                "message": {"role": "user", "content": envelope},
                "session_id": self._state.session_id,
            }
        )
        # Tool result occupies one "user message" in our digest list.
        self._state.digests.append(
            _digest_message(
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": output}]}
            )
        )
        self._state.pending_tool_id = None

    def close(self) -> None:
        if self._state is not None and self._state.subprocess is not None:
            try:
                self._state.subprocess.close(kill=True)
            except Exception:
                pass
        self._state = None

    @property
    def state(self) -> Optional[SessionState]:
        return self._state

    # ─────────────────────── internals ───────────────────────

    def _new_spec(
        self,
        *,
        session_id: str,
        model: Optional[str],
        system_prompt: Optional[str],
        effort: Optional[str],
        resume: bool,
        json_schema: Optional[str] = None,
        append_system_prompt: Optional[str] = None,
    ) -> LaunchSpec:
        return LaunchSpec(
            claude_bin=self._claude_bin,
            session_id=session_id,
            model=model,
            system_prompt=system_prompt,
            append_system_prompt=append_system_prompt,
            effort=effort,
            cwd=self._cwd,
            resume=resume,
            json_schema=json_schema,
        )

    def _spawn_new(
        self,
        *,
        model: Optional[str],
        system_prompt: Optional[str],
        effort: Optional[str],
        messages: list[dict[str, Any]],
        stderr_observer,
        append_system_prompt: Optional[str] = None,
        json_schema: Optional[str] = None,
    ) -> None:
        session_id = str(uuid.uuid4())
        spec = self._new_spec(
            session_id=session_id,
            model=model,
            system_prompt=system_prompt,
            append_system_prompt=append_system_prompt,
            effort=effort,
            json_schema=json_schema,
            resume=False,
        )
        proc = ClaudeSubprocess(
            spec,
            stderr_ring_bytes=self._stderr_ring_bytes,
            stderr_observer=stderr_observer,
            kill_grace_s=self._kill_grace_s,
        )
        proc.start()
        self._state = SessionState(session_id=session_id, spec=spec, subprocess=proc)
        self._send_messages(self._state, messages)

    def _rotate(
        self,
        *,
        model: Optional[str],
        system_prompt: Optional[str],
        effort: Optional[str],
        messages: list[dict[str, Any]],
        stderr_observer,
        append_system_prompt: Optional[str] = None,
        json_schema: Optional[str] = None,
    ) -> None:
        if self._state is not None and self._state.subprocess is not None:
            try:
                self._state.subprocess.close(kill=True)
            except Exception:
                pass
        self._state = None
        self._spawn_new(
            model=model,
            system_prompt=system_prompt,
            append_system_prompt=append_system_prompt,
            effort=effort,
            json_schema=json_schema,
            messages=messages,
            stderr_observer=stderr_observer,
        )

    def _respawn_with_resume(self, *, stderr_observer) -> None:
        assert self._state is not None
        spec = self._new_spec(
            session_id=self._state.session_id,
            model=self._state.spec.model,
            system_prompt=self._state.spec.system_prompt,
            append_system_prompt=self._state.spec.append_system_prompt,
            effort=self._state.spec.effort,
            json_schema=self._state.spec.json_schema,
            resume=True,
        )
        proc = ClaudeSubprocess(
            spec,
            stderr_ring_bytes=self._stderr_ring_bytes,
            stderr_observer=stderr_observer,
            kill_grace_s=self._kill_grace_s,
        )
        proc.start()
        self._state.subprocess = proc
        self._state.spec = spec

    def _send_messages(self, state: SessionState, msgs: list[dict[str, Any]]) -> None:
        """Translate ``msgs`` into CC stdin turns and append digests."""
        assert state.subprocess is not None
        for msg in msgs:
            role = msg.get("role")
            if role == "assistant":
                # CC owns assistant turns; never send to stdin. Digest only.
                state.digests.append(_digest_message(msg))
                continue
            if role == "system":
                # System prompts are passed via CLI flag, not in-message.
                state.digests.append(_digest_message(msg))
                continue
            if role != "user":
                # Unknown roles: digest only; do not forward.
                state.digests.append(_digest_message(msg))
                continue

            rendered = render_user_content(msg.get("content"))
            state.subprocess.send_user_line(
                {
                    "type": "user",
                    "message": {"role": "user", "content": rendered},
                    "session_id": state.session_id,
                }
            )
            state.digests.append(_digest_message(msg))


def render_user_content(content: Any) -> "str | list[dict[str, Any]]":
    """Turn a user message's ``content`` into the shape CC stdin expects.

    Returns either a plain string (when the message is all text) or a list of
    Anthropic-native content blocks (for vision/mixed payloads). Claude Code's
    stream-json stdin accepts both shapes and forwards native content blocks
    to the underlying API untouched, so images flow through unchanged.

    Tool results are the one Hermes-specific translation: since
    ``<hermes:tool_result>`` is a text envelope the model was instructed to
    expect, tool_result blocks are rendered into synthetic text blocks at this
    layer rather than forwarded as Anthropic tool_result blocks (which would
    need a matching ``tool_use`` id that CC's session doesn't know about).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    # First pass: convert each block into either a rendered string (for text /
    # tool_result) or a passthrough content-block dict (for image / document /
    # other native Anthropic blocks).
    rendered_blocks: list[dict[str, Any] | str] = []
    for block in content:
        if not isinstance(block, dict):
            rendered_blocks.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            rendered_blocks.append(block.get("text", ""))
        elif btype == "tool_result":
            rendered_blocks.append(_render_tool_result_block(block))
        elif btype in ("image", "document"):
            # Passthrough — Anthropic's native content-block shape flows
            # straight through CC's stream-json stdin to the underlying API.
            rendered_blocks.append(block)
        elif btype == "tool_use":
            # Shouldn't appear on a user-role message, but be defensive.
            rendered_blocks.append(block)
        else:
            # Unknown block type — serialize to text as a last resort so the
            # model at least sees the intent.
            rendered_blocks.append(json.dumps(block, ensure_ascii=False, default=str))

    # If everything rendered to a string, collapse to a single string payload.
    if all(isinstance(b, str) for b in rendered_blocks):
        return "\n\n".join(b for b in rendered_blocks if b)

    # Otherwise emit a content-block list. Strings become text blocks;
    # dicts (images, etc.) pass through verbatim.
    blocks: list[dict[str, Any]] = []
    for b in rendered_blocks:
        if isinstance(b, str):
            if b:
                blocks.append({"type": "text", "text": b})
        else:
            blocks.append(b)
    return blocks


def _render_tool_result_block(block: dict[str, Any]) -> str:
    tool_id = block.get("tool_use_id") or block.get("id") or ""
    inner = block.get("content")
    if isinstance(inner, list):
        inner_text = "\n".join(
            ib.get("text", "")
            for ib in inner
            if isinstance(ib, dict) and ib.get("type") == "text"
        )
    elif isinstance(inner, str):
        inner_text = inner
    else:
        inner_text = str(inner) if inner is not None else ""
    return encode_tool_result(
        tool_id, inner_text, is_error=bool(block.get("is_error"))
    )


# Backward-compatible alias — older callers may import the renamed helper.
_render_user_content_to_text = render_user_content
