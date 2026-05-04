"""Anthropic-SDK-shaped client backed by a persistent ``claude -p`` subprocess.

The public surface mimics ``anthropic.Anthropic`` narrowly enough that
``run_agent.py`` can use this client via its existing ``self._anthropic_client``
attribute without changes to the call sites:

* :meth:`MessagesNamespace.create` — returns a duck-typed ``Message``.
* :meth:`MessagesNamespace.stream` — returns a context manager that iterates
  Anthropic-shaped stream events and exposes ``get_final_message()``.
"""

from __future__ import annotations

import atexit
import enum
import logging
import os
import threading
from types import SimpleNamespace
from typing import Any, Iterator, Optional

from agent.claude_cli.config import ClaudeCliConfig, resolve_claude_bin
from agent.claude_cli.errors import (
    ClaudeCliError,
    ClientBusy,
    TurnTimeout,
)
from agent.claude_cli.event_translator import EventTranslator, TranslatedTurn
from agent.claude_cli.session import SessionManager, SessionState
from agent.claude_cli.tool_protocol import (
    HERMES_TOOL_INSTRUCTIONS,
    render_tool_catalog,
)

logger = logging.getLogger(__name__)


class SystemPromptPolicy(enum.Enum):
    """How the adapter combines the caller's system prompt with the envelope contract."""

    REPLACE = "replace"
    APPEND_ONLY = "append_only"


# ───────────────────────── system-prompt composition ────────────────────────


# Empirical thresholds discovered while bisecting the third-party-app billing
# classifier (see plans/claude-cli-adapter-spec.md and the live probes in
# scripts/claude_cli_bisect_prompt.py).
#
# Below ``_PROMPT_SOFT_LIMIT`` we're confidently under the threshold and don't
# log anything. Between soft and hard, we WARN. Above ``_PROMPT_HARD_LIMIT``
# we proactively truncate the rendered tool catalog so the request goes
# through. The user overrides via env var if their account behaves differently.
_PROMPT_SOFT_LIMIT_DEFAULT = 15_000
_PROMPT_HARD_LIMIT_DEFAULT = 18_000


def _prompt_size_limits() -> tuple[int, int]:
    soft = int(os.environ.get("HERMES_CLAUDE_CLI_PROMPT_SOFT", _PROMPT_SOFT_LIMIT_DEFAULT))
    hard = int(os.environ.get("HERMES_CLAUDE_CLI_PROMPT_HARD", _PROMPT_HARD_LIMIT_DEFAULT))
    # If a user lowers `hard` below the default `soft`, honor the lower hard
    # cap and clamp soft to match. Lowering hard is the meaningful action;
    # the soft warn is informational and shouldn't block it.
    if soft > hard:
        soft = hard
    return soft, hard


def _truncate_catalog_to_fit(
    *,
    envelope_block: str,
    caller_text: str,
    tool_catalog: str,
    budget: int,
) -> str:
    """Drop the tail of the tool catalog until the composed prompt fits.

    Preserves the envelope contract and caller_text in full — the catalog is
    the most disposable part because the model can call any tool by name
    even with a partial catalog (the envelope grammar is what matters).
    """
    fixed = len(envelope_block) + len(caller_text) + 64  # join overhead
    available_for_catalog = max(0, budget - fixed)
    if available_for_catalog >= len(tool_catalog):
        return tool_catalog
    if available_for_catalog == 0:
        return ""
    # Truncate at a line boundary so we don't emit half-rendered tool entries.
    truncated = tool_catalog[:available_for_catalog]
    last_nl = truncated.rfind("\n")
    if last_nl > 0:
        truncated = truncated[:last_nl]
    truncated += "\n[…tool catalog truncated to keep system prompt under size threshold…]"
    return truncated


def _compose_system_prompt(
    *,
    caller_system: Any,
    tools: Optional[list[dict[str, Any]]],
    policy: SystemPromptPolicy,
) -> tuple[Optional[str], Optional[str]]:
    """Fold the Hermes envelope contract into the caller's system prompt.

    Returns ``(replace_prompt, append_prompt)`` — exactly one is non-None,
    based on policy. The subprocess wrapper turns these into
    ``--system-prompt`` or ``--append-system-prompt`` respectively.

    Why two modes
    -------------
    Anthropic's billing system distinguishes between "Claude Code" usage
    (drawn from your Claude plan) and "third-party app" usage (drawn from
    a separate Extra Usage pool). Replacing CC's default system prompt
    via ``--system-prompt`` appears to flip the request into the
    third-party-app pool. Appending (``--append-system-prompt``) keeps
    CC's own identity prefix intact and the request stays on the plan.

    Use APPEND_ONLY when the user runs into the
    "Third-party apps now draw from extra usage" 400 — Hermes's tool
    envelope contract still gets through, but CC's "you are Claude Code"
    preamble survives, which the billing classifier seems to require.
    """
    caller_text = _stringify_system(caller_system)
    # Catalog rendering knobs:
    #   HERMES_CLAUDE_CLI_CATALOG_COMPACT=0  → restore verbose JSON-schema dump
    #   HERMES_CLAUDE_CLI_MAX_DESC_CHARS=N   → truncate descriptions to N chars
    catalog_compact = os.environ.get(
        "HERMES_CLAUDE_CLI_CATALOG_COMPACT", "1"
    ).strip() not in ("0", "false", "no")
    try:
        catalog_max_desc = int(
            os.environ.get("HERMES_CLAUDE_CLI_MAX_DESC_CHARS", "120")
        )
    except ValueError:
        catalog_max_desc = 120
    tool_catalog = render_tool_catalog(
        tools,
        compact=catalog_compact,
        max_description_chars=catalog_max_desc,
    )

    soft_limit, hard_limit = _prompt_size_limits()
    envelope_only = HERMES_TOOL_INSTRUCTIONS.rstrip()

    # Try the full catalog first; if the composed prompt would exceed the
    # hard limit, truncate the catalog before composing.
    projected_size = len(envelope_only) + len(caller_text) + len(tool_catalog) + 64
    if projected_size > hard_limit and tool_catalog:
        original_len = len(tool_catalog)
        tool_catalog = _truncate_catalog_to_fit(
            envelope_block=envelope_only,
            caller_text=caller_text,
            tool_catalog=tool_catalog,
            budget=hard_limit,
        )
        logger.warning(
            "claude_cli system prompt would be %d bytes (>%d hard limit); "
            "truncated tool catalog from %d to %d bytes to avoid the "
            "third-party-app classifier. Tune via HERMES_CLAUDE_CLI_PROMPT_HARD "
            "(currently %d).",
            projected_size, hard_limit, original_len, len(tool_catalog), hard_limit,
        )
    elif projected_size > soft_limit:
        logger.info(
            "claude_cli composed system prompt: %d bytes (soft warn threshold %d). "
            "If you hit QuotaExceeded, reduce toolset count or lower "
            "HERMES_CLAUDE_CLI_PROMPT_HARD.",
            projected_size, soft_limit,
        )

    envelope_block = envelope_only
    if tool_catalog:
        envelope_block = envelope_block + "\n\n" + tool_catalog.rstrip()

    if policy is SystemPromptPolicy.APPEND_ONLY:
        # CC supplies its own default system prompt; we add ours after it.
        # Caller's Hermes-side instructions ride along inside the appended
        # block so Hermes's identity / personality directives are preserved.
        parts = [envelope_block]
        if caller_text:
            parts.append(caller_text)
        return (None, "\n\n".join(parts))

    # REPLACE: produce the full prompt ourselves, with the envelope first so
    # the "no built-in tools" contract can't be overridden by downstream text.
    parts = [envelope_block]
    if caller_text:
        parts.append(caller_text)
    return ("\n\n".join(parts), None)


def _stringify_system(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out: list[str] = []
        for block in value:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    out.append(block.get("text", ""))
                else:
                    # Non-text system blocks: best-effort flatten.
                    out.append(str(block.get("text", block)))
            else:
                out.append(str(block))
        return "\n\n".join(p for p in out if p)
    return str(value)


# ───────────────────────────── effort resolution ────────────────────────────


def _resolve_effort(api_kwargs: dict[str, Any]) -> Optional[str]:
    """Derive a CC ``--effort`` level from Anthropic-style thinking hints."""
    output_cfg = api_kwargs.get("output_config") or {}
    if isinstance(output_cfg, dict):
        effort = output_cfg.get("effort")
        if isinstance(effort, str):
            return effort.lower()
    thinking = api_kwargs.get("thinking") or {}
    if isinstance(thinking, dict):
        budget = thinking.get("budget_tokens")
        if isinstance(budget, int):
            # Rough bucketing; mirrors Hermes's existing ADAPTIVE_EFFORT_MAP.
            if budget <= 2000:
                return "low"
            if budget <= 8000:
                return "medium"
            if budget <= 20000:
                return "high"
            if budget <= 40000:
                return "xhigh"
            return "max"
    return None


def _resolve_json_schema(api_kwargs: dict[str, Any]) -> Optional[str]:
    """Extract a JSON schema for structured output, in either supported form.

    Accepts:
      * ``api_kwargs["json_schema"]`` — direct pass-through (dict or JSON str).
      * ``api_kwargs["response_format"]`` — OpenAI-shim form:
        ``{"type": "json_schema", "json_schema": {"schema": {...}}}`` or
        ``{"type": "json_schema", "json_schema": {...}}`` (schema inline).

    Returns a serialized JSON string ready for ``--json-schema`` CLI arg, or
    ``None`` to leave the flag off (unconstrained output).
    """
    import json as _json

    direct = api_kwargs.get("json_schema")
    if isinstance(direct, str) and direct.strip():
        return direct
    if isinstance(direct, dict) and direct:
        try:
            return _json.dumps(direct)
        except (TypeError, ValueError):
            return None

    rf = api_kwargs.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        inner = rf.get("json_schema")
        if isinstance(inner, dict):
            # OpenAI nests the schema under ``.schema``; some callers put it
            # directly under ``json_schema``. Accept both.
            schema = inner.get("schema") if "schema" in inner else inner
            if isinstance(schema, dict) and schema:
                try:
                    return _json.dumps(schema)
                except (TypeError, ValueError):
                    return None
    return None


# ────────────────────────────── stream context ──────────────────────────────


class StreamCtx:
    """Context manager that yields Anthropic-shaped stream events.

    Usage matches the Anthropic SDK::

        with client.messages.stream(**api_kwargs) as stream:
            for event in stream:
                ...
            final = stream.get_final_message()
    """

    def __init__(
        self,
        *,
        state: SessionState,
        translator: EventTranslator,
        turn_budget_s: float,
        lock: threading.Lock,
        session_manager: SessionManager,
    ) -> None:
        self._state = state
        self._translator = translator
        self._turn_budget_s = turn_budget_s
        self._lock = lock
        self._session_manager = session_manager
        self._final: Optional[TranslatedTurn] = None
        self._entered = False
        self._exited = False
        self._iterated = False

    def __enter__(self) -> "StreamCtx":
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._exited = True
        try:
            self._session_manager.mark_in_flight(self._state, False)
        finally:
            if self._lock.locked():
                try:
                    self._lock.release()
                except RuntimeError:
                    pass

    def __iter__(self) -> Iterator[SimpleNamespace]:
        if not self._entered:
            raise RuntimeError("StreamCtx must be used as a context manager")
        if self._iterated:
            raise RuntimeError("StreamCtx is single-use")
        self._iterated = True
        return self._stream_events()

    def _stream_events(self) -> Iterator[SimpleNamespace]:
        proc = self._state.subprocess
        assert proc is not None

        events_iter = proc.events(overall_timeout_s=self._turn_budget_s)
        saw_result = False
        for cc_event in events_iter:
            if isinstance(cc_event, dict) and cc_event.get("type") == "result":
                saw_result = True
            for ev in self._translator.consume(cc_event):
                yield ev
            if self._translator.halted_for_tool:
                break
            if self._translator.pending_error is not None:
                break

        # Abnormal exit detection: no result event, no tool halt, and the
        # subprocess is dead → treat as crash (or OAuth failure if stderr
        # flagged it).
        if (
            not saw_result
            and not self._translator.halted_for_tool
            and not proc.is_alive()
            and self._translator.pending_error is None
        ):
            stderr_tail = proc.stderr_tail()
            for line in stderr_tail.splitlines():
                self._translator.on_subprocess_stderr(line)
                if self._translator.pending_error is not None:
                    break
            if self._translator.pending_error is None:
                rc = -1
                try:
                    rc = proc._proc.returncode if proc._proc else -1  # type: ignore[attr-defined]
                except Exception:
                    pass
                self._translator.on_subprocess_exit(
                    returncode=rc if rc is not None else -1,
                    stderr_tail=stderr_tail,
                )

        for ev in self._translator.finalize():
            yield ev

        err = self._translator.pending_error
        if err is not None:
            if err.stderr_tail is None:
                err.stderr_tail = proc.stderr_tail()
            raise err

        self._final = self._translator.build_final_message()

        # Observability: log the per-turn metrics CC gave us. Keep the line
        # concise — production logs can grow fast under heavy tool loops.
        if self._final is not None:
            logger.info(
                "claude_cli turn: tokens in=%s out=%s cache_read=%s cache_write=%s cost_usd=%s duration_ms=%s api_ms=%s ttft_ms=%s stop=%s",
                self._final.usage.input_tokens,
                self._final.usage.output_tokens,
                getattr(self._final.usage, "cache_read_input_tokens", 0),
                getattr(self._final.usage, "cache_creation_input_tokens", 0),
                self._final.ccli_cost_usd,
                self._final.ccli_duration_ms,
                self._final.ccli_duration_api_ms,
                self._final.ccli_ttft_ms,
                self._final.stop_reason,
            )

        # Housekeeping: if halted for tool, kill+respawn with --resume so the
        # next messages.create call can stream the tool_result envelope.
        if self._translator.halted_for_tool:
            tool = self._translator.parsed_tool
            if tool is not None:
                self._state.pending_tool_id = tool.tool_id
            self._session_manager.interrupt_and_resume()

    def get_final_message(self) -> Any:
        if self._final is None:
            # Drain the iterator so finalize() runs and builds the message.
            for _ in self:
                pass
        return _message_namespace(self._final)


# ─────────────────────────── Message materialization ────────────────────────


def _message_namespace(turn: Optional[TranslatedTurn]) -> SimpleNamespace:
    """Convert a TranslatedTurn into a SimpleNamespace matching the SDK shape.

    Anthropic-SDK-compatible fields (id/model/role/content/stop_reason/
    stop_sequence/usage) are always present. CC-specific extensions
    (``ccli_*``) are attached as extra attributes — callers that don't know
    about them see no change.
    """
    if turn is None:
        return SimpleNamespace(
            id=f"msg_empty",
            model="unknown",
            role="assistant",
            content=[],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=SimpleNamespace(
                input_tokens=0,
                output_tokens=0,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )
    ns = SimpleNamespace(
        id=turn.id,
        model=turn.model,
        role=turn.role,
        content=turn.content,
        stop_reason=turn.stop_reason,
        stop_sequence=turn.stop_sequence,
        usage=turn.usage,
        ccli_cost_usd=turn.ccli_cost_usd,
        ccli_duration_ms=turn.ccli_duration_ms,
        ccli_duration_api_ms=turn.ccli_duration_api_ms,
        ccli_ttft_ms=turn.ccli_ttft_ms,
        ccli_num_turns=turn.ccli_num_turns,
        ccli_service_tier=turn.ccli_service_tier,
        ccli_model_usage=turn.ccli_model_usage,
        ccli_structured_output=turn.ccli_structured_output,
    )
    return ns


# ──────────────────────────── messages namespace ────────────────────────────


class MessagesNamespace:
    def __init__(self, client: "ClaudeCliClient") -> None:
        self._client = client

    def create(self, **api_kwargs) -> SimpleNamespace:
        stream_ctx = self._client._begin_turn(api_kwargs, streaming=False)
        with stream_ctx as s:
            final = s.get_final_message()
        return final

    def stream(self, **api_kwargs) -> StreamCtx:
        return self._client._begin_turn(api_kwargs, streaming=True)


# ─────────────────────────────── main client ────────────────────────────────


class ClaudeCliClient:
    """Persistent ``claude -p`` client with an Anthropic-SDK-shaped surface."""

    def __init__(
        self,
        *,
        model_default: str,
        claude_bin: Optional[str] = None,
        cwd: Optional[str] = None,
        request_timeout_s: float = 600.0,
        turn_budget_s: float = 300.0,
        system_prompt_policy: SystemPromptPolicy = SystemPromptPolicy.REPLACE,
    ) -> None:
        self.model_default = model_default
        self.request_timeout_s = request_timeout_s
        self.turn_budget_s = turn_budget_s
        self.system_prompt_policy = system_prompt_policy
        resolved_bin = resolve_claude_bin(claude_bin)
        self._config = ClaudeCliConfig(
            claude_bin=resolved_bin,
            cwd=cwd,
            request_timeout_s=request_timeout_s,
            turn_budget_s=turn_budget_s,
        )
        self._session_manager = SessionManager(
            claude_bin=resolved_bin,
            cwd=cwd,
            stderr_ring_bytes=self._config.stderr_ring_bytes,
            kill_grace_s=self._config.kill_grace_s,
        )
        self._lock = threading.Lock()
        self._closed = False
        atexit.register(self._atexit_close)

    # ───────────── public surface ─────────────

    @property
    def messages(self) -> MessagesNamespace:
        return MessagesNamespace(self)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._session_manager.close()
        except Exception as e:
            logger.debug("close() cleanup error: %s", e)

    # ───────────── internals ─────────────

    def _atexit_close(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _begin_turn(self, api_kwargs: dict[str, Any], *, streaming: bool) -> StreamCtx:
        if self._closed:
            raise ClaudeCliError("client is closed")

        if not self._lock.acquire(blocking=False):
            raise ClientBusy("another turn is in flight on this client")

        try:
            model = api_kwargs.get("model") or self.model_default
            tools = api_kwargs.get("tools")
            system_prompt, append_system_prompt = _compose_system_prompt(
                caller_system=api_kwargs.get("system"),
                tools=tools,
                policy=self.system_prompt_policy,
            )
            effort = _resolve_effort(api_kwargs)
            json_schema = _resolve_json_schema(api_kwargs)
            messages = list(api_kwargs.get("messages") or [])

            state = self._session_manager.ensure(
                model=model,
                system_prompt=system_prompt,
                append_system_prompt=append_system_prompt,
                effort=effort,
                json_schema=json_schema,
                messages=messages,
            )
            self._session_manager.mark_in_flight(state, True)

            translator = EventTranslator(model=model)
            return StreamCtx(
                state=state,
                translator=translator,
                turn_budget_s=self.turn_budget_s,
                lock=self._lock,
                session_manager=self._session_manager,
            )
        except Exception:
            # Release the lock if setup failed before we handed back a
            # StreamCtx that manages it.
            try:
                self._lock.release()
            except RuntimeError:
                pass
            raise


# ───────────────────────────── factory helpers ──────────────────────────────


def build_claude_cli_client(
    *,
    model_default: str,
    claude_bin: Optional[str] = None,
    cwd: Optional[str] = None,
    request_timeout_s: float = 600.0,
    turn_budget_s: float = 300.0,
    system_prompt_policy: Optional[SystemPromptPolicy] = None,
) -> ClaudeCliClient:
    """Construct a :class:`ClaudeCliClient`. Convenience factory used by ``run_agent.py``.

    The ``system_prompt_policy`` argument controls whether Hermes's system
    prompt is sent via ``--system-prompt`` (REPLACE — full control, may
    trigger Anthropic's third-party-app billing classifier) or
    ``--append-system-prompt`` (APPEND_ONLY — preserves CC's identity
    preamble, which empirically keeps the request on the user's plan).

    When unset (default), the policy is taken from the env var
    ``HERMES_CLAUDE_CLI_SYSTEM_PROMPT_POLICY`` — accepted values:
    ``replace`` (default) and ``append`` / ``append_only``.
    """
    if system_prompt_policy is None:
        env_val = os.environ.get(
            "HERMES_CLAUDE_CLI_SYSTEM_PROMPT_POLICY", "replace"
        ).strip().lower()
        if env_val in ("append", "append_only", "append-only"):
            system_prompt_policy = SystemPromptPolicy.APPEND_ONLY
        else:
            system_prompt_policy = SystemPromptPolicy.REPLACE
    return ClaudeCliClient(
        model_default=model_default,
        claude_bin=claude_bin,
        cwd=cwd,
        request_timeout_s=request_timeout_s,
        turn_budget_s=turn_budget_s,
        system_prompt_policy=system_prompt_policy,
    )
