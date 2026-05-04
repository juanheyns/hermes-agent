"""Translate ``claude -p`` stream-json events into Anthropic-SDK-shaped events.

The rest of ``run_agent.py`` consumes Anthropic SDK ``messages.stream()`` events
directly — ``content_block_start``, ``content_block_delta`` with ``text_delta``
or ``thinking_delta``, ``content_block_stop``, ``message_delta`` carrying a
``stop_reason``, and ``message_stop``. The translator here re-shapes CC's
stream-json schema onto that contract and in parallel builds the final
``Message`` object that ``stream.get_final_message()`` will return.

Design highlights
-----------------

* The translator is a pure generator plus a small state machine. It has no
  subprocess ownership — caller feeds it parsed JSON events.
* Tool use is detected via :class:`~agent.claude_cli.tool_protocol.HaltScanner`
  scanning the accumulated assistant text. On ``</hermes:tool>`` the translator
  emits a synthetic ``tool_use`` content block derived from the parsed
  envelope and stops consuming further CC events for this turn.
* A small look-ahead window (``_SAFE_FORWARD_LOOKBACK`` chars) is held back
  from each ``text_delta`` so an opener tag split across CC deltas never leaks
  to the caller's streaming callback before we can halt.
* ``duplicate-text`` pitfall (confirmed in the spike) — the CC ``assistant``
  final-message event and the streaming deltas both contain the full text. We
  emit ``text_delta`` events only from the streaming path; the ``assistant``
  event is used solely as a sanity check on the buffered text.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterator, Optional

from agent.claude_cli.errors import (
    ApiError,
    ClaudeCliError,
    QuotaExceeded,
    SubprocessCrashed,
    ToolParseError,
)
from agent.claude_cli.tool_protocol import (
    HaltScanner,
    ParsedToolCall,
    parse_tool_envelope,
)

_SAFE_FORWARD_LOOKBACK = 24  # >= len("<hermes:tool")
_OPENER_HINT = "<hermes:tool"

# Patterns in CC stderr that indicate OAuth trouble.
_OAUTH_FAIL_MARKERS = (
    "oauth token expired",
    "please run claude auth",
    "session authentication failed",
    "401",
)

# Substrings in the API error body that indicate plan / billing exhaustion
# (as opposed to a transient rate limit). Anthropic's wording shifts over
# time; keep this list tolerant.
_QUOTA_MARKERS = (
    "extra usage",
    "plan limits",
    "workspace admin to add more",
    "quota exceeded",
    "credit balance is too low",
    "billing",
)


def _classify_api_error(body: str, status_code: Any) -> ApiError:
    """Pick the most specific ApiError subclass for a CC-surfaced API failure.

    For the "third-party apps now draw from extra usage" variant, append a
    one-line hint to the message so users see the concrete workaround:
    Anthropic's classifier triggers on system prompts heavy with tool-schema
    content. Lowering ``HERMES_CLAUDE_CLI_PROMPT_HARD`` forces the adapter
    to truncate the catalog more aggressively before the request leaves.
    """
    body_lower = (body or "").lower()
    sc = status_code if isinstance(status_code, int) else None
    if any(m in body_lower for m in _QUOTA_MARKERS):
        # Targeted hint for the classifier-triggered case.
        if "extra usage" in body_lower or "plan limits" in body_lower:
            hint = (
                "\n[claude_cli hint] If you have a Claude plan but still see this error, "
                "Anthropic's third-party-app classifier may be flagging the request. "
                "Try: HERMES_CLAUDE_CLI_PROMPT_HARD=12000 to force a smaller tool "
                "catalog. If that doesn't help, top up Extra Usage in your "
                "workspace billing or switch to provider=anthropic with an API key."
            )
            return QuotaExceeded(body + hint, status_code=sc)
        return QuotaExceeded(body, status_code=sc)
    return ApiError(body, status_code=sc)


def _ns(**fields) -> SimpleNamespace:
    return SimpleNamespace(**fields)


@dataclass
class TranslatedTurn:
    """The final Message shape returned by ``stream.get_final_message()``.

    Carries Anthropic-SDK-compatible fields plus a small set of CC-specific
    extensions (``ccli_*``) for observability. Hermes's existing
    ``normalize_usage`` reads the SDK-shape fields, so cost tracking works
    end-to-end with no call-site changes.
    """

    id: str
    model: str
    role: str = "assistant"
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    content: list[SimpleNamespace] = field(default_factory=list)
    usage: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
    )
    # CC-specific extras. Accessed by observability/logging code via
    # getattr(response, 'ccli_*', default) so callers that don't know about
    # them see no change in behavior.
    ccli_cost_usd: Optional[float] = None
    ccli_duration_ms: Optional[int] = None
    ccli_duration_api_ms: Optional[int] = None
    ccli_ttft_ms: Optional[int] = None
    ccli_num_turns: Optional[int] = None
    ccli_service_tier: Optional[str] = None
    ccli_model_usage: Optional[dict[str, Any]] = None
    # Validated structured output when --json-schema was passed. Present only
    # on turns where the caller requested structured output AND the model
    # successfully produced a matching object.
    ccli_structured_output: Optional[Any] = None


class EventTranslator:
    """Turn a stream of CC events into Anthropic-shaped events + final message.

    Usage::

        translator = EventTranslator(model=model_id)
        for cc_event in cc_event_iter:
            for ev in translator.consume(cc_event):
                yield ev
            if translator.halted_for_tool:
                break
        for ev in translator.finalize():
            yield ev
        final_message = translator.build_final_message()
    """

    def __init__(self, *, model: str, message_id: Optional[str] = None) -> None:
        self.model = model
        self.message_id = message_id or f"msg_{uuid.uuid4().hex[:16]}"

        self._halt = HaltScanner()
        self._assistant_text_buf: list[str] = []
        self._pending_forward: str = ""  # unflushed trailing chars
        self._text_block_open = False
        self._thinking_block_open = False
        self._thinking_content_buf: list[str] = []
        self._block_index = -1

        self._halted_for_tool = False
        self._parsed_tool: Optional[ParsedToolCall] = None
        self._stop_reason: Optional[str] = None
        self._content_blocks: list[SimpleNamespace] = []
        self._pending_error: Optional[ClaudeCliError] = None

        # ── Metrics captured from CC events (see scripts/claude_cli_event_probe.py) ──
        # From stream_event.message_start: canonical Anthropic msg id + concrete
        # model slug (e.g. "claude-sonnet-4-6") to replace our placeholder.
        self._cc_message_id: Optional[str] = None
        self._cc_model: Optional[str] = None
        # Usage counters. Populated progressively from stream_event.message_start /
        # message_delta.usage, then overwritten by the authoritative 'result'
        # event's `usage` block on turn end.
        self._usage_input_tokens: int = 0
        self._usage_output_tokens: int = 0
        self._usage_cache_creation_input_tokens: int = 0
        self._usage_cache_read_input_tokens: int = 0
        self._usage_service_tier: Optional[str] = None
        # CC-specific metadata from the 'result' event.
        self._cc_cost_usd: Optional[float] = None
        self._cc_duration_ms: Optional[int] = None
        self._cc_duration_api_ms: Optional[int] = None
        self._cc_ttft_ms: Optional[int] = None
        self._cc_num_turns: Optional[int] = None
        self._cc_model_usage: Optional[dict[str, Any]] = None
        # Populated when the caller requested structured output via --json-schema
        # and CC's synthetic StructuredOutput tool returned validated data.
        self._cc_structured_output: Optional[Any] = None

    # ────────────────────────── properties ──────────────────────────

    @property
    def halted_for_tool(self) -> bool:
        return self._halted_for_tool

    @property
    def parsed_tool(self) -> Optional[ParsedToolCall]:
        return self._parsed_tool

    @property
    def pending_error(self) -> Optional[ClaudeCliError]:
        return self._pending_error

    # ────────────────────────── main entry ──────────────────────────

    def consume(self, cc_event: dict[str, Any]) -> Iterator[SimpleNamespace]:
        """Feed one parsed CC stream-json event. Yield Anthropic-shaped events."""
        etype = cc_event.get("type")

        if etype == "system":
            # Init preamble; ignore.
            return

        if etype == "stream_event":
            inner = cc_event.get("event") or {}
            inner_type = inner.get("type")
            if inner_type == "content_block_delta":
                delta = inner.get("delta") or {}
                dtype = delta.get("type")
                if dtype == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        yield from self._on_text(text)
                elif dtype == "thinking_delta":
                    th = delta.get("thinking", "")
                    if th:
                        yield from self._on_thinking(th)
            elif inner_type == "message_start":
                # Canonical Anthropic message_start — grab the real msg id,
                # concrete model slug, and initial usage counters.
                msg = inner.get("message") or {}
                mid = msg.get("id")
                if isinstance(mid, str) and mid:
                    self._cc_message_id = mid
                mmodel = msg.get("model")
                if isinstance(mmodel, str) and mmodel:
                    self._cc_model = mmodel
                self._absorb_usage(msg.get("usage"))
                # ttft_ms is emitted on the wrapping stream_event by newer CC
                # versions; keep whichever we see first.
                ttft = cc_event.get("ttft_ms")
                if isinstance(ttft, int) and self._cc_ttft_ms is None:
                    self._cc_ttft_ms = ttft
            elif inner_type == "message_delta":
                # message_delta carries running usage updates (output_tokens grows).
                self._absorb_usage(inner.get("usage"))
            # Other stream_event subtypes (content_block_start/stop from CC
            # itself) are ignored — we synthesize our own block framing.
            return

        if etype == "content_block_delta":
            # Some CC versions flatten the event (no stream_event wrapper).
            delta = cc_event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                text = delta.get("text", "")
                if text:
                    yield from self._on_text(text)
            elif dtype == "thinking_delta":
                th = delta.get("thinking", "")
                if th:
                    yield from self._on_thinking(th)
            return

        if etype == "assistant":
            # Final assistant message echo — do NOT forward text (doubling).
            # Use only to cross-check stop reason and pick up any final usage
            # numbers the 'result' event might not carry (rare).
            msg = cc_event.get("message") or {}
            if not self._halted_for_tool:
                cc_stop = msg.get("stop_reason")
                if cc_stop:
                    self._stop_reason = cc_stop
            if self._cc_message_id is None:
                mid = msg.get("id")
                if isinstance(mid, str) and mid:
                    self._cc_message_id = mid
            if self._cc_model is None:
                mmodel = msg.get("model")
                if isinstance(mmodel, str) and mmodel:
                    self._cc_model = mmodel
            self._absorb_usage(msg.get("usage"))
            return

        if etype == "result":
            subtype = cc_event.get("subtype") or "success"
            # The 'result' event carries the authoritative final usage counts
            # and CC-specific metadata. Overwrites anything streamed earlier.
            self._absorb_usage(cc_event.get("usage"))
            self._cc_duration_ms = cc_event.get("duration_ms")
            self._cc_duration_api_ms = cc_event.get("duration_api_ms")
            self._cc_num_turns = cc_event.get("num_turns")
            cost = cc_event.get("total_cost_usd")
            if isinstance(cost, (int, float)):
                self._cc_cost_usd = float(cost)
            mu = cc_event.get("modelUsage")
            if isinstance(mu, dict) and mu:
                self._cc_model_usage = dict(mu)
            # Structured output path: when the caller passes --json-schema,
            # CC injects a synthetic 'StructuredOutput' tool, the model fills
            # it with validated data, and CC surfaces the parsed result here.
            # The ``content`` blocks still contain the model's follow-up prose.
            so = cc_event.get("structured_output")
            if so is not None:
                self._cc_structured_output = so
            # ── Detect API errors surfaced by CC ────────────────────────
            # When the upstream Anthropic API rejects the request, CC still
            # reports subtype="success" — the failure is encoded in the
            # is_error / api_error_status fields and the ``result`` text
            # carries the API's error body.
            is_error = bool(cc_event.get("is_error"))
            status = cc_event.get("api_error_status")
            if (is_error or isinstance(status, int)) and not self._halted_for_tool:
                # Don't clobber a more specific error already set (e.g. an
                # OAuthExpired raised from stderr inspection).
                if self._pending_error is None:
                    err_body = cc_event.get("result") or ""
                    self._pending_error = _classify_api_error(err_body, status)
                return
            if subtype != "success" and not self._halted_for_tool:
                # CC flagged an abnormal turn (e.g. interrupted). Surface as a crash.
                self._pending_error = SubprocessCrashed(
                    f"claude result subtype={subtype!r}"
                )
            return

        # Unknown event type — ignore (forward-compat).

    def _absorb_usage(self, usage: Any) -> None:
        """Merge a CC usage dict into our running counters.

        CC emits usage at several points:
          * ``stream_event.message_start.message.usage`` — initial counts
          * ``stream_event.message_delta.usage`` — running output_tokens
          * ``assistant.message.usage`` — final per-message usage
          * ``result.usage`` — authoritative aggregate for the turn

        For token counts we use "largest seen wins" — later events in a turn
        are always >= earlier ones. For cache counters we do the same.
        This is robust to missing or partial usage payloads.
        """
        if not isinstance(usage, dict):
            return
        it = usage.get("input_tokens")
        if isinstance(it, (int, float)) and int(it) > self._usage_input_tokens:
            self._usage_input_tokens = int(it)
        ot = usage.get("output_tokens")
        if isinstance(ot, (int, float)) and int(ot) > self._usage_output_tokens:
            self._usage_output_tokens = int(ot)
        ccit = usage.get("cache_creation_input_tokens")
        if (
            isinstance(ccit, (int, float))
            and int(ccit) > self._usage_cache_creation_input_tokens
        ):
            self._usage_cache_creation_input_tokens = int(ccit)
        crit = usage.get("cache_read_input_tokens")
        if (
            isinstance(crit, (int, float))
            and int(crit) > self._usage_cache_read_input_tokens
        ):
            self._usage_cache_read_input_tokens = int(crit)
        st = usage.get("service_tier")
        if isinstance(st, str) and st:
            self._usage_service_tier = st

    def on_subprocess_stderr(self, line: str) -> None:
        """Examine a single stderr line for auth-failure markers."""
        low = line.lower()
        for marker in _OAUTH_FAIL_MARKERS:
            if marker in low:
                from agent.claude_cli.errors import OAuthExpired

                self._pending_error = OAuthExpired(line.strip())
                return

    def on_subprocess_exit(self, returncode: int, stderr_tail: str) -> None:
        """Called when the subprocess exits without a proper ``result`` event."""
        if self._pending_error is None and not self._halted_for_tool:
            self._pending_error = SubprocessCrashed(
                f"claude exited rc={returncode}",
                stderr_tail=stderr_tail,
            )

    # ────────────────────────── finalize ──────────────────────────

    def finalize(self) -> Iterator[SimpleNamespace]:
        """Emit any remaining events so the final Message is consistent.

        Called after all CC events have been consumed (or after a tool halt
        has occurred). Flushes held-back text, closes open blocks, emits the
        ``message_delta`` / ``message_stop`` pair.
        """
        if self._pending_error is not None:
            # Caller will raise; no message_stop.
            return

        if self._halted_for_tool:
            # Halt path already emitted the tool_use block; still need to
            # close any open text block and emit message_stop.
            yield from self._flush_pending_text_up_to_opener()
            yield from self._close_text_block_if_open()
            # Emit tool_use block framing.
            tool = self._parsed_tool
            assert tool is not None
            self._block_index += 1
            tool_block = _ns(
                type="tool_use",
                id=tool.tool_id,
                name=tool.name,
                input=dict(tool.args),
            )
            self._content_blocks.append(tool_block)
            yield _ns(
                type="content_block_start",
                index=self._block_index,
                content_block=tool_block,
            )
            yield _ns(
                type="content_block_stop",
                index=self._block_index,
            )
            self._stop_reason = "tool_use"
        else:
            # Normal completion: flush all held-back text and close blocks.
            if self._pending_forward:
                yield from self._forward_text(self._pending_forward)
                self._pending_forward = ""
            yield from self._close_text_block_if_open()
            yield from self._close_thinking_block_if_open()
            if self._stop_reason is None:
                self._stop_reason = "end_turn"

        yield _ns(
            type="message_delta",
            delta=_ns(stop_reason=self._stop_reason, stop_sequence=None),
            usage=_ns(output_tokens=self._usage_output_tokens),
        )
        yield _ns(type="message_stop")

    def build_final_message(self) -> TranslatedTurn:
        msg = TranslatedTurn(
            id=self._cc_message_id or self.message_id,
            model=self._cc_model or self.model,
            stop_reason=self._stop_reason,
            content=list(self._content_blocks),
            usage=SimpleNamespace(
                input_tokens=self._usage_input_tokens,
                output_tokens=self._usage_output_tokens,
                cache_creation_input_tokens=self._usage_cache_creation_input_tokens,
                cache_read_input_tokens=self._usage_cache_read_input_tokens,
                service_tier=self._usage_service_tier,
            ),
            ccli_cost_usd=self._cc_cost_usd,
            ccli_duration_ms=self._cc_duration_ms,
            ccli_duration_api_ms=self._cc_duration_api_ms,
            ccli_ttft_ms=self._cc_ttft_ms,
            ccli_num_turns=self._cc_num_turns,
            ccli_service_tier=self._usage_service_tier,
            ccli_model_usage=self._cc_model_usage,
            ccli_structured_output=self._cc_structured_output,
        )
        return msg

    # ───────────────────────── text handling ─────────────────────────

    def _on_text(self, text: str) -> Iterator[SimpleNamespace]:
        self._assistant_text_buf.append(text)
        # Feed the halt scanner *and* manage the forward buffer.
        maybe_truncated = self._halt.feed(text)
        if maybe_truncated is not None:
            # Halt triggered. Discard any pending forward that sits past the
            # opener — everything after the opener is part of the envelope.
            self._halted_for_tool = True
            try:
                self._parsed_tool = parse_tool_envelope(maybe_truncated)
            except ToolParseError as e:
                # Fallback: surface the raw buffer as text; do NOT raise — let
                # the caller see a plain turn_end so the model can self-correct.
                self._halted_for_tool = False
                self._stop_reason = "end_turn"
                # Treat everything up to the opener as text; drop the malformed
                # envelope characters from output.
                opener_idx = maybe_truncated.lower().find(_OPENER_HINT)
                safe_text = (
                    maybe_truncated[:opener_idx]
                    if opener_idx != -1
                    else maybe_truncated
                )
                # Replace pending_forward with only the safe portion.
                self._pending_forward = ""
                if safe_text:
                    yield from self._forward_text(safe_text[len(self._assembled_text_forwarded()) :])
                return
            # Flush any safe text up to the opener.
            yield from self._flush_pending_text_up_to_opener()
            return

        # No halt yet. Flush safe prefix to the caller, hold back the tail.
        self._pending_forward += text
        safe_len = max(0, len(self._pending_forward) - _SAFE_FORWARD_LOOKBACK)
        if safe_len > 0:
            to_forward = self._pending_forward[:safe_len]
            self._pending_forward = self._pending_forward[safe_len:]
            yield from self._forward_text(to_forward)

    def _assembled_text_forwarded(self) -> str:
        """Sum of text already emitted as text_delta events (for length math)."""
        # Caller doesn't have a direct counter; reconstruct from content buf.
        total = "".join(
            getattr(b, "text", "") for b in self._content_blocks if getattr(b, "type", "") == "text"
        )
        # Also include the currently-open text block's running text.
        total += self._current_text_block_text()
        return total

    def _current_text_block_text(self) -> str:
        return getattr(self, "_cur_text_running", "")

    def _forward_text(self, text: str) -> Iterator[SimpleNamespace]:
        if not text:
            return
        if not self._text_block_open:
            self._block_index += 1
            self._text_block_open = True
            self._cur_text_running = ""
            block = _ns(type="text", text="")
            yield _ns(
                type="content_block_start",
                index=self._block_index,
                content_block=block,
            )
            self._current_text_block_ref = block
        self._cur_text_running += text
        yield _ns(
            type="content_block_delta",
            index=self._block_index,
            delta=_ns(type="text_delta", text=text),
        )

    def _flush_pending_text_up_to_opener(self) -> Iterator[SimpleNamespace]:
        """Forward the portion of ``_pending_forward`` before any opener tag."""
        if not self._pending_forward:
            return
        m = re.search(re.escape(_OPENER_HINT), self._pending_forward, re.IGNORECASE)
        if m is None:
            safe = self._pending_forward
            self._pending_forward = ""
        else:
            safe = self._pending_forward[: m.start()]
            self._pending_forward = ""  # remainder is part of the envelope
        if safe:
            yield from self._forward_text(safe)

    def _close_text_block_if_open(self) -> Iterator[SimpleNamespace]:
        if not self._text_block_open:
            return
        # Commit the accumulated text into the final content block.
        final_text = getattr(self, "_cur_text_running", "")
        block = _ns(type="text", text=final_text)
        self._content_blocks.append(block)
        yield _ns(type="content_block_stop", index=self._block_index)
        self._text_block_open = False
        self._cur_text_running = ""

    # ──────────────────────── thinking handling ───────────────────────

    def _on_thinking(self, text: str) -> Iterator[SimpleNamespace]:
        if self._text_block_open:
            # Close text block first — thinking becomes its own block.
            yield from self._close_text_block_if_open()
        if not self._thinking_block_open:
            self._block_index += 1
            self._thinking_block_open = True
            self._thinking_content_buf = []
            block = _ns(type="thinking", thinking="")
            yield _ns(
                type="content_block_start",
                index=self._block_index,
                content_block=block,
            )
        self._thinking_content_buf.append(text)
        yield _ns(
            type="content_block_delta",
            index=self._block_index,
            delta=_ns(type="thinking_delta", thinking=text),
        )

    def _close_thinking_block_if_open(self) -> Iterator[SimpleNamespace]:
        if not self._thinking_block_open:
            return
        final = "".join(self._thinking_content_buf)
        block = _ns(type="thinking", thinking=final)
        self._content_blocks.append(block)
        yield _ns(type="content_block_stop", index=self._block_index)
        self._thinking_block_open = False
        self._thinking_content_buf = []
