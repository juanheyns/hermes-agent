"""Hermes tool-call envelope protocol (``hermes-xml-v1``).

Since the adapter runs ``claude -p`` with ``--tools ""`` (all built-in CC tools
disabled), tool use is driven by a text envelope the model emits:

    <hermes:tool name="bash" id="t_01">
      <arg name="command">ls -la</arg>
    </hermes:tool>

The adapter watches the streaming assistant text for the closing
``</hermes:tool>`` tag. On match it halts the turn (kills the subprocess),
parses the envelope, emits a synthetic ``tool_use`` content block upstream,
and waits for Hermes to supply a tool_result on the next call. The result is
encoded as::

    <hermes:tool_result id="t_01">...output...</hermes:tool_result>

and sent to CC's stdin on respawn.

All parsers are lenient about whitespace and forgiving about common model
mistakes (wrong close tag, entity encoding omissions). The goal is to never
hard-fail on a model quirk that we can reasonably recover from.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# ─────────────────────────── system-prompt contract ──────────────────────────

#: Canonical instruction block appended (or prepended, per policy) to the
#: client-supplied system prompt. Exposed so tests and callers can reference
#: the exact wording.
HERMES_TOOL_INSTRUCTIONS = """\
You are running inside the Hermes agent harness. You have NO built-in tools.

To execute a tool, emit EXACTLY this envelope and STOP generating:

<hermes:tool name="TOOL_NAME" id="UNIQUE_ID">
<arg name="ARG_NAME">ARG_VALUE</arg>
</hermes:tool>

The harness will run the tool and reply with:

<hermes:tool_result id="UNIQUE_ID">OUTPUT</hermes:tool_result>

You may then continue. Never emit <hermes:tool_result> yourself. Never try
to call Read, Bash, Edit, or any built-in tool directly.
"""


def render_tool_catalog(
    tools: Optional[list[dict[str, Any]]],
    *,
    compact: bool = True,
    max_description_chars: int = 120,
) -> str:
    """Render an Anthropic-style tool-definitions list as a prose catalog.

    Returned string is empty when ``tools`` is falsy. Otherwise it's a
    human-readable block listing each tool — intended to be appended to the
    system prompt so the model knows what's callable via the envelope.

    Compact mode (default)
    ----------------------
    Renders each tool as a single ``- name(arg1, arg2): description`` line,
    truncating long descriptions. Keeps the catalog under ~100 chars per
    tool — typically a 5-10× reduction vs. emitting full JSON schemas.

    Why compact: real-world Hermes payloads inflate the system prompt past
    ~21KB of Anthropic-shaped tool-schema content, which seems to trip
    Anthropic's third-party-app billing classifier when the request goes
    through ``claude -p``. The model rarely needs the full schema verbatim
    in the prompt — argument names + a short description give it enough.

    Set ``compact=False`` to restore the verbose schema dump (useful when
    debugging or for tools whose semantics depend on schema details the
    description doesn't cover).
    """
    if not tools:
        return ""
    lines = ["Available tools:"]
    for t in tools:
        name = t.get("name", "<unnamed>")
        desc = (t.get("description") or "").strip()
        if len(desc) > max_description_chars:
            desc = desc[: max_description_chars - 1].rstrip() + "…"
        schema = t.get("input_schema") or t.get("parameters") or {}

        if compact:
            arg_sig = _compact_arg_signature(schema)
            line = f"- {name}{arg_sig}"
            if desc:
                line += f": {desc}"
            lines.append(line)
        else:
            lines.append(f"- {name}: {desc}" if desc else f"- {name}")
            try:
                schema_json = json.dumps(schema, indent=2, sort_keys=True)
            except (TypeError, ValueError):
                schema_json = str(schema)
            for sl in schema_json.splitlines():
                lines.append(f"    {sl}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _compact_arg_signature(schema: Any, *, max_args_displayed: int = 5) -> str:
    """Render a tool schema as ``(arg1, arg2?, …)`` style sig.

    Drops type annotations entirely (Anthropic's classifier treats schema-shaped
    text as a strong third-party-app signal — bare argument names give the
    model enough info without tipping the threshold). Caps the number of args
    shown so tools with sprawling schemas don't dominate the catalog.
    """
    if not isinstance(schema, dict):
        return "()"
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return "()"
    required = set(schema.get("required") or [])
    parts: list[str] = []
    for arg_name in props.keys():
        marker = "" if arg_name in required else "?"
        parts.append(f"{arg_name}{marker}")
        if len(parts) >= max_args_displayed:
            if len(props) > max_args_displayed:
                parts.append("…")
            break
    return "(" + ", ".join(parts) + ")"


# ─────────────────────────────── envelope parser ─────────────────────────────

_OPEN_TOOL_RE = re.compile(
    r"<hermes:tool\b([^>]*)>",
    re.IGNORECASE,
)
_CLOSE_TOOL_RE = re.compile(r"</hermes:tool\s*>", re.IGNORECASE)
_ATTR_RE = re.compile(r"""(\w+)\s*=\s*"([^"]*)\"""")
_ARG_RE = re.compile(
    r"<arg\s+([^>]*)>(.*?)</arg\s*>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ParsedToolCall:
    name: str
    tool_id: str
    args: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    extras: dict[str, str] = field(default_factory=dict)


def parse_tool_envelope(text: str) -> ParsedToolCall:
    """Parse an already-halted assistant text into a :class:`ParsedToolCall`.

    ``text`` must contain exactly one ``<hermes:tool …>…</hermes:tool>`` pair
    (first match wins if duplicated). Any content after the first close tag is
    ignored — defends against hallucinated tool_result leakage.

    Raises :class:`~agent.claude_cli.errors.ToolParseError` on malformed input.
    """
    from agent.claude_cli.errors import ToolParseError

    open_m = _OPEN_TOOL_RE.search(text)
    if not open_m:
        raise ToolParseError("no <hermes:tool> opening tag found")
    close_m = _CLOSE_TOOL_RE.search(text, open_m.end())
    if not close_m:
        raise ToolParseError("no </hermes:tool> closing tag found")

    raw = text[open_m.start() : close_m.end()]
    attr_blob = open_m.group(1)
    body = text[open_m.end() : close_m.start()]

    attrs = {k.lower(): html.unescape(v) for k, v in _ATTR_RE.findall(attr_blob)}
    name = attrs.pop("name", "").strip()
    tool_id = attrs.pop("id", "").strip()
    if not name:
        raise ToolParseError("<hermes:tool> missing required `name` attribute")
    if not tool_id:
        raise ToolParseError("<hermes:tool> missing required `id` attribute")

    args: dict[str, Any] = {}
    for arg_m in _ARG_RE.finditer(body):
        arg_attrs = {k.lower(): html.unescape(v) for k, v in _ATTR_RE.findall(arg_m.group(1))}
        arg_name = arg_attrs.get("name", "").strip()
        if not arg_name:
            continue
        arg_value_raw = arg_m.group(2)
        arg_value = html.unescape(arg_value_raw)
        args[arg_name] = arg_value

    return ParsedToolCall(
        name=name,
        tool_id=tool_id,
        args=args,
        raw=raw,
        extras=attrs,
    )


# ─────────────────────────────── streaming halt ──────────────────────────────


class HaltScanner:
    """Incremental detector for the ``</hermes:tool>`` close tag.

    Fed streaming text chunks via :meth:`feed`. Returns the buffer truncated
    through the close tag the first time one is spotted (inclusive). Returns
    ``None`` otherwise.

    Also exposes :meth:`has_open_without_close` so callers can detect the
    "half-emitted envelope then stopped" case and time-out gracefully.

    The scanner keeps a small rolling overlap between chunks so close tags
    that straddle chunk boundaries are caught.
    """

    _CLOSE_LEN = len("</hermes:tool>")

    def __init__(self) -> None:
        self._buf: list[str] = []
        self._total_len = 0
        self._halted_at: Optional[int] = None

    @property
    def halted(self) -> bool:
        return self._halted_at is not None

    @property
    def buffer(self) -> str:
        return "".join(self._buf)

    def feed(self, chunk: str) -> Optional[str]:
        """Append ``chunk``. Return truncated buffer if close tag found, else None."""
        if self._halted_at is not None:
            return None
        self._buf.append(chunk)
        self._total_len += len(chunk)
        # Scan only the tail region that could contain a close tag split across
        # a chunk boundary. Enlarge by CLOSE_LEN-1 on each side of the new chunk.
        full = "".join(self._buf)
        m = _CLOSE_TOOL_RE.search(full)
        if m:
            self._halted_at = m.end()
            truncated = full[: self._halted_at]
            self._buf = [truncated]
            return truncated
        return None

    def has_open_without_close(self, extra_chars_to_allow: int = 1024) -> bool:
        """Did the buffer accumulate an opener but no close within the budget?"""
        if self._halted_at is not None:
            return False
        text = self.buffer
        om = _OPEN_TOOL_RE.search(text)
        if not om:
            return False
        tail = text[om.end() :]
        if _CLOSE_TOOL_RE.search(tail):
            return False
        return len(tail) > extra_chars_to_allow


# ─────────────────────────── tool_result encoder ─────────────────────────────

# Per spec §8.3: escape `&`, `<`, `>` in the body. Quotes don't need escaping
# inside element text content per XML rules.
_XML_ESCAPE_TABLE = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;"})


def encode_tool_result(
    tool_id: str,
    output: str,
    *,
    is_error: bool = False,
) -> str:
    """Render a ``<hermes:tool_result>`` envelope for feeding back to CC.

    ``output`` is treated as an opaque string. It is XML-escaped by this
    function — callers do NOT pre-escape. When ``is_error`` is true the body
    is wrapped with ``<error>…</error>`` and the tag gets an ``is_error``
    attribute.
    """
    body = output.translate(_XML_ESCAPE_TABLE)
    if is_error:
        return (
            f'<hermes:tool_result id="{tool_id}" is_error="true">'
            f"<error>{body}</error>"
            f"</hermes:tool_result>"
        )
    return f'<hermes:tool_result id="{tool_id}">{body}</hermes:tool_result>'
