"""Unit tests for agent.claude_cli.tool_protocol."""

from __future__ import annotations

import pytest

from agent.claude_cli.errors import ToolParseError
from agent.claude_cli.tool_protocol import (
    HERMES_TOOL_INSTRUCTIONS,
    HaltScanner,
    encode_tool_result,
    parse_tool_envelope,
    render_tool_catalog,
)


# ─────────────────────────── parse_tool_envelope ─────────────────────────────


def test_parse_basic_envelope():
    text = (
        '<hermes:tool name="bash" id="t_01">'
        '<arg name="command">ls -la</arg>'
        "</hermes:tool>"
    )
    call = parse_tool_envelope(text)
    assert call.name == "bash"
    assert call.tool_id == "t_01"
    assert call.args == {"command": "ls -la"}


def test_parse_multiple_args_with_whitespace():
    text = """
        <hermes:tool name="search" id="t_02">
            <arg name="query">python</arg>
            <arg name="limit">5</arg>
        </hermes:tool>
    """
    call = parse_tool_envelope(text)
    assert call.name == "search"
    assert call.args == {"query": "python", "limit": "5"}


def test_parse_entity_decoded_values():
    text = (
        '<hermes:tool name="echo" id="t_03">'
        '<arg name="text">a &amp; b &lt;c&gt; &quot;d&quot;</arg>'
        "</hermes:tool>"
    )
    call = parse_tool_envelope(text)
    assert call.args == {"text": 'a & b <c> "d"'}


def test_parse_missing_opener_raises():
    with pytest.raises(ToolParseError):
        parse_tool_envelope("no envelope here")


def test_parse_missing_close_raises():
    with pytest.raises(ToolParseError):
        parse_tool_envelope('<hermes:tool name="x" id="y"><arg name="a">b</arg>')


def test_parse_missing_name_raises():
    with pytest.raises(ToolParseError):
        parse_tool_envelope('<hermes:tool id="t_01"></hermes:tool>')


def test_parse_missing_id_raises():
    with pytest.raises(ToolParseError):
        parse_tool_envelope('<hermes:tool name="bash"></hermes:tool>')


def test_parse_discards_content_after_close():
    """Defends against the hallucinated-tool_result pattern seen in the spike."""
    text = (
        '<hermes:tool name="bash" id="t_04"><arg name="command">pwd</arg></hermes:tool>'
        '<hermes:tool_result id="t_04">/tmp</hermes:tool_result>'
        "then the model babbled on"
    )
    call = parse_tool_envelope(text)
    assert call.tool_id == "t_04"
    assert call.args == {"command": "pwd"}
    # The raw field captures only the first envelope.
    assert call.raw.endswith("</hermes:tool>")
    assert "tool_result" not in call.raw


def test_parse_picks_first_when_two_openers():
    text = (
        '<hermes:tool name="bash" id="t_a"><arg name="command">pwd</arg></hermes:tool>'
        '<hermes:tool name="bash" id="t_b"><arg name="command">ls</arg></hermes:tool>'
    )
    call = parse_tool_envelope(text)
    assert call.tool_id == "t_a"


# ────────────────────────────── HaltScanner ─────────────────────────────────


def test_halt_detects_close_tag_in_single_chunk():
    scanner = HaltScanner()
    result = scanner.feed(
        'Let me check. <hermes:tool name="bash" id="t_01">'
        '<arg name="command">pwd</arg></hermes:tool>'
    )
    assert result is not None
    assert scanner.halted
    assert result.endswith("</hermes:tool>")


def test_halt_detects_close_tag_across_chunks():
    scanner = HaltScanner()
    # Feed in tiny chunks so the close tag spans a boundary.
    parts = [
        '<hermes:tool name="bash" id="t_01"><arg name="command">pwd</arg></her',
        "mes:tool>",
    ]
    found = None
    for p in parts:
        r = scanner.feed(p)
        if r is not None:
            found = r
    assert found is not None
    assert scanner.halted


def test_halt_does_not_fire_without_close():
    scanner = HaltScanner()
    r = scanner.feed("no envelope here yet")
    assert r is None
    assert not scanner.halted


def test_halt_fires_only_once():
    scanner = HaltScanner()
    first = scanner.feed(
        '<hermes:tool name="bash" id="t_01"><arg name="command">pwd</arg></hermes:tool>'
    )
    assert first is not None
    # Further feeds return None (already halted).
    second = scanner.feed(" and then more text")
    assert second is None


def test_has_open_without_close_budget():
    scanner = HaltScanner()
    scanner.feed('<hermes:tool name="bash" id="t_01"><arg name="command">')
    # Under the budget — not yet flagged.
    assert not scanner.has_open_without_close(extra_chars_to_allow=1024)
    # Pad out to exceed the budget.
    scanner.feed("x" * 2000)
    assert scanner.has_open_without_close(extra_chars_to_allow=1024)


# ──────────────────────────── encode_tool_result ────────────────────────────


def test_encode_basic():
    out = encode_tool_result("t_01", "hello")
    assert out == '<hermes:tool_result id="t_01">hello</hermes:tool_result>'


def test_encode_xml_escapes_body():
    out = encode_tool_result("t_01", "5 < 10 & 20 > 1")
    assert "&lt;" in out and "&amp;" in out and "&gt;" in out


def test_encode_is_error():
    out = encode_tool_result("t_01", "boom", is_error=True)
    assert 'is_error="true"' in out
    assert "<error>boom</error>" in out


# ─────────────────────────── render_tool_catalog ────────────────────────────


def test_catalog_empty_is_empty():
    assert render_tool_catalog(None) == ""
    assert render_tool_catalog([]) == ""


def test_catalog_renders_tool():
    out = render_tool_catalog(
        [
            {
                "name": "bash",
                "description": "Run a shell command",
                "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
            }
        ]
    )
    assert "bash" in out
    assert "Run a shell command" in out
    # Compact mode: argument name appears in the signature (no full schema dump).
    assert "command" in out


def test_catalog_compact_omits_full_schema():
    """Default rendering must not include verbose JSON schema dumps."""
    out = render_tool_catalog([
        {"name": "x", "description": "d",
         "input_schema": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}, "required": ["a"]}},
    ])
    # No JSON-property quoting
    assert '"properties"' not in out
    assert '"type"' not in out
    # Default mode drops type annotations to minimize schema-shape signal —
    # just shows arg names with ? for optional.
    assert "x(a, b?)" in out


def test_catalog_verbose_mode_still_includes_schema():
    out = render_tool_catalog(
        [{"name": "x", "description": "d", "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}}}],
        compact=False,
    )
    assert '"properties"' in out


def test_catalog_truncates_long_descriptions():
    long_desc = "lorem ipsum dolor sit amet " * 50  # ~1300 chars
    out = render_tool_catalog(
        [{"name": "x", "description": long_desc, "input_schema": {"type": "object", "properties": {}}}],
        max_description_chars=80,
    )
    assert "…" in out
    # Line for the tool stays bounded
    line = [l for l in out.splitlines() if l.startswith("- x")][0]
    assert len(line) < 200


def test_catalog_compact_size_vs_verbose():
    """Compact rendering should be dramatically smaller for realistic tools."""
    tools = [
        {
            "name": f"tool_{i}",
            "description": f"Does thing number {i}.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "lines": {"type": "integer"},
                    "mode": {"type": "string"},
                },
                "required": ["path"],
            },
        }
        for i in range(20)
    ]
    compact = render_tool_catalog(tools, compact=True)
    verbose = render_tool_catalog(tools, compact=False)
    assert len(compact) < len(verbose) // 4, (
        f"compact {len(compact)} bytes vs verbose {len(verbose)} bytes "
        "— compact should be at least 4× smaller"
    )


def test_instructions_mention_envelope_format():
    assert "<hermes:tool" in HERMES_TOOL_INSTRUCTIONS
    assert "</hermes:tool>" in HERMES_TOOL_INSTRUCTIONS
    assert "<hermes:tool_result" in HERMES_TOOL_INSTRUCTIONS
