"""Tests for vision (content-block passthrough) and JSON-schema structured output."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent.claude_cli.client import _resolve_json_schema
from agent.claude_cli.session import render_user_content
from agent.claude_cli.subprocess_ import LaunchSpec

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


def _read_trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ───────────────────────── vision: content-block passthrough ─────────────────────────

_FAKE_IMAGE_BLOCK = {
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/png",
        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
    },
}


def test_image_only_content_renders_as_content_list():
    out = render_user_content([_FAKE_IMAGE_BLOCK])
    assert isinstance(out, list)
    assert out[0] == _FAKE_IMAGE_BLOCK


def test_image_with_text_renders_as_content_list():
    content = [{"type": "text", "text": "What is in this image?"}, _FAKE_IMAGE_BLOCK]
    out = render_user_content(content)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0] == {"type": "text", "text": "What is in this image?"}
    assert out[1] == _FAKE_IMAGE_BLOCK


def test_document_block_also_passes_through():
    doc = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0x"}}
    out = render_user_content([{"type": "text", "text": "summarize:"}, doc])
    assert isinstance(out, list)
    assert out[1] == doc


def test_end_to_end_stdin_carries_image_block(fake_bin, monkeypatch, tmp_path):
    """The stdin payload CC receives must include the image content block verbatim."""
    trace_file = tmp_path / "vision_trace.jsonl"
    stdin_capture = tmp_path / "vision_stdin.jsonl"
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    monkeypatch.setenv("CCCLI_FAKE_TRACE", str(trace_file))
    monkeypatch.setenv("CCCLI_FAKE_STDIN_CAPTURE", str(stdin_capture))

    from agent.claude_cli import build_claude_cli_client

    client = build_claude_cli_client(
        model_default="sonnet", claude_bin=fake_bin, turn_budget_s=15
    )
    try:
        client.messages.create(
            system="",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this:"},
                        _FAKE_IMAGE_BLOCK,
                    ],
                }
            ],
        )
    finally:
        client.close()

    captured = [json.loads(l) for l in stdin_capture.read_text().splitlines() if l.strip()]
    assert captured
    user_line = captured[0]
    assert user_line["type"] == "user"
    content = user_line["message"]["content"]
    assert isinstance(content, list), f"expected content-list for image payload, got {type(content)}"
    # Image block must be present verbatim.
    image_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["source"]["data"] == _FAKE_IMAGE_BLOCK["source"]["data"]


# ───────────────────────── JSON schema: _resolve_json_schema ─────────────────────────


def test_resolve_json_schema_direct_string():
    schema_str = '{"type":"object"}'
    assert _resolve_json_schema({"json_schema": schema_str}) == schema_str


def test_resolve_json_schema_direct_dict():
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    out = _resolve_json_schema({"json_schema": schema})
    assert json.loads(out) == schema


def test_resolve_json_schema_openai_response_format_with_schema_key():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    kwargs = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "person", "schema": schema},
        }
    }
    out = _resolve_json_schema(kwargs)
    assert out is not None
    assert json.loads(out) == schema


def test_resolve_json_schema_openai_response_format_inline():
    """Some callers put the schema directly under ``json_schema`` with no nested ``schema`` key."""
    schema = {"type": "object", "properties": {"y": {"type": "number"}}, "required": ["y"]}
    kwargs = {"response_format": {"type": "json_schema", "json_schema": schema}}
    out = _resolve_json_schema(kwargs)
    assert out is not None
    assert json.loads(out) == schema


def test_resolve_json_schema_absent_returns_none():
    assert _resolve_json_schema({}) is None
    assert _resolve_json_schema({"messages": []}) is None
    assert _resolve_json_schema({"response_format": {"type": "text"}}) is None


# ───────────────────────── launch-spec argv emission ─────────────────────────


def test_launchspec_emits_json_schema_flag():
    spec = LaunchSpec(
        claude_bin="claude",
        session_id="s_1",
        model="sonnet",
        json_schema='{"type":"object"}',
    )
    argv = spec.build_argv()
    assert "--json-schema" in argv
    idx = argv.index("--json-schema")
    assert argv[idx + 1] == '{"type":"object"}'


def test_launchspec_omits_json_schema_flag_when_absent():
    spec = LaunchSpec(claude_bin="claude", session_id="s_1", model="sonnet")
    assert "--json-schema" not in spec.build_argv()


# ───────────────────────── json-schema triggers session rotation ─────────────────────────


def test_json_schema_is_emitted_in_subprocess_argv(fake_bin, monkeypatch, tmp_path):
    trace_file = tmp_path / "schema_trace.jsonl"
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    monkeypatch.setenv("CCCLI_FAKE_TRACE", str(trace_file))

    from agent.claude_cli import build_claude_cli_client

    client = build_claude_cli_client(
        model_default="sonnet", claude_bin=fake_bin, turn_budget_s=15
    )
    try:
        client.messages.create(
            system="",
            messages=[{"role": "user", "content": "hi"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object", "properties": {"answer": {"type": "string"}}}},
            },
        )
    finally:
        client.close()

    traces = _read_trace(trace_file)
    assert traces
    argv = traces[0]["argv"]
    assert "--json-schema" in argv
    idx = argv.index("--json-schema")
    assert json.loads(argv[idx + 1]) == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }


def test_result_structured_output_captured(monkeypatch):
    """When CC's result event carries a `structured_output` payload,
    EventTranslator must surface it on the final Message."""
    from agent.claude_cli.event_translator import EventTranslator

    t = EventTranslator(model="sonnet")
    for _ in t.consume({
        "type": "result",
        "subtype": "success",
        "usage": {"input_tokens": 100, "output_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        "structured_output": {"capital": "Tokyo", "population_millions": 125.7},
    }):
        pass
    for _ in t.finalize():
        pass
    msg = t.build_final_message()
    assert msg.ccli_structured_output == {"capital": "Tokyo", "population_millions": 125.7}


def test_changing_json_schema_between_turns_rotates_session(fake_bin, monkeypatch, tmp_path):
    """Two successive calls with different schemas must each spawn a new subprocess."""
    trace_file = tmp_path / "rot_trace.jsonl"
    monkeypatch.setenv("CCCLI_FAKE_MODE", "plain")
    monkeypatch.setenv("CCCLI_FAKE_TRACE", str(trace_file))

    from agent.claude_cli import build_claude_cli_client

    client = build_claude_cli_client(
        model_default="sonnet", claude_bin=fake_bin, turn_budget_s=15
    )
    try:
        client.messages.create(
            system="",
            messages=[{"role": "user", "content": "hi"}],
            json_schema={"type": "object", "properties": {"a": {"type": "string"}}},
        )
        client.messages.create(
            system="",
            messages=[{"role": "user", "content": "again"}],
            json_schema={"type": "object", "properties": {"b": {"type": "number"}}},
        )
    finally:
        client.close()

    traces = _read_trace(trace_file)
    argvs = [t["argv"] for t in traces]
    # Both spawns should have --json-schema, each carrying a DIFFERENT schema.
    schema_args = []
    for argv in argvs:
        if "--json-schema" in argv:
            idx = argv.index("--json-schema")
            schema_args.append(argv[idx + 1])
    assert len(schema_args) >= 2, f"expected ≥2 spawns carrying --json-schema, got {schema_args}"
    assert schema_args[0] != schema_args[1], "schema args should differ between the two calls"
