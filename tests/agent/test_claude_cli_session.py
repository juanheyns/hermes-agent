"""Unit tests for agent.claude_cli.session — pure logic parts only.

Tests here cover message translation, digesting, and divergence detection.
Subprocess-lifecycle tests live in ``test_claude_cli_subprocess_mock.py`` since
they need a real child process.
"""

from __future__ import annotations

import pytest

from agent.claude_cli.session import (
    SessionManager,
    _digest_message,
    render_user_content,
)

# Back-compat alias so the existing test bodies continue to read well.
_render_user_content_to_text = render_user_content


# ───────────────────── _render_user_content_to_text ─────────────────────


def test_render_plain_string():
    assert _render_user_content_to_text("hello") == "hello"


def test_render_list_of_text_blocks():
    content = [
        {"type": "text", "text": "one"},
        {"type": "text", "text": "two"},
    ]
    out = _render_user_content_to_text(content)
    assert "one" in out and "two" in out
    assert out.count("\n\n") >= 1


def test_render_tool_result_block():
    content = [
        {
            "type": "tool_result",
            "tool_use_id": "t_42",
            "content": "output body",
        },
    ]
    out = _render_user_content_to_text(content)
    assert '<hermes:tool_result id="t_42">' in out
    assert "output body" in out
    assert out.endswith("</hermes:tool_result>")


def test_render_tool_result_with_error_flag():
    content = [
        {
            "type": "tool_result",
            "tool_use_id": "t_err",
            "content": "boom",
            "is_error": True,
        },
    ]
    out = _render_user_content_to_text(content)
    assert 'is_error="true"' in out
    assert "<error>boom</error>" in out


def test_render_tool_result_with_list_content():
    content = [
        {
            "type": "tool_result",
            "tool_use_id": "t_x",
            "content": [
                {"type": "text", "text": "line one"},
                {"type": "text", "text": "line two"},
            ],
        }
    ]
    out = _render_user_content_to_text(content)
    assert "line one" in out
    assert "line two" in out


def test_render_image_block_passes_through_as_content_list():
    """Vision is supported: image blocks flow through verbatim as content-list."""
    img = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
    out = render_user_content([{"type": "text", "text": "see this:"}, img])
    # With a non-text block present, output MUST be a list of content blocks.
    assert isinstance(out, list)
    assert out[0] == {"type": "text", "text": "see this:"}
    assert out[1] == img


def test_render_pure_text_content_still_collapses_to_string():
    """No images → still a string payload (preserves the common case)."""
    out = render_user_content(
        [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
    )
    assert isinstance(out, str)
    assert "one" in out and "two" in out


def test_render_mixed_image_and_tool_result_produces_block_list():
    """tool_result becomes a synthetic text block inside the content list."""
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}},
        {"type": "tool_result", "tool_use_id": "t_7", "content": "42"},
    ]
    out = render_user_content(content)
    assert isinstance(out, list)
    assert out[0]["type"] == "image"
    assert out[1]["type"] == "text"
    assert '<hermes:tool_result id="t_7">42</hermes:tool_result>' in out[1]["text"]


# ────────────────────────── _digest_message ──────────────────────────


def test_digest_stable_for_identical_input():
    m = {"role": "user", "content": "hello"}
    assert _digest_message(m) == _digest_message(m)


def test_digest_changes_when_content_changes():
    a = {"role": "user", "content": "hello"}
    b = {"role": "user", "content": "hi"}
    assert _digest_message(a) != _digest_message(b)


def test_digest_changes_when_role_changes():
    a = {"role": "user", "content": "hi"}
    b = {"role": "assistant", "content": "hi"}
    assert _digest_message(a) != _digest_message(b)


# ───────────────── SessionManager (no subprocess spawned) ─────────────────


def test_session_manager_initial_state_is_none():
    mgr = SessionManager(claude_bin="/bin/false", cwd=None)
    assert mgr.state is None


def test_session_manager_close_is_idempotent():
    mgr = SessionManager(claude_bin="/bin/false", cwd=None)
    mgr.close()
    mgr.close()  # must not raise
    assert mgr.state is None
