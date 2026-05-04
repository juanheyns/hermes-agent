"""Tests that CC-surfaced API errors are translated into typed exceptions.

When CC's ``result`` event has ``is_error: true`` (typically from an
upstream Anthropic 4xx/5xx), my translator must surface a meaningful
:class:`ApiError` subclass instead of silently producing an empty Message.
"""

from __future__ import annotations

import pytest

from agent.claude_cli.errors import (
    ApiError,
    OAuthExpired,
    QuotaExceeded,
)
from agent.claude_cli.event_translator import EventTranslator


def _err_result(body: str, status: int = 400) -> dict:
    return {
        "type": "result",
        "subtype": "success",  # CC reports subtype=success even for upstream errors
        "is_error": True,
        "api_error_status": status,
        "result": body,
        "stop_reason": "stop_sequence",
        "session_id": "s_test",
        "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }


def test_quota_exhausted_message_classified_as_quota_exceeded():
    """Real error from Hermes user: third-party-app quota hit."""
    t = EventTranslator(model="opus")
    body = (
        'API Error: 400 {"type":"error","error":{"type":"invalid_request_error",'
        '"message":"Third-party apps now draw from extra usage, not plan limits. '
        'Ask your workspace admin to add more and keep going."},'
        '"request_id":"req_abc"}'
    )
    list(t.consume(_err_result(body, status=400)))
    assert isinstance(t.pending_error, QuotaExceeded)
    assert t.pending_error.status_code == 400
    assert "Third-party apps" in str(t.pending_error.message)


def test_generic_api_error_classified_as_api_error():
    t = EventTranslator(model="sonnet")
    list(t.consume(_err_result("API Error: 500 {\"type\":\"server_error\"}", status=500)))
    assert isinstance(t.pending_error, ApiError)
    assert not isinstance(t.pending_error, QuotaExceeded)
    assert t.pending_error.status_code == 500


def test_credit_balance_message_classified_as_quota_exceeded():
    t = EventTranslator(model="sonnet")
    list(t.consume(_err_result("Your credit balance is too low to access the API.", status=400)))
    assert isinstance(t.pending_error, QuotaExceeded)


def test_billing_message_classified_as_quota_exceeded():
    t = EventTranslator(model="sonnet")
    list(t.consume(_err_result("Billing required to continue.", status=402)))
    assert isinstance(t.pending_error, QuotaExceeded)


def test_no_error_when_result_is_clean_success():
    """Regression guard: a normal success result must NOT trip the error path."""
    t = EventTranslator(model="sonnet")
    list(t.consume({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 100,
        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }))
    assert t.pending_error is None


def test_oauth_expired_in_stderr_still_takes_priority():
    """If stderr says OAuth expired AND result has is_error, OAuth path fires first."""
    t = EventTranslator(model="sonnet")
    t.on_subprocess_stderr("Error 401: OAuth token expired. Please run claude auth.")
    assert isinstance(t.pending_error, OAuthExpired)
    # Subsequent result event with is_error should NOT overwrite.
    list(t.consume(_err_result("some other error", status=400)))
    assert isinstance(t.pending_error, OAuthExpired)


def test_api_error_attached_to_session_id_when_present():
    t = EventTranslator(model="sonnet")
    list(t.consume(_err_result("rate limited", status=429)))
    assert isinstance(t.pending_error, ApiError)
    # status_code is the key field for downstream classifiers
    assert t.pending_error.status_code == 429
