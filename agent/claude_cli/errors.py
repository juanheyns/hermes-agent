"""Typed errors raised by the claude_cli adapter."""

from __future__ import annotations


class ClaudeCliError(Exception):
    """Base class for all claude_cli adapter errors.

    Attributes
    ----------
    code : str
        Machine-readable short code (``spawn_failed``, ``claude_crashed``, …).
    stderr_tail : str | None
        Tail of the subprocess stderr at the time the error was raised, if
        available. Bounded size — safe to log.
    session_id : str | None
        The CC session id this error pertains to, if known.
    """

    code: str = "internal"

    def __init__(
        self,
        message: str = "",
        *,
        stderr_tail: str | None = None,
        session_id: str | None = None,
    ) -> None:
        super().__init__(message or self.code)
        self.message = message
        self.stderr_tail = stderr_tail
        self.session_id = session_id


class SpawnFailed(ClaudeCliError):
    """`claude` binary missing or could not be launched."""

    code = "spawn_failed"


class SubprocessCrashed(ClaudeCliError):
    """`claude -p` exited unexpectedly during a turn."""

    code = "claude_crashed"


class OAuthExpired(ClaudeCliError):
    """CC reported an auth failure. User must re-authenticate via ``claude auth``."""

    code = "oauth_expired"


class TurnTimeout(ClaudeCliError):
    """No ``result`` event received within the turn budget."""

    code = "turn_timeout"


class ToolParseError(ClaudeCliError):
    """Could not parse a ``<hermes:tool>`` envelope from the assistant output."""

    code = "tool_parse_error"


class ClientBusy(ClaudeCliError):
    """Another turn is already in flight on this client."""

    code = "client_busy"


class ApiError(ClaudeCliError):
    """CC reported an upstream API error in its ``result`` event.

    Examples
    --------
    * 400 with body "Third-party apps now draw from extra usage, not plan
      limits." — user has exhausted their Claude plan's third-party-app quota.
    * 401 / 403 — auth failure (also surfaced as :class:`OAuthExpired` via the
      stderr observer when CC prints "Please run claude auth").
    * 429 — rate limit hit before the request reached the model.
    """

    code = "api_error"

    def __init__(
        self,
        message: str = "",
        *,
        status_code: int | None = None,
        stderr_tail: str | None = None,
        session_id: str | None = None,
    ) -> None:
        super().__init__(message, stderr_tail=stderr_tail, session_id=session_id)
        self.status_code = status_code


class QuotaExceeded(ApiError):
    """Specific subtype for plan / extra-usage / billing exhaustion.

    Hermes surfaces this with a tailored message so the user knows the fix
    is to top up their Anthropic plan or workspace, not to retry.
    """

    code = "quota_exceeded"
