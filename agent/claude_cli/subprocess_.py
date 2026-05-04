"""Wrapper around a persistent ``claude -p`` subprocess.

Thread model
------------
* Main thread does all stdin writes and consumes parsed events.
* One daemon reader thread per subprocess drains stdout into a thread-safe
  queue, parsing each line as JSON. Non-JSON lines are dropped with a DEBUG
  log.
* One daemon reader thread drains stderr into a rolling buffer + forwards each
  line to an optional observer callback (used by :class:`EventTranslator` to
  detect OAuth expiry markers).

Concurrency invariants
----------------------
* Only one turn is in flight at a time per subprocess. Callers must serialize.
* Reader threads exit when the subprocess closes its streams or when
  :meth:`close` is called.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import queue
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from agent.claude_cli.errors import SpawnFailed, SubprocessCrashed

logger = logging.getLogger(__name__)


# ────────────────────────── debug log of every spawn ──────────────────────────
#
# Writes a self-contained reproducer per session so you can re-run the exact
# `claude -p` invocation outside Hermes to isolate adapter bugs from CLI bugs.
#
# Layout (under ~/.hermes/logs/claude_cli/):
#   <ts>_<short-session>.repro.sh   — bash script with the full command + heredoc
#                                     of stdin lines; close it manually if the
#                                     session is still live, then `bash <file>`
#   claude_cli.log                  — append-only, one-line summary per spawn
#                                     (timestamp + session id + cmd path)
#
# Disable by setting HERMES_CLAUDE_CLI_LOG=0.

_LOG_ENABLED_ENV = "HERMES_CLAUDE_CLI_LOG"
_HEREDOC_MARKER = "__HERMES_STDIN__"


def _log_root() -> Optional[Path]:
    if os.environ.get(_LOG_ENABLED_ENV, "1") == "0":
        return None
    try:
        from hermes_cli.config import get_hermes_home

        root = get_hermes_home() / "logs" / "claude_cli"
    except Exception:
        # Fall back to ~/.hermes/logs/claude_cli when hermes_cli isn't
        # importable (e.g. running the adapter standalone in tests).
        root = Path.home() / ".hermes" / "logs" / "claude_cli"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.debug("could not create claude_cli log dir: %s", e)
        return None
    return root


class _SessionLogger:
    """Writes a pasteable bash reproducer for one ClaudeSubprocess session."""

    def __init__(self, session_id: str, argv: list[str], cwd: Optional[str]) -> None:
        self.repro_path: Optional[Path] = None
        self._lock = threading.Lock()
        self._closed = False
        self._heredoc_open = False

        root = _log_root()
        if root is None:
            return
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_sid = session_id.split("-")[0] if "-" in session_id else session_id[:8]
        self.repro_path = root / f"{ts}_{short_sid}.repro.sh"
        cmd = shlex.join(argv)
        try:
            with self.repro_path.open("w") as f:
                f.write("#!/bin/bash\n")
                f.write(f"# claude_cli reproducer — session {session_id}\n")
                f.write(f"# Spawned at {ts}\n")
                f.write(f"# Run with:  bash {self.repro_path}\n")
                f.write("#\n")
                f.write("# Notes:\n")
                f.write("#   * If this session is still live (heredoc unclosed),\n")
                f.write(f"#     append `{_HEREDOC_MARKER}` on its own line first.\n")
                f.write("#   * To re-run after success, replace --session-id\n")
                f.write("#     <uuid> with `$(uuidgen)` or delete the matching\n")
                f.write("#     ~/.claude/projects/<cwd-hash>/<uuid>.jsonl file —\n")
                f.write("#     CC refuses to reuse a session id.\n")
                f.write("set -e\n")
                if cwd:
                    f.write(f"cd {shlex.quote(cwd)}\n")
                f.write("\n")
                f.write(f"{cmd} <<'{_HEREDOC_MARKER}'\n")
            try:
                self.repro_path.chmod(0o755)
            except Exception:
                pass
            self._heredoc_open = True
            # Append a one-line summary to the rolling index.
            try:
                root_log = root.parent / "claude_cli.log"
                with root_log.open("a") as f:
                    f.write(f"{ts}  session={session_id}  repro={self.repro_path.name}\n")
            except Exception:
                pass
        except Exception as e:
            logger.debug("claude_cli log write failed: %s", e)
            self.repro_path = None

    def append_stdin(self, line: str) -> None:
        """Record one stdin JSON line for later replay."""
        if self.repro_path is None or self._closed:
            return
        with self._lock:
            try:
                with self.repro_path.open("a") as f:
                    f.write(line.rstrip("\n") + "\n")
            except Exception as e:
                logger.debug("claude_cli log append failed: %s", e)

    def close(self, *, exit_code: int) -> None:
        if self.repro_path is None or self._closed:
            return
        with self._lock:
            self._closed = True
            try:
                with self.repro_path.open("a") as f:
                    if self._heredoc_open:
                        f.write(f"{_HEREDOC_MARKER}\n")
                    f.write(f"# subprocess exited rc={exit_code}\n")
            except Exception:
                pass

# Sentinels for the stdout queue.
_STDOUT_EOF = object()


@dataclass
class LaunchSpec:
    """All the flags that shape a ``claude -p`` invocation.

    Constructed by :class:`~agent.claude_cli.session.SessionManager` from the
    Anthropic ``api_kwargs`` plus adapter config.
    """

    claude_bin: str
    session_id: str
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    append_system_prompt: Optional[str] = None
    effort: Optional[str] = None
    cwd: Optional[str] = None
    resume: bool = False
    include_partial_messages: bool = True
    json_schema: Optional[str] = None  # pre-serialized JSON string (or None to omit)
    extra_flags: list[str] = field(default_factory=list)

    def build_argv(self) -> list[str]:
        argv = [
            self.claude_bin,
            "-p",
            "--verbose",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--tools",
            "",
            "--permission-mode",
            "bypassPermissions",
            # Disable MCP servers too — ``--tools ""`` only covers built-in
            # tools, not MCP. Without this, user-configured MCP servers
            # (Slack, Linear, …) get advertised to the model and confuse
            # the "you have NO tools" contract in the envelope prompt.
            "--strict-mcp-config",
        ]
        if self.include_partial_messages:
            argv.append("--include-partial-messages")
        if self.model:
            argv += ["--model", self.model]
        if self.system_prompt is not None:
            argv += ["--system-prompt", self.system_prompt]
        if self.append_system_prompt is not None:
            argv += ["--append-system-prompt", self.append_system_prompt]
        if self.effort:
            argv += ["--effort", self.effort]
        if self.json_schema:
            argv += ["--json-schema", self.json_schema]
        if self.resume:
            argv += ["--resume", self.session_id]
        else:
            argv += ["--session-id", self.session_id]
        argv += list(self.extra_flags)
        return argv


class ClaudeSubprocess:
    """Persistent ``claude -p`` subprocess with threaded stdio pumps."""

    def __init__(
        self,
        spec: LaunchSpec,
        *,
        stderr_ring_bytes: int = 4000,
        stderr_observer: Optional[Callable[[str], None]] = None,
        kill_grace_s: float = 0.5,
    ) -> None:
        self.spec = spec
        self._stderr_ring_bytes = stderr_ring_bytes
        self._stderr_observer = stderr_observer
        self._kill_grace_s = kill_grace_s

        self._proc: Optional[subprocess.Popen] = None
        self._stdout_q: "queue.Queue[Any]" = queue.Queue()
        self._stderr_buf: list[str] = []
        self._stderr_total_bytes = 0
        self._stderr_lock = threading.Lock()
        self._reader_threads: list[threading.Thread] = []
        self._closed = False
        self._session_log: Optional[_SessionLogger] = None

    # ───────────────────────── lifecycle ─────────────────────────

    def start(self) -> None:
        """Spawn the subprocess and start reader threads."""
        if self._proc is not None:
            raise RuntimeError("subprocess already started")
        argv = self.spec.build_argv()
        # Write a self-contained reproducer for this session BEFORE spawning,
        # so the file exists even if spawn itself fails.
        self._session_log = _SessionLogger(self.spec.session_id, argv, self.spec.cwd)
        if self._session_log.repro_path is not None:
            logger.info("claude_cli reproducer: %s", self._session_log.repro_path)
        logger.info(
            "spawning claude: bin=%s session=%s resume=%s model=%s cwd=%s",
            self.spec.claude_bin,
            self.spec.session_id,
            self.spec.resume,
            self.spec.model,
            self.spec.cwd,
        )
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.spec.cwd,
                bufsize=1,
                # Detach from the parent's signal group so Ctrl-C handling in
                # Hermes doesn't immediately propagate to CC before we've had
                # a chance to cleanly interrupt.
                start_new_session=True,
            )
        except FileNotFoundError as e:
            raise SpawnFailed(
                f"failed to launch claude binary at {self.spec.claude_bin!r}: {e}"
            ) from e
        except OSError as e:
            raise SpawnFailed(f"failed to spawn claude: {e}") from e

        t_out = threading.Thread(
            target=self._pump_stdout, name="claude-cli-stdout", daemon=True
        )
        t_err = threading.Thread(
            target=self._pump_stderr, name="claude-cli-stderr", daemon=True
        )
        t_out.start()
        t_err.start()
        self._reader_threads = [t_out, t_err]

    def close(self, *, kill: bool = True) -> int:
        """Terminate the subprocess. Returns the exit code (or -1 if unknown)."""
        if self._closed:
            return -1
        self._closed = True
        proc = self._proc
        if proc is None:
            return -1
        try:
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            except Exception:
                pass
            if kill and proc.poll() is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                t0 = time.monotonic()
                while time.monotonic() - t0 < self._kill_grace_s:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.02)
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
            rc = proc.wait(timeout=5)
        finally:
            for t in self._reader_threads:
                t.join(timeout=1.0)
            self._reader_threads = []
            if self._session_log is not None:
                try:
                    self._session_log.close(exit_code=rc if isinstance(rc, int) else -1)
                except Exception:
                    pass
        return rc if rc is not None else -1

    # ─────────────────────── stdin: write turn ───────────────────────

    def send_user_line(self, payload: dict[str, Any]) -> None:
        """Send one JSON line to the subprocess stdin (flushed)."""
        proc = self._ensure_proc()
        if proc.stdin is None:
            raise SubprocessCrashed("subprocess stdin is None", stderr_tail=self.stderr_tail())
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            proc.stdin.write(line)
            proc.stdin.flush()
        except BrokenPipeError as e:
            raise SubprocessCrashed(
                f"stdin pipe broken: {e}", stderr_tail=self.stderr_tail()
            ) from e
        if self._session_log is not None:
            self._session_log.append_stdin(line)

    # ─────────────────────── stdout: read events ─────────────────────

    def events(self, *, overall_timeout_s: float) -> Iterator[dict[str, Any]]:
        """Yield parsed CC stream-json events until the subprocess pauses or dies.

        Ends when:
          * a ``result`` event is yielded (caller decides whether to continue);
          * stdout EOF;
          * overall timeout elapses — raises :class:`TurnTimeout` via caller.
        """
        deadline = time.monotonic() + overall_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                item = self._stdout_q.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            if item is _STDOUT_EOF:
                return
            yield item
            if isinstance(item, dict) and item.get("type") == "result":
                return

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stderr_tail(self) -> str:
        with self._stderr_lock:
            return "".join(self._stderr_buf)

    # ─────────────────────────── internals ───────────────────────────

    def _ensure_proc(self) -> subprocess.Popen:
        if self._proc is None:
            raise RuntimeError("subprocess not started")
        return self._proc

    def _pump_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            for line in self._proc.stdout:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.debug("non-json stdout line dropped: %s", stripped[:200])
                    continue
                self._stdout_q.put(obj)
        except Exception as e:
            logger.debug("stdout pump exception: %s", e)
        finally:
            self._stdout_q.put(_STDOUT_EOF)

    def _pump_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        try:
            for line in self._proc.stderr:
                if not line:
                    continue
                if self._stderr_observer is not None:
                    try:
                        self._stderr_observer(line)
                    except Exception:
                        pass
                with self._stderr_lock:
                    self._stderr_buf.append(line)
                    self._stderr_total_bytes += len(line)
                    # Rolling cap.
                    while self._stderr_total_bytes > self._stderr_ring_bytes and self._stderr_buf:
                        dropped = self._stderr_buf.pop(0)
                        self._stderr_total_bytes -= len(dropped)
        except Exception as e:
            logger.debug("stderr pump exception: %s", e)


def send_signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Send ``sig`` to the subprocess's process group if we created one."""
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except Exception:
            pass
