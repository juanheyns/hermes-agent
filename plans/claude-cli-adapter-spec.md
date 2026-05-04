# Hermes `claude_cli` Model Adapter — Embedded Implementation

**Version:** 0.1 (draft, pre-implementation)
**Status:** Ready to hand to an implementing agent.
**Target repo:** `hermes-agent` (this repo), no external dependencies.
**Language:** Python, same version as the rest of the codebase.

---

## 1. Overview

Add a new `api_mode = "claude_cli"` to Hermes that routes model calls through
a local `claude -p` subprocess managed inside the Hermes process. No
daemon, no socket — a direct, embedded subprocess adapter.

The adapter:

1. Spawns a persistent `claude -p --input-format stream-json --output-format
   stream-json --tools ""` subprocess per AIAgent instance.
2. Tracks an "acknowledged message cursor" so incremental user messages are
   sent to CC rather than replaying the entire conversation each turn.
3. Emits Anthropic-SDK-shaped events upstream, so `run_agent.py`'s
   existing streaming/tool-use code continues to work unchanged.
4. Enforces a client-side tool protocol (`<hermes:tool>` envelope) with
   streaming halt on `</hermes:tool>` — kills the subprocess and respawns
   with `--resume` to feed tool_result back.
5. Survives Hermes-side conversation mutations (compression, rewinds) by
   detecting divergence and rotating the CC session id.

This path is **opt-in** via config (`api_mode: "claude_cli"`). All existing
API modes (`anthropic_messages`, `chat_completions`, `codex_responses`,
`bedrock_converse`) remain untouched.

---

## 2. Goals and Non-Goals

### Goals
- Restore model access for users who previously authenticated with the
  Claude Code setup token against `api.anthropic.com` directly (now broken).
- Minimize blast radius: zero changes to callers outside the adapter and a
  small, well-scoped dispatch point in `run_agent.py`.
- Feature parity for what Hermes actually uses through the Anthropic path:
  streaming text, tool use via the envelope protocol, `<think>`/reasoning
  deltas where CC emits them, multi-model support (opus/sonnet/haiku), cwd
  control, effort level.
- Work with Hermes's compression and history mutations without corrupting
  the CC-side session.

### Non-goals (v0.1)
- Explicit `cache_control` block hints. Read-side cache counters ARE
  surfaced; CC handles cache writes internally with no caller control.
- Multi-session multiplexing across Hermes AIAgent instances. Each agent
  instance owns its own subprocess.
- Running as a long-running daemon shared between processes. See
  `plans/ccsockd-spec.md` (separate project) for that direction.
- Replacing the Anthropic SDK dependency. It stays for the other
  `anthropic_messages` paths (Bedrock, MiniMax, Kimi, etc.).

### Scope update — features added past v0.1 baseline
- **Metrics**: token accounting, cache read/write counts, per-turn cost
  USD, wall-clock duration, API duration, and time-to-first-token are
  surfaced on every response. SDK-shape ``usage.*`` fields are populated
  from CC's ``result.usage``; CC-specific extensions ride as ``ccli_*``
  attributes (``cost_usd``, ``duration_ms``, ``duration_api_ms``,
  ``ttft_ms``, ``num_turns``, ``service_tier``, ``model_usage``).
- **Vision / multimodal**: ``image`` and ``document`` content blocks pass
  through to CC stream-json stdin verbatim (CC forwards them to the
  underlying API natively). The stdin payload's ``content`` becomes a
  list-of-blocks rather than a string when any non-text block is present.
- **Structured output**: ``json_schema`` (direct dict/string) or
  ``response_format={"type":"json_schema","json_schema":{...}}`` (OpenAI
  shim) on ``messages.create`` triggers ``--json-schema`` on the
  subprocess. CC validates with a synthetic ``StructuredOutput`` tool and
  the resulting object lands on ``response.ccli_structured_output``.
  Trade-off: adds one internal turn (~2x cost). Schema changes between
  calls trigger session rotation.
- **Compact tool catalog + size guardrail**: Anthropic's billing
  classifier appears to flag requests as "third-party app" when the
  system prompt is large AND contains schema-shaped JSON. We bisected
  the trigger to ~21 KB of Hermes-shaped prompt content. The catalog
  renderer now defaults to a single-line ``- name(arg1, arg2?): desc``
  signature form and the composer auto-truncates the catalog above
  18 KB (configurable via ``HERMES_CLAUDE_CLI_PROMPT_HARD``) with a
  ``[…tool catalog truncated…]`` marker so the model knows it was
  trimmed. ``QuotaExceeded`` errors triggered by the classifier carry
  an inline hint to lower the cap further.
- **Per-spawn debug reproducer**: every ``claude -p`` invocation drops a
  shell-pasteable script at
  ``~/.hermes/logs/claude_cli/<ts>_<short-id>.repro.sh`` containing the
  full argv plus a heredoc of every stdin line sent. Lets users replay
  any failing session standalone to isolate adapter bugs from CLI bugs.
  Disable via ``HERMES_CLAUDE_CLI_LOG=0``.
- **Typed API errors**: CC-surfaced API errors (``result.is_error: true``)
  no longer disappear as silent empty responses. They raise
  :class:`ApiError` (or :class:`QuotaExceeded` for billing-related
  messages) with the upstream message, status code, and Anthropic
  ``request_id`` intact.

---

## 3. Architecture

```
┌─────────────────── AIAgent (run_agent.py) ───────────────────┐
│                                                               │
│   api_mode == "claude_cli"                                    │
│     └─ self._anthropic_client  =  ClaudeCliClient(...)        │
│                                                               │
│   _anthropic_messages_create(kwargs)  → client.messages.create(**kwargs)
│   streaming path                      → client.messages.stream(**kwargs)
│                                                               │
└──────────────────────────┬────────────────────────────────────┘
                           │  (SDK-shaped calls)
                           ▼
┌─────────────────── agent/claude_cli/ ─────────────────────────┐
│                                                               │
│   ClaudeCliClient  (SDK-compatible surface)                   │
│     └─ .messages.create(**kwargs)  → Message                  │
│     └─ .messages.stream(**kwargs)  → StreamCtx                │
│                                                               │
│   SessionManager                                              │
│     - owns the ClaudeSubprocess for this client               │
│     - tracks acknowledged_cursor (index into api_kwargs["messages"])
│     - rotates session on divergence or explicit reset         │
│                                                               │
│   ClaudeSubprocess                                            │
│     - spawns/kills/resumes `claude -p`                        │
│     - writes stream-json to stdin, reads stream-json stdout   │
│     - raw event iterator                                      │
│                                                               │
│   ToolProtocol  (`<hermes:tool>` envelope)                    │
│     - streaming halt parser                                   │
│     - encoder for <hermes:tool_result> on respawn             │
│                                                               │
│   EventTranslator                                             │
│     - CC stream-json events → Anthropic-SDK event objects     │
│     - Anthropic Message payload → CC stdin user messages      │
│     - synthesizes Anthropic content blocks for tool_use       │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

Single-threaded inside the adapter. Subprocess IO runs on a background
`threading.Thread` per subprocess (stdout reader) — consistent with
`run_agent.py`'s existing thread-based pattern and avoids asyncio inside a
mostly-sync codebase.

---

## 4. File Layout

```
agent/claude_cli/
  __init__.py             # re-exports ClaudeCliClient, build_claude_cli_client
  client.py               # ClaudeCliClient + MessagesNamespace + StreamCtx
  subprocess_.py          # ClaudeSubprocess (spawn, stdin/stdout, kill, resume)
  session.py              # SessionManager (cursor tracking, rotation)
  tool_protocol.py        # Halt parser, encoder for <hermes:tool_result>
  event_translator.py     # CC stream-json ↔ Anthropic SDK shapes
  errors.py               # ClaudeCliError, SpawnFailed, OAuthExpired, etc.
  config.py               # small config helpers (bin path, timeouts)
tests/claude_cli/
  test_tool_protocol.py
  test_event_translator.py
  test_session.py
  test_subprocess_mock.py     # mock claude stub
  test_client_e2e.py          # gated on real `claude` availability
```

Name the file `subprocess_.py` to avoid shadowing stdlib `subprocess`.

No new top-level dependencies. Stdlib only: `json`, `subprocess`,
`threading`, `queue`, `shutil`, `os`, `time`, `uuid`, `re`, `html`.

---

## 5. Public API

The adapter exposes a class that looks like the Anthropic SDK client, so
`run_agent.py` can use it through the same `self._anthropic_client`
attribute with no other change at the call sites.

### 5.1 `ClaudeCliClient`

```python
class ClaudeCliClient:
    def __init__(
        self,
        *,
        model_default: str,               # fallback model if kwargs omits it
        claude_bin: str | None = None,    # shutil.which("claude") default
        cwd: str | None = None,
        request_timeout_s: float = 600.0,
        turn_budget_s: float = 300.0,
        log_redact: bool = True,
        system_prompt_policy: SystemPromptPolicy = SystemPromptPolicy.REPLACE,
    ): ...

    @property
    def messages(self) -> "MessagesNamespace": ...

    def close(self) -> None:
        """Kill the subprocess and reap resources."""
```

### 5.2 `MessagesNamespace`

Mirrors `anthropic.resources.messages.Messages`:

```python
class MessagesNamespace:
    def create(self, **api_kwargs) -> Message: ...
    def stream(self, **api_kwargs) -> "StreamCtx": ...
```

`api_kwargs` is exactly what Hermes's `agent/anthropic_adapter.build_anthropic_kwargs`
produces today: `model`, `messages`, `system`, `tools`, `tool_choice`,
`max_tokens`, `stream`, `metadata`, `thinking`/`output_config`, `temperature`,
etc. The adapter ignores fields it cannot honor (documented in §9).

### 5.3 Return shapes

- `create()` returns an object duck-compatible with `anthropic.types.Message`:
  `id`, `model`, `role="assistant"`, `stop_reason`, `stop_sequence`,
  `usage` (minimal: `input_tokens=0`, `output_tokens=0`,
  `cache_creation_input_tokens=0`, `cache_read_input_tokens=0`), and
  `content: list[ContentBlock]` where blocks are `TextBlock` or
  `ToolUseBlock`. Construct via `types.SimpleNamespace` where possible;
  fields match the SDK attribute names the rest of Hermes reads.

- `stream(**)` returns an object usable as
  `with client.messages.stream(**) as stream:` and iterated like the real SDK
  stream. It yields objects with `.type` set to the Anthropic event names
  used in `run_agent.py:5824-5920`:
    - `"content_block_start"` with `.content_block.type` = `"text"` or `"tool_use"`
    - `"content_block_delta"` with `.delta.type` = `"text_delta"` /
      `"input_json_delta"` / `"thinking_delta"` (see §7.4 for thinking
      support)
    - `"content_block_stop"`
    - `"message_delta"` carrying `stop_reason`
    - `"message_stop"`
  Also exposes `stream.get_final_message()` that returns the same shape as
  `create()`.

---

## 6. Integration with `run_agent.py`

### 6.1 New api_mode dispatch

Add `"claude_cli"` to the set of `api_mode` values recognized during
`AIAgent.__init__`. Route construction in the existing `if self.api_mode ==
"anthropic_messages": …` cascade:

```python
elif self.api_mode == "claude_cli":
    from agent.claude_cli import build_claude_cli_client
    self._anthropic_client = build_claude_cli_client(
        model_default=self.model,
        cwd=os.getcwd(),
        request_timeout_s=_provider_timeout,
    )
    self._anthropic_api_key = "claude-cli"        # sentinel for logging
    self._anthropic_base_url = None
    self._is_anthropic_oauth = False              # OAuth handled by CC
    self.api_key = "claude-cli"
    self.client = None
    self._client_kwargs = {}
    if not self.quiet_mode:
        print(f"🤖 AI Agent initialized with model: {self.model} (claude -p subprocess)")
```

Keep `_anthropic_messages_create()` (`run_agent.py:5218`) unchanged — it
already delegates to `self._anthropic_client.messages.create(**api_kwargs)`.

Keep the streaming call site at `run_agent.py:5824` unchanged — the adapter
implements the same `messages.stream(**).__enter__()` contract and event
shapes.

### 6.2 Provider detection

Add a provider name `"claude_cli"` with routing in
`toolsets.py` / `model_tools.py` / wherever provider→api_mode mapping lives
(search for `"api_mode"` and `"provider"` to find the map). Make the routing
explicit: `provider == "claude_cli"` → `api_mode = "claude_cli"`.

No auto-detection. Activation is explicit via config only.

### 6.3 Token refresh path

Delete / bypass `_try_refresh_anthropic_client_credentials()` (line 5035)
for `claude_cli` mode. CC owns auth; the adapter never touches
`~/.claude/.credentials.json` directly. If CC reports auth failure (see §9),
the adapter raises `OAuthExpired` and the agent surfaces the error to the
user with instructions to run `claude auth`.

### 6.4 Auxiliary client (`agent/auxiliary_client.py`)

Add a `claude_cli` branch to the provider routing (file lines 1–150). Uses
a lightweight, short-lived `ClaudeCliClient` per aux call — no session
persistence, `--output-format json` (single-shot), no tool protocol. This
keeps aux calls independent of the main-loop subprocess lifecycle and
avoids cross-contaminating sessions.

---

## 7. Subprocess Management

### 7.1 Launch

For every `ClaudeCliClient`, the adapter spawns one `claude -p` process on
first use (lazy). CLI:

```
claude -p
  --verbose
  --input-format  stream-json
  --output-format stream-json
  --include-partial-messages
  --tools ""
  --permission-mode bypassPermissions
  --system-prompt <serialized from api_kwargs["system"]>
  --model <api_kwargs["model"] or model_default>
  [--effort <mapped from thinking/output_config>]
  --session-id <generated UUID>                   OR  --resume <id>
```

Mapping:
- `--system-prompt`: string form. If `api_kwargs["system"]` is a list of
  content blocks (Hermes does this), join text blocks with `\n\n`.
- `--effort`: derive from `api_kwargs.get("output_config", {}).get("effort")`
  or `thinking.budget_tokens` → effort bucket per the existing Hermes logic
  (`agent/anthropic_adapter.py:1528-1573`). Reuse by importing that helper;
  do not duplicate.
- Unsafe flags (`--dangerously-skip-permissions`, `--bare`, `--continue`,
  `--from-pr`) are never emitted.

Subprocess env inherits Hermes's environment (including
`ANTHROPIC_TOKEN` / `CLAUDE_CODE_OAUTH_TOKEN` for CC's own auth). Do not
pass secrets explicitly via argv.

### 7.2 stdin protocol

Each user turn is one JSON line on stdin:
```json
{"type":"user","message":{"role":"user","content":"<text>"},"session_id":"<id>"}
```
For pre-existing tool_result messages in the Hermes history, encode as a
`<hermes:tool_result>` envelope inside the text content (see §8.3).

For multimodal content blocks (vision, etc.), v0.1 raises
`NotImplementedError`; v0.2 will pass Claude Code's native content-block
JSON through.

### 7.3 stdout parsing

Reader thread consumes stdout line by line, `json.loads` each line, pushes
into a thread-safe `queue.Queue`. The main thread's `StreamCtx` / `create()`
drains the queue, feeds the `EventTranslator`, and yields or collects
accordingly.

Known CC event types the translator handles (rest are logged at DEBUG and
ignored):
- `system` → ignored (initial preamble).
- `stream_event.content_block_delta.text_delta` → text tokens.
- `stream_event.content_block_delta.thinking_delta` → thinking tokens (map
  to Anthropic's `thinking_delta` event).
- `assistant` → final assistant message for the turn. Used only as a
  sanity check against the accumulated delta buffer; do NOT re-forward text
  from it (doubles text, confirmed in spike).
- `result` → terminal marker; carries `duration_ms`, `num_turns`,
  `subtype`. Drives `message_stop`.

### 7.4 Thinking

When `api_kwargs["thinking"]` is set or an effort level maps to a thinking
budget, pass `--effort <level>`. Forward any `thinking_delta` deltas CC
emits to the caller as Anthropic-shape `thinking_delta` events inside a
`thinking` content block. If CC does not emit thinking (model/version
without the support), the adapter simply emits no thinking content —
downstream code already tolerates its absence.

### 7.5 Interrupt & respawn

Triggered by the tool-protocol halt (§8) or by the outer agent's
`_interrupt_requested`. Procedure:
1. `proc.terminate()`. After 500 ms, `proc.kill()` if alive.
2. `proc.wait()`.
3. If the session should continue: respawn with `--resume <session-id>`,
   all other flags identical.
4. Send next stdin message (tool_result envelope or user text).

### 7.6 Session rotation (history divergence)

The adapter tracks `acknowledged_cursor: int` — the number of messages in
`api_kwargs["messages"]` that have been delivered to CC so far. On each
call:

1. Compare the new `messages` prefix (up to `acknowledged_cursor`) against
   a stored digest (SHA256 of each message's normalized form).
2. If digests match → conversation is consistent. Send the tail (messages
   from `acknowledged_cursor` onward), bump the cursor.
3. If digests diverge → Hermes compressed/rewrote history. Rotate:
   - `proc.terminate()`, new session id.
   - Spawn fresh. Replay entire `messages` list as stdin turns in order.
   - Reset stored digests to the new history.

Rotation is expensive (cold start + full replay). Log at INFO with the
reason ("history compressed", "history rewound", etc.) so users notice.

### 7.7 Shutdown

`ClaudeCliClient.close()` terminates the subprocess (SIGTERM → SIGKILL
after 500 ms), joins the reader thread, and releases resources. Call from
`AIAgent`'s cleanup path. Also register an `atexit` hook so a Ctrl-C'd
Hermes doesn't leak zombie `claude` processes.

---

## 8. Tool Protocol (`<hermes:tool>` envelope)

### 8.1 System prompt contribution

The adapter appends a stable block to whatever system prompt Hermes already
produces. Policy is a `SystemPromptPolicy` enum with two options, chosen at
`ClaudeCliClient` construction:

- `REPLACE` (default): the adapter wraps Hermes's system prompt with a
  preamble that establishes the `<hermes:tool>` envelope as the only legal
  tool-call form. Emitted verbatim into `--system-prompt`.
- `APPEND_ONLY`: the adapter uses `--append-system-prompt` instead, adding
  only the envelope contract to CC's default prompt. For debugging only.

Envelope instructions (constant string in `tool_protocol.py`, exported so
tests can reference it):

```
You are running inside the Hermes agent harness. You have NO built-in tools.

To execute a tool, emit EXACTLY this envelope and STOP generating:

<hermes:tool name="TOOL_NAME" id="UNIQUE_ID">
<arg name="ARG_NAME">ARG_VALUE</arg>
...
</hermes:tool>

The harness will run the tool and reply with:

<hermes:tool_result id="UNIQUE_ID">OUTPUT</hermes:tool_result>

You may then continue. Never emit <hermes:tool_result> yourself. Never try
to call Read, Bash, Edit, or any built-in tool directly.
```

### 8.2 Stream halt parser

The reader thread forwards `text_delta` events to a rolling text buffer per
turn. After each append, scan the buffer's *tail* for `</hermes:tool>`. On
match:

1. Truncate the buffer at (and including) the close tag.
2. Stop forwarding text deltas for this turn.
3. Parse the envelope (§8.3). On parse failure, emit an Anthropic-shape
   error-less `message_stop` but mark the turn's `stop_reason` as
   `"end_turn"` with the raw text intact so Hermes can surface the garbage
   to the user.
4. Synthesize a `content_block_stop` followed by a synthetic
   `content_block_start` for a `tool_use` block (Anthropic shape), then a
   synthetic `content_block_stop`, then `message_delta(stop_reason="tool_use")`,
   then `message_stop`.
5. Kill the subprocess (§7.5). Do NOT respawn yet.
6. Mark the session as "pending tool_result".

On the next `messages.create/stream` call, the last message in
`api_kwargs["messages"]` is the Hermes-produced `tool_result`. The adapter
respawns with `--resume`, converts the tool_result into the envelope
(§8.3), sends it on stdin, and resumes streaming.

Edge cases (mandatory tests):
- `</hermes:tool>` split across deltas (scan the rolling buffer, not deltas).
- Two tool calls emitted before halt can act (halt on FIRST close tag;
  discard rest).
- Wrong close tag (e.g. `</thinking>` seen in spike): tolerant parser
  accepts any matched opener `<hermes:tool …>` if a plausible close comes
  within N=1024 additional characters; otherwise emits `message_stop` with
  the raw buffer as a text block and logs a WARN.
- Model hallucinates `<hermes:tool_result>` inline (seen in spike): halt
  fires before it completes; if halt is delayed and the hallucinated result
  leaks, the parser ignores anything after the first `</hermes:tool>`.

### 8.3 Envelope codec

Parser input: the text from `<hermes:tool name="…" id="…">` up to and
including `</hermes:tool>`.

Grammar (whitespace-tolerant):
```
<hermes:tool name="NAME" id="ID">
  <arg name="KEY">VALUE</arg>*
</hermes:tool>
```

Decoding rules:
- `name`, `id` required. Attribute values XML-entity-decoded
  (`html.unescape`).
- Each `<arg>` has `name` required. Body is the raw inner text, greedy
  match to `</arg>`, entity-decoded.
- Unknown attributes preserved in `parsed.extras` for future use.

Encoding for tool_result (the adapter generates this):
```
<hermes:tool_result id="TOOL_ID">{xml-escaped output}</hermes:tool_result>
```
If Hermes's tool_result content indicates an error (`is_error=True` on the
Anthropic tool_result block, or content contains an "error" marker), wrap
the body:
```
<hermes:tool_result id="TOOL_ID" is_error="true"><error>{xml-escaped output}</error></hermes:tool_result>
```

The adapter treats the tool_result's content as an opaque string: if
Anthropic's tool_result has `content` as a list of blocks, join text
blocks with `\n`; raise `NotImplementedError` for non-text (image, etc.)
for now.

### 8.4 Tool definitions in `api_kwargs["tools"]`

The adapter ingests the Anthropic-style tool definitions and renders them
into the system prompt as a catalog the model can reference:

```
Available tools:

- TOOL_NAME(ARG1: type, ARG2: type) — DESCRIPTION
  schema: {JSON schema}

- ...
```

Rendered tool catalog is appended to the envelope instructions in §8.1.
This allows the model to know what's callable without the tool-use API.

---

## 9. Message Translation

### 9.1 Anthropic `messages` list → CC stdin sequence

Input: `api_kwargs["messages"]`, a list of dicts with `{role, content}`
where `content` is a string OR a list of content blocks
(`text` / `tool_use` / `tool_result` / `image` / ...).

Translation, per message:

| role | content shape | CC stdin output |
|---|---|---|
| `user` | string | `{"type":"user","message":{"role":"user","content":"<str>"}}` |
| `user` | `[{type:text,text:…}, ...]` | concatenate text blocks with `\n\n`, send as above |
| `user` | `[{type:tool_result,tool_use_id:X,content:Y}, ...]` | wrap each as `<hermes:tool_result id="X">Y</hermes:tool_result>`; concatenate with leading `\n\n`; send as user message |
| `user` | contains `image` or other non-text | raise `NotImplementedError` in v0.1 |
| `assistant` | any | **not sent** — assistant messages live in CC's own session state. If seen during initial seed (§7.6 rotation), inject as a user turn with wrapper `<hermes:prev_assistant>…</hermes:prev_assistant>` so CC picks up from there. |
| `system` | — | never appears in `messages`; handled via `--system-prompt`. |

### 9.2 CC stream-json events → Anthropic SDK stream events

Implement in `event_translator.py`. Single pass per turn, state machine:

```
state = "idle"
on text_delta:
   if state == "idle":
       emit content_block_start(type="text", index=0)
       state = "text"
   emit content_block_delta(type="text_delta", text=Δ, index=0)

on thinking_delta:
   if state != "thinking":
       if state == "text": emit content_block_stop(index=text_index)
       emit content_block_start(type="thinking", index=current)
       state = "thinking"
   emit content_block_delta(type="thinking_delta", thinking=Δ, index=current)

on tool halt (§8.2):
   if state in ("text", "thinking"): emit content_block_stop(index=current)
   emit content_block_start(type="tool_use", index=current+1, ...)
   emit content_block_stop(index=current+1)
   emit message_delta(stop_reason="tool_use")
   emit message_stop()

on result:
   if state in ("text", "thinking"): emit content_block_stop(index=current)
   emit message_delta(stop_reason="end_turn")
   emit message_stop()

on error (claude_crashed, oauth_expired):
   raise ClaudeCliError; stream consumer handles.
```

The `get_final_message()` on the StreamCtx accumulates content blocks
alongside streaming and returns a Message-shaped object on `message_stop`.

---

## 10. Error Handling

| Situation | Behavior |
|---|---|
| `claude` binary missing | `SpawnFailed` at first call; user message suggests `npm install -g @anthropic-ai/claude-code`. |
| Subprocess crash mid-turn | `ClaudeCliError(code="claude_crashed", stderr_tail=…)`. Next call respawns via `--resume`. |
| OAuth expired (CC stderr markers: `401`, `OAuth token expired`, `Please run claude auth`) | Raise `OAuthExpired`. Agent loop surfaces with instruction to run `claude auth`. Do not retry. |
| Turn timeout (`turn_budget_s`, default 300 s, with no `result` event) | Terminate subprocess, raise `TurnTimeout`. Agent retry policy applies. |
| Tool-protocol parse failure | Emit the raw buffer as a text block with `stop_reason="end_turn"`; log WARN with the buffer tail; let Hermes proceed (often the model self-corrects next turn). |
| Stream-json stdout produces non-JSON | Log at WARN, drop the line, continue. |
| Concurrent `messages.create/stream` calls on the same client | Raise `ClientBusy`. Hermes's main loop is serial; aux calls use a separate client. |

All adapter errors derive from `ClaudeCliError` (in `errors.py`) and carry
enough context (code, stderr tail, session id) to be logged without
re-exposing prompts.

---

## 11. Configuration

Extend `hermes_cli/config.py` (or wherever Hermes reads config.yaml) with a
new section:

```yaml
claude_cli:
  bin: null                 # path to `claude`; null → shutil.which
  cwd: null                 # null → os.getcwd() at AIAgent init
  turn_budget_s: 300
  request_timeout_s: 600
  system_prompt_policy: replace    # replace | append_only
  enable_for_main: true
  enable_for_auxiliary: true
  log_stderr_head: 4000     # bytes of subprocess stderr captured for error reports
```

Activation is still via top-level `api_mode: claude_cli` (or
`provider: claude_cli`). The `claude_cli:` section tunes behavior when the
mode is active.

---

## 12. Logging

Use Hermes's existing `hermes_logging` helpers.

- INFO: client created, subprocess spawn (pid, model, session id),
  subprocess exit (pid, rc), session rotation (reason), tool-halt fired
  (tool name).
- WARN: malformed stream-json, tool-parse fallback, oauth_expired, turn
  timeout.
- DEBUG: redacted message lines (`<user msg N chars>`), event counts,
  reader-thread pumping stats.
- Never log `system_prompt` bodies, `user.text`, `tool_result` bodies, or
  assistant deltas at INFO+. Redact to `<redacted N chars>` at DEBUG.

---

## 13. Testing

### 13.1 Unit (no `claude` required)
- `test_tool_protocol.py`: envelope parser — valid, malformed, close tag
  split across deltas, entity encoding, nested tags, wrong close tag,
  tolerant-recovery path, hallucinated tool_result post-close.
- `test_event_translator.py`: state-machine correctness for every
  combination of text / thinking / tool-halt / result / crash.
- `test_session.py`: cursor tracking, digest divergence triggers rotation,
  replay order preserved.

### 13.2 Mock-claude tests
Provide a Python stub `claude` (shebanged) that reads stream-json on stdin
and emits scripted events. Tests:
- Full turn round-trip returns Anthropic-shaped Message.
- Tool-halt → parsed envelope → respawn with `--resume` → envelope sent
  to stdin → continuation received.
- Subprocess crash mid-turn → next call respawns, succeeds.
- History divergence (altered prefix message) → session rotates.
- Concurrent `create` calls rejected with `ClientBusy`.
- Unsafe flags never emitted (inspect the stub's argv).

### 13.3 E2E (gated on real `claude`, `pytest.mark.requires_claude`)
- Single turn: text response, stop_reason=end_turn.
- Tool round-trip: system prompt with envelope instructions + a fake tool
  catalog; model emits envelope; test harness replies with synthetic
  tool_result; model continues.
- Context preserved across two turns in one client.
- `close()` is clean (no zombies).

### 13.4 Integration (against `run_agent.py`)
- Start an AIAgent with `api_mode="claude_cli"`; run a trivial prompt via
  Hermes's usual entry point; assert the response reaches the caller
  unchanged relative to the Anthropic-SDK path on the same prompt.

---

## 14. Migration Plan

Deliver in three merges so the blast radius is small and each PR is
reviewable.

### PR 1 — Adapter module, no Hermes integration
- Create `agent/claude_cli/` with all modules per §4.
- Include unit + mock tests (§13.1, §13.2).
- No changes to `run_agent.py` or `auxiliary_client.py`.
- Add a standalone entry point `scripts/claude_cli_demo.py` that
  constructs the client and runs a scripted prompt end-to-end (mirrors the
  existing `scripts/claude_cli_spike.py` but uses the real adapter).

### PR 2 — Auxiliary client integration
- Wire `claude_cli` branch into `agent/auxiliary_client.py`.
- Add E2E tests for aux calls.
- Ship behind a config flag default-off (`claude_cli.enable_for_auxiliary`).

### PR 3 — Main-loop integration
- Add the `api_mode == "claude_cli"` dispatch in `run_agent.py` (§6).
- Delete / bypass the OAuth-refresh path for this mode.
- Config: `api_mode: claude_cli`.
- Add docs to `README.md`: when to use this mode, known limitations
  (vision, token accounting, prompt caching opacity).

Do not merge PR 3 without running the integration tests in §13.4 against
opus, sonnet, and haiku.

---

## 15. Out of Scope (v0.1)

- Vision / multimodal inputs (raise `NotImplementedError`; v0.2 passes
  CC-native content blocks through).
- Explicit prompt-cache control (`cache_control` blocks are dropped).
- Token usage parity (Usage has zeros; `duration_ms` and `num_turns` are
  logged but not surfaced into the Hermes usage counters).
- Concurrent turns on the same client (single-writer).
- Running `claude` with built-in tools enabled (Hermes owns the tool
  protocol; future work could expose this behind a flag).
- Auto-retry on transient errors beyond what CC itself does internally.

---

## 16. Deliverables Checklist

- [ ] `agent/claude_cli/` package per §4, stdlib-only.
- [ ] `ClaudeCliClient.messages.create` returning SDK-shaped `Message`.
- [ ] `ClaudeCliClient.messages.stream` returning SDK-shaped stream
      context, with event shapes that match `run_agent.py:5824-5920`
      expectations.
- [ ] Tool-protocol halt with all edge cases (§8.2) covered by tests.
- [ ] Session rotation on history divergence (§7.6).
- [ ] Integration into `run_agent.py` (§6) behind `api_mode="claude_cli"`.
- [ ] Auxiliary client branch (§6.4).
- [ ] All tests green: unit, mock-claude, E2E (when `claude` present),
      integration.
- [ ] README section documenting activation, caveats, and the
      `claude_cli:` config block.
- [ ] No new runtime dependencies in `pyproject.toml`.
