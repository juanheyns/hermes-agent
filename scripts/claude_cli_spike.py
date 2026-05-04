#!/usr/bin/env python3
"""
Spike: prototype a persistent `claude -p` subprocess as a model backend.

Tests five hypotheses required to commit to the claude_cli adapter:

  1. A single long-running `claude -p --input-format stream-json
     --output-format stream-json` process can accept multiple user messages
     in sequence over stdin (no restart per turn).
  2. Cold start (launch -> first event) vs warm latency (subsequent message
     -> first event) — we want warm << cold to justify persistence.
  3. Context is preserved across messages within the process without us
     resending history (CC owns conversation state).
  4. A text-based tool-call protocol driven entirely by `--system-prompt`
     works when built-in tools are disabled via `--tools ""`.
  5. Session rotation: we can kill the process and spawn a new one with
     `--resume <session-id>` to reattach to the same conversation.

Run:
    python scripts/claude_cli_spike.py [sonnet|opus|haiku]

The script is intentionally self-contained (stdlib only) and verbose so
the stream-json event shapes are visible for the adapter work.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid


SYSTEM_PROMPT = """\
You are operating inside a custom agent harness called Hermes. You have NO built-in tools.

When you need to execute a shell command, emit EXACTLY this envelope and then stop:

<hermes:tool name="bash" id="UNIQUE_ID">
<arg name="command">THE COMMAND</arg>
</hermes:tool>

The harness will execute the command and reply with:

<hermes:tool_result id="UNIQUE_ID">OUTPUT</hermes:tool_result>

After receiving the result you may continue. Do NOT attempt to call Read, Bash,
Edit, or any built-in tool — only emit <hermes:tool> envelopes. If you have
nothing to do, answer in plain text.
"""

# Tiny safelist for the tool round-trip test. We do NOT blindly execute
# whatever the model emits; if it asks for something outside this set, the
# spike fabricates output so we can still verify the protocol round-trip
# without running arbitrary commands.
SAFE_COMMANDS = ("ls", "pwd", "date", "echo", "whoami", "uname")


def launch_claude(model: str, session_id: str, resume: bool = False) -> subprocess.Popen:
    cmd = [
        "claude", "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--verbose",  # stream-json output requires verbose in -p mode
        "--tools", "",
        "--system-prompt", SYSTEM_PROMPT,
        "--model", model,
        "--replay-user-messages",
    ]
    if resume:
        cmd += ["--resume", session_id]
    else:
        cmd += ["--session-id", session_id]

    print(f"[spike] launching: claude -p --model {model} "
          f"--{'resume' if resume else 'session-id'} {session_id}")
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def start_readers(proc: subprocess.Popen):
    q: "queue.Queue[tuple[str, str, float]]" = queue.Queue()
    stop = threading.Event()

    def _pump(stream, tag):
        try:
            for line in stream:
                if stop.is_set():
                    break
                q.put((tag, line.rstrip("\n"), time.monotonic()))
        finally:
            q.put((f"{tag}_eof", "", time.monotonic()))

    threading.Thread(target=_pump, args=(proc.stdout, "out"), daemon=True).start()
    threading.Thread(target=_pump, args=(proc.stderr, "err"), daemon=True).start()
    return q, stop


def send_user(proc: subprocess.Popen, text: str, session_id: str):
    # Best-guess stream-json input shape (CC SDK convention).
    payload = {
        "type": "user",
        "message": {"role": "user", "content": text},
        "session_id": session_id,
    }
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()


def collect_turn(q, t_send: float, label: str, timeout_s: float = 90.0, dump_events: bool = False):
    """Drain events until we see a terminal 'result' event for the turn.

    Returns (text, first_event_latency, all_events).
    """
    t_first = None
    accumulated = []
    events = []
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        try:
            tag, line, t = q.get(timeout=1.0)
        except queue.Empty:
            continue

        if tag == "err":
            if line.strip():
                print(f"  [stderr] {line}")
            continue
        if tag.endswith("_eof"):
            print(f"  [{tag}] stream closed")
            if tag == "out_eof":
                return "".join(accumulated), t_first, events
            continue

        if not line.strip():
            continue

        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            print(f"  [non-json] {line[:160]}")
            continue

        if t_first is None:
            t_first = t - t_send
            print(f"  [{label}] first event +{t_first:.2f}s  type={evt.get('type')}")

        events.append(evt)
        if dump_events:
            print(f"  [{label}] EVT {json.dumps(evt)[:200]}")

        etype = evt.get("type")

        # Accumulate assistant text from whichever shape CC emits.
        if etype == "assistant":
            msg = evt.get("message", {}) or {}
            for block in msg.get("content", []) or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    accumulated.append(block.get("text", ""))
        elif etype == "stream_event":
            inner = evt.get("event", {}) or {}
            if inner.get("type") == "content_block_delta":
                delta = inner.get("delta", {}) or {}
                if delta.get("type") == "text_delta":
                    accumulated.append(delta.get("text", ""))
        elif etype == "content_block_delta":
            delta = evt.get("delta", {}) or {}
            if delta.get("type") == "text_delta":
                accumulated.append(delta.get("text", ""))

        # Terminal event for a single turn in stream-json output.
        if etype == "result":
            print(f"  [{label}] result subtype={evt.get('subtype')} "
                  f"duration_ms={evt.get('duration_ms')} "
                  f"num_turns={evt.get('num_turns')}")
            return "".join(accumulated), t_first, events

    print(f"  [{label}] TIMEOUT after {timeout_s}s")
    return "".join(accumulated), t_first, events


def parse_tool_call(text: str):
    m = re.search(
        r'<hermes:tool\s+name="([^"]+)"\s+id="([^"]+)"\s*>(.*?)</hermes:tool>',
        text, re.DOTALL,
    )
    if not m:
        return None
    name, tid, body = m.group(1), m.group(2), m.group(3)
    arg = re.search(r'<arg\s+name="command"\s*>(.*?)</arg>', body, re.DOTALL)
    cmd = arg.group(1).strip() if arg else ""
    return {"name": name, "id": tid, "command": cmd}


def safe_run(cmd: str) -> str:
    head = cmd.strip().split()[0] if cmd.strip() else ""
    if head not in SAFE_COMMANDS:
        return f"[spike-sandboxed] refused to run '{head}'. Pretend output: three files."
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return (r.stdout + r.stderr)[:1500]
    except subprocess.TimeoutExpired:
        return "[spike] command timed out"


def shutdown(proc: subprocess.Popen, stop: threading.Event):
    stop.set()
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "sonnet"
    session_id = str(uuid.uuid4())
    print(f"[spike] model={model} session_id={session_id}\n")

    # ---- Phase A: long-running process, multiple turns ----
    t_launch = time.monotonic()
    proc = launch_claude(model, session_id, resume=False)
    q, stop = start_readers(proc)

    try:
        # Turn 1 — context anchor
        print("[spike] turn 1: context anchor")
        t_send = time.monotonic()
        send_user(proc, "My favorite color is cerulean. Acknowledge briefly.", session_id)
        text1, first1, _ = collect_turn(q, t_send, "turn1", timeout_s=60)
        cold_from_launch = time.monotonic() - t_launch
        print(f"  cold launch->first-event: {first1 and (t_send - t_launch + first1):.2f}s")
        print(f"  turn1 first-event latency: {first1:.2f}s")
        print(f"  turn1 text: {text1[:300]!r}\n")

        # Turn 2 — context recall (no history resent)
        print("[spike] turn 2: context recall")
        t_send = time.monotonic()
        send_user(proc, "What's my favorite color? One word.", session_id)
        text2, first2, _ = collect_turn(q, t_send, "turn2", timeout_s=60)
        print(f"  turn2 warm first-event latency: {first2:.2f}s")
        print(f"  turn2 text: {text2[:300]!r}")
        recall_ok = "cerulean" in (text2 or "").lower()
        print(f"  context preserved across turns: {recall_ok}\n")

        # Turn 3 — tool protocol
        print("[spike] turn 3: tool-call text protocol")
        t_send = time.monotonic()
        send_user(
            proc,
            "Please tell me the current working directory. Use the <hermes:tool> envelope as instructed in your system prompt.",
            session_id,
        )
        text3, first3, _ = collect_turn(q, t_send, "turn3", timeout_s=60)
        print(f"  turn3 warm first-event latency: {first3:.2f}s")
        print(f"  turn3 raw text: {text3[:600]!r}")

        call = parse_tool_call(text3)
        if call:
            print(f"  PARSED tool call: {call}")
            result = safe_run(call["command"])
            print(f"  tool output: {result[:200]!r}")

            # Turn 4 — feed result back
            print("\n[spike] turn 4: feeding tool result back")
            t_send = time.monotonic()
            send_user(
                proc,
                f'<hermes:tool_result id="{call["id"]}">{result}</hermes:tool_result>',
                session_id,
            )
            text4, first4, _ = collect_turn(q, t_send, "turn4", timeout_s=60)
            print(f"  turn4 warm first-event latency: {first4:.2f}s")
            print(f"  turn4 text: {text4[:400]!r}")
            protocol_ok = True
        else:
            print("  !! no <hermes:tool> envelope detected — protocol needs tuning")
            protocol_ok = False

    finally:
        shutdown(proc, stop)

    # ---- Phase B: resume the same session from a fresh subprocess ----
    print("\n[spike] PHASE B: --resume in a new subprocess")
    t_launch2 = time.monotonic()
    proc2 = launch_claude(model, session_id, resume=True)
    q2, stop2 = start_readers(proc2)
    try:
        t_send = time.monotonic()
        send_user(proc2, "What was my favorite color again?", session_id)
        textR, firstR, _ = collect_turn(q2, t_send, "resume", timeout_s=60)
        resume_cold = (t_send - t_launch2) + (firstR or 0)
        print(f"  resume cold launch->first-event: {resume_cold:.2f}s")
        print(f"  resume text: {textR[:300]!r}")
        resume_ok = "cerulean" in (textR or "").lower()
        print(f"  resumed-session context preserved: {resume_ok}")
    finally:
        shutdown(proc2, stop2)

    # ---- Summary ----
    print("\n[spike] SUMMARY")
    print(f"  model                     : {model}")
    print(f"  turn1 first-event (cold)  : {first1:.2f}s" if first1 else "  turn1 first-event (cold)  : n/a")
    print(f"  turn2 first-event (warm)  : {first2:.2f}s" if first2 else "  turn2 first-event (warm)  : n/a")
    print(f"  turn3 first-event (warm)  : {first3:.2f}s" if first3 else "  turn3 first-event (warm)  : n/a")
    print(f"  context preserved         : {recall_ok}")
    print(f"  tool protocol round-trip  : {protocol_ok}")
    print(f"  --resume reattach works   : {resume_ok}")
    print(f"  session_id                : {session_id}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[spike] interrupted")
        sys.exit(130)
