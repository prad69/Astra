"""astra.web

Concept: the studio. cli.py is the front door for a terminal; this is the
front door for a browser. Same harness, same tools, same policy gate --
the only new idea is that the event stream now has to reach a reader who
is not attached to stdout, and that the thing being built has to be
visible while it is being built.

Three pieces make that work, and nothing else here is new:
  - A Project owns one Harness, one workdir, and one append-only event
    log. The Harness's on_event seam is where the log gets filled, so the
    browser sees exactly what the terminal printer sees.
  - Subscribers are queues. publish() appends to the log and fans out to
    every open queue under one lock, so a reader that connects mid-run
    gets the backlog and the live tail with no gap between them and no
    duplicate at the seam.
  - The preview serves the project directory itself. A "product" built by
    the agent is files on disk; showing it is a static file server pointed
    at the same jail the tools write into, which is why the path check
    here mirrors the one in tools.resolve rather than trusting the URL.

Design rules this file embodies:
  - Localhost only, by default. The agent writes files and runs commands;
    the server that fronts it is not an internet-facing service, and
    binding it to 0.0.0.0 is a decision the caller has to make out loud.
  - Approval is an event, not a prompt. safe mode in a browser works the
    same way it does in a terminal -- the run thread blocks on an answer
    -- except the question goes out over the stream and the answer comes
    back as a POST. A question nobody answers times out into a refusal,
    because the default approver refuses.
  - Stopping repairs the transcript. Raising out of before_tool ends the
    run promptly but leaves tool calls unanswered, and the API rejects
    that shape on the next turn. session.repair fills them in with the
    same interrupted notice a crash would have produced.
"""

import json
import mimetypes
import os
import posixpath
import queue
import re
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import session
from .harness import Harness
from .security import Policy

UI_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ui.html")

# Names that become directories under the projects root, so the same
# characters a shell would have to quote are simply not allowed.
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

HEARTBEAT_SECONDS = 15
APPROVAL_TIMEOUT = 300
RESULT_CLIP = 4000
MAX_EVENTS = 5000

HIDDEN_DIRS = {".git", ".astra", "node_modules", "__pycache__", ".venv"}


class RunStopped(Exception):
    """Raised out of the policy gate to end a run the user asked to stop."""


class StoppablePolicy(Policy):
    """A Policy that refuses to gate anything once the run is stopping.

    before_tool is the only callback the loop invokes between tool calls,
    which makes it the one place a stop can take effect without polling.
    _run_one calls it outside its try, so raising here leaves the loop
    rather than turning into an "ERROR: ..." the model would read and
    work around.
    """

    def __init__(self, mode, approver=None, stopped=None):
        super().__init__(mode, approver)
        self.stopped = stopped or (lambda: False)

    def check(self, call):
        if self.stopped():
            raise RunStopped("stopped by the user")
        return super().check(call)


class Project:
    """One workdir, one agent, one event log, and the readers watching it."""

    def __init__(self, studio, name):
        self.name = name
        self.workdir = os.path.join(studio.root, name)
        os.makedirs(self.workdir, exist_ok=True)

        self.lock = threading.Lock()
        self.events = []
        self.seq = 0
        self.subscribers = set()
        self.pending = {}

        self.busy = False
        self.stopping = False

        self.harness = Harness(
            workdir=self.workdir,
            model=studio.model,
            policy=StoppablePolicy(studio.mode, approver=self._approve,
                                   stopped=lambda: self.stopping),
            on_event=self._on_harness_event,
            max_turns=studio.max_turns,
        )
        self._adopt_history()

    # -- the event log ----------------------------------------------------

    def publish(self, kind, **data):
        """Append one event to the log and hand it to every open reader."""
        with self.lock:
            self.seq += 1
            event = {"seq": self.seq, "kind": kind, "t": time.time(), **data}
            self.events.append(event)
            if len(self.events) > MAX_EVENTS:
                del self.events[:len(self.events) - MAX_EVENTS]
            for q in self.subscribers:
                q.put(event)
        return event

    def subscribe(self):
        """Register a reader and hand back (queue, backlog) atomically.

        Taking the snapshot inside the same lock that registers the queue
        is what makes the seam gapless: an event published from another
        thread either lands in the backlog or in the queue, never in
        neither and never in both.
        """
        q = queue.Queue()
        with self.lock:
            self.subscribers.add(q)
            return q, list(self.events)

    def unsubscribe(self, q):
        with self.lock:
            self.subscribers.discard(q)

    def _adopt_history(self):
        """Resume the newest session and replay it into the event log.

        Reopening a project should look like reopening a conversation, so
        the messages that survived on disk become events the browser can
        render -- marked as replay, because they already happened.
        """
        if not self.harness.resume():
            return
        for message in self.harness.messages:
            role = message.get("role")
            if role == "user":
                self.publish("user", text=message.get("text", ""), replay=True)
            elif role == "assistant" and message.get("text"):
                self.publish("assistant", text=message["text"], replay=True)
            elif role == "tool":
                self.publish("tool_end", name=message.get("name", "tool"),
                             text=_clip(message.get("text", "")), replay=True)
        self.publish("resumed", path=os.path.basename(self.harness.session_path or ""),
                     count=len(self.harness.messages))

    # -- the harness seams ------------------------------------------------

    def _on_harness_event(self, kind, payload):
        """Translate harness events into stream events."""
        if kind == "assistant":
            if payload.get("text"):
                self.publish("assistant", text=payload["text"])
        elif kind == "tool_start":
            self.publish("tool_start", name=payload["name"],
                         args=payload.get("args", {}))
        elif kind == "tool_end":
            self.publish("tool_end", name=payload.get("name", "tool"),
                         text=_clip(payload.get("text", "")))

    def _approve(self, call, reason):
        """Ask the browser, and block this run thread until it answers."""
        gate = threading.Event()
        answer = {"approved": False}
        approval_id = uuid.uuid4().hex[:8]

        with self.lock:
            self.pending[approval_id] = (gate, answer)

        self.publish("approval", id=approval_id, name=call["name"],
                     args=call.get("args", {}))
        answered = gate.wait(APPROVAL_TIMEOUT)

        with self.lock:
            self.pending.pop(approval_id, None)

        approved = bool(answered and answer["approved"])
        self.publish("approval_resolved", id=approval_id, approved=approved,
                     timed_out=not answered)
        return approved

    def resolve_approval(self, approval_id, approved):
        """Answer a pending approval. False when there is no such question."""
        with self.lock:
            entry = self.pending.get(approval_id)
        if entry is None:
            return False
        gate, answer = entry
        answer["approved"] = bool(approved)
        gate.set()
        return True

    # -- running ----------------------------------------------------------

    def start(self, prompt):
        """Begin a run in its own thread. False when one is already going."""
        with self.lock:
            if self.busy:
                return False
            self.busy = True
            self.stopping = False

        self.publish("user", text=prompt)
        self.publish("status", busy=True)
        threading.Thread(target=self._run, args=(prompt,), daemon=True).start()
        return True

    def stop(self):
        """Ask the current run to end at its next tool call."""
        if not self.busy:
            return False
        self.stopping = True
        # A run parked on an approval would otherwise wait out the full
        # timeout before reaching the gate that notices the stop.
        with self.lock:
            pending = list(self.pending.items())
        for approval_id, (gate, answer) in pending:
            answer["approved"] = False
            gate.set()
        self.publish("stopping")
        return True

    def _run(self, prompt):
        try:
            answer = self.harness.run(prompt)
            self.publish("done", text=answer or "")
        except RunStopped:
            # The loop left between a tool call and its result, so the
            # transcript has tool_use blocks with nothing answering them.
            # Repair it here or the next run sends a turn the API refuses.
            session.repair(self.harness.messages)
            self.harness.flush()
            self.publish("stopped")
        except Exception as e:
            self.publish("error", text=f"{type(e).__name__}: {e}")
        finally:
            with self.lock:
                self.busy = False
                self.stopping = False
            self.publish("status", busy=False)

    # -- files ------------------------------------------------------------

    def resolve(self, path):
        """Resolve a path inside the project, or raise if it escapes."""
        root = os.path.realpath(self.workdir)
        full = os.path.realpath(os.path.join(root, path))
        if full != root and not full.startswith(root + os.sep):
            raise PermissionError(f"{path!r} escapes the project")
        return full

    def files(self):
        """Every visible file in the project, as sorted relative paths."""
        root = os.path.realpath(self.workdir)
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in HIDDEN_DIRS]
            for name in filenames:
                if name.startswith("."):
                    continue
                found.append(os.path.relpath(os.path.join(dirpath, name), root))
        found.sort()
        return found[:1000]


class Studio:
    """The projects on disk and the live agent behind each one."""

    def __init__(self, root, model=None, mode="yolo", max_turns=120):
        self.root = os.path.realpath(root)
        os.makedirs(self.root, exist_ok=True)
        self.model = model
        self.mode = mode
        self.max_turns = max_turns
        self.lock = threading.Lock()
        self.projects = {}

    def names(self):
        """Every project directory, newest activity first."""
        entries = [n for n in os.listdir(self.root)
                   if NAME_RE.fullmatch(n)
                   and os.path.isdir(os.path.join(self.root, n))]
        entries.sort(key=lambda n: os.path.getmtime(os.path.join(self.root, n)),
                     reverse=True)
        return entries

    def project(self, name):
        """Get or build the live Project for a name, creating its directory."""
        if not NAME_RE.fullmatch(name or ""):
            raise ValueError(f"invalid project name {name!r}")
        with self.lock:
            if name not in self.projects:
                self.projects[name] = Project(self, name)
            return self.projects[name]


def _clip(text, limit=RESULT_CLIP):
    """Shorten a tool result to something a browser can hold comfortably."""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[{len(text) - limit} more characters]..."


class _Handler(BaseHTTPRequestHandler):
    """The routes. Every one of them is a thin shell over Studio/Project."""

    protocol_version = "HTTP/1.1"
    studio = None
    server_version = "astra-studio"

    def log_message(self, fmt, *args):
        pass  # the event stream is the log worth reading

    # -- plumbing ---------------------------------------------------------

    def _send(self, status, body, content_type="application/octet-stream",
              extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status, message):
        self._json({"error": message}, status)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except ValueError:
            return {}

    def _project(self, name):
        """Resolve a project, answering the request itself on failure."""
        try:
            return self.studio.project(name)
        except ValueError as e:
            self._error(400, str(e))
            return None

    # -- GET --------------------------------------------------------------

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            return self._serve_ui()
        if path == "/api/projects":
            return self._json({"projects": self.studio.names()})
        if path == "/api/events":
            return self._serve_events(query.get("project", [""])[0])
        if path == "/api/files":
            return self._serve_files(query.get("project", [""])[0])
        if path == "/api/file":
            return self._serve_file(query.get("project", [""])[0],
                                    query.get("path", [""])[0])
        if path.startswith("/preview/"):
            return self._serve_preview(path[len("/preview/"):])
        return self._error(404, "no such route")

    def _serve_ui(self):
        try:
            with open(UI_PATH, "rb") as f:
                body = f.read()
        except OSError as e:
            return self._error(500, f"cannot read ui.html: {e}")
        self._send(200, body, "text/html; charset=utf-8",
                   {"Cache-Control": "no-store"})

    def _serve_events(self, name):
        """Stream the project's events as SSE until the reader goes away.

        Close-delimited rather than chunked: the body has no length and
        never ends on its own, so the connection is the frame. A comment
        heartbeat keeps idle proxies from deciding otherwise.
        """
        project = self._project(name)
        if project is None:
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        q, backlog = project.subscribe()
        try:
            for event in backlog:
                self._write_event(event)
            # seq 0 marks this as a synthesised catch-up rather than a
            # logged event, so a reconnecting reader still learns whether
            # a run is in flight without the log growing a status per read.
            self._write_event({"seq": 0, "kind": "status", "busy": project.busy})
            while True:
                try:
                    event = q.get(timeout=HEARTBEAT_SECONDS)
                except queue.Empty:
                    self._write_raw(": ping\n\n")
                    continue
                self._write_event(event)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            project.unsubscribe(q)

    def _write_event(self, event):
        self._write_raw(f"data: {json.dumps(event)}\n\n")

    def _write_raw(self, text):
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    def _serve_files(self, name):
        project = self._project(name)
        if project is None:
            return
        self._json({"files": project.files()})

    def _serve_file(self, name, path):
        project = self._project(name)
        if project is None:
            return
        try:
            full = project.resolve(path)
            with open(full, "r", errors="replace") as f:
                text = f.read(400_000)
        except (PermissionError, OSError) as e:
            return self._error(404, str(e))
        self._json({"path": path, "text": text})

    def _serve_preview(self, rest):
        """Serve the project directory as a site, so a build is viewable."""
        parts = rest.split("/", 1)
        name = urllib.parse.unquote(parts[0])
        relative = urllib.parse.unquote(parts[1]) if len(parts) > 1 else ""

        project = self._project(name)
        if project is None:
            return

        relative = posixpath.normpath(relative or "index.html").lstrip("/")
        try:
            full = project.resolve(relative)
        except PermissionError as e:
            return self._error(403, str(e))

        if os.path.isdir(full):
            full = os.path.join(full, "index.html")
        if not os.path.isfile(full):
            return self._send(
                404,
                b"<!doctype html><meta charset=utf-8>"
                b"<style>body{font:14px ui-monospace,monospace;background:#14110d;"
                b"color:#9a8f7d;padding:2rem}</style>"
                b"<p>Nothing to preview yet. Ask the agent to write an "
                b"<code>index.html</code>.</p>",
                "text/html; charset=utf-8")

        kind = mimetypes.guess_type(full)[0] or "application/octet-stream"
        try:
            with open(full, "rb") as f:
                body = f.read()
        except OSError as e:
            return self._error(500, str(e))
        self._send(200, body, kind, {"Cache-Control": "no-store"})

    # -- POST -------------------------------------------------------------

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        payload = self._read_json()

        if path == "/api/project":
            return self._create_project(payload)
        if path == "/api/run":
            return self._run(payload)
        if path == "/api/stop":
            return self._stop(payload)
        if path == "/api/approve":
            return self._approve(payload)
        return self._error(404, "no such route")

    def _create_project(self, payload):
        name = (payload.get("name") or "").strip()
        if not NAME_RE.fullmatch(name):
            return self._error(400, "names may use letters, digits, dot, dash, "
                                    "underscore, and must start with a letter "
                                    "or digit")
        self.studio.project(name)
        self._json({"name": name, "projects": self.studio.names()})

    def _run(self, payload):
        project = self._project(payload.get("project", ""))
        if project is None:
            return
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return self._error(400, "empty prompt")
        if not project.start(prompt):
            return self._error(409, "a run is already in progress")
        self._json({"ok": True})

    def _stop(self, payload):
        project = self._project(payload.get("project", ""))
        if project is None:
            return
        self._json({"ok": project.stop()})

    def _approve(self, payload):
        project = self._project(payload.get("project", ""))
        if project is None:
            return
        found = project.resolve_approval(payload.get("id", ""),
                                         bool(payload.get("approved")))
        self._json({"ok": found})


def serve(root="./projects", host="127.0.0.1", port=7777, model=None,
          mode="yolo", max_turns=120):
    """Run the studio server until interrupted. Returns an exit status."""
    studio = Studio(root, model=model, mode=mode, max_turns=max_turns)

    handler = type("_BoundHandler", (_Handler,), {"studio": studio})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True

    print(f"astra studio  http://{host}:{port}")
    print(f"  projects  {studio.root}")
    print(f"  model     {studio.model or 'default'}")
    print(f"  mode      {mode}")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("  note      bound beyond localhost; this server can write files "
              "and run commands")
    print("Ctrl-C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.shutdown()
        server.server_close()
    return 0
