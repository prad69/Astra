"""astra.tools

Concept: turning plain functions into tools the loop can call, and giving
the agent a small, safe filesystem/shell surface to act with. Two ideas
carry the file: (1) a tool is just a function plus a JSON-schema spec
derived from its signature -- the decorator reads argument names so no one
hand-writes a schema twice; (2) every path any tool touches is resolved
against the working directory first, so "escape the sandbox via a path"
is a single choke point (resolve) rather than six places to get wrong.

Design rules this file embodies:
  - All schema parameters are string-typed. Claude's tool use is
    happiest with flat string args; a tool that wants an int just parses
    the string itself (see bash's timeout). One type to reason about beats
    six edge cases in schema generation.
  - edit_file's uniqueness rule: a snippet must match exactly once before
    it is touched. Zero matches usually means the caller misquoted the
    file; more than one means the edit is ambiguous about which occurrence
    was meant. Both are reported back as tool text, not exceptions, so the
    model gets a chance to look again and retry.
  - read_file tolerates a torn last line (no trailing newline) the same
    way it tolerates any other line -- readlines() splits on newlines and
    a missing final one just yields a shorter last entry, no special case.
  - fetch_url has its own choke point, _url_block_reason, for the same
    reason resolve exists: reaching the network is an escape hatch out of
    the sandbox, so every URL -- the one asked for and every redirect the
    server proposes -- is checked in exactly one place. The check resolves
    the hostname and refuses private, loopback, and link-local addresses,
    because "https://..." says nothing about where the name points and
    169.254.169.254 is a cloud credential endpoint, not a web page.
"""

import fnmatch
import html
import inspect
import ipaddress
import os
import re
import socket
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv"}

FETCH_SCHEMES = {"http", "https"}
FETCH_TIMEOUT = 20
FETCH_MAX_BYTES = 400_000
FETCH_MAX_CHARS = 6000
FETCH_USER_AGENT = "astra-agent/0.1 (+workshop demo)"


@dataclass
class Tool:
    """A callable the loop can dispatch: a name, a provider-ready spec, and the function itself."""
    name: str
    spec: dict
    run: Callable


def tool(description, **params):
    """Decorator: wrap a plain function as a Tool with an auto-built schema.

    Reads the function's parameter names from its signature; a parameter
    with a default is optional, one without is required. **params supplies
    the human-readable description for each parameter by name.
    """
    def decorator(fn):
        sig = inspect.signature(fn)
        required = [p.name for p in sig.parameters.values()
                    if p.default is inspect.Parameter.empty]
        properties = {p: {"type": "string", "description": params.get(p, "")}
                      for p in sig.parameters}
        spec = {"schema": {
            "name": fn.__name__,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }}
        return Tool(name=fn.__name__, spec=spec, run=fn)
    return decorator


def _url_block_reason(url):
    """Return why this URL must not be fetched, or None if it is allowed.

    Two gates. The scheme gate rejects anything that is not http(s), which
    is what keeps file:// and ftp:// from turning a "web" tool into an
    arbitrary reader. The address gate resolves the hostname and rejects
    any answer that is not a public address -- localhost, 10./192.168./
    172.16., and the 169.254 link-local range that holds cloud metadata
    services. A name is checked, not trusted: plenty of public hostnames
    resolve to 127.0.0.1 on purpose.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError as e:
        return f"malformed URL ({e})"

    if parsed.scheme not in FETCH_SCHEMES:
        return f"scheme {parsed.scheme or '(none)'!r} is not http or https"
    if not parsed.hostname:
        return "URL has no hostname"

    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or
                                   (443 if parsed.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        return f"hostname {parsed.hostname!r} does not resolve ({e.strerror or e})"

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return f"hostname {parsed.hostname!r} resolves to non-public address {ip}"
    return None


class _GuardedRedirect(urllib.request.HTTPRedirectHandler):
    """Re-run the URL gate on every redirect the server proposes.

    urllib follows redirects on its own, so a URL that passes the gate can
    still land somewhere that never would -- a public host answering 302
    with http://169.254.169.254/. Checking only the URL the caller typed
    would gate the front door and leave the hallway open.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        reason = _url_block_reason(newurl)
        if reason is not None:
            raise urllib.error.URLError(f"blocked redirect to {newurl}: {reason}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _extract_text(markup):
    """Reduce an HTML document to its title and readable text.

    Deliberately regex-based rather than a parser: stdlib-only is a project
    rule, and the consumer is a model that wants prose, not a faithful DOM.
    script/style/noscript go first (their contents are text to a regex but
    noise to a reader), then block-level tags become newlines so paragraph
    breaks survive, then every remaining tag is dropped.
    """
    title_match = re.search(r"<title[^>]*>(.*?)</title>", markup,
                            re.IGNORECASE | re.DOTALL)
    title = html.unescape(title_match.group(1)).strip() if title_match else ""

    text = re.sub(r"<!--.*?-->", " ", markup, flags=re.DOTALL)
    text = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", text,
                  flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<br\s*/?>|</(p|div|li|tr|h[1-6]|section|article)>", "\n",
                  text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return title, text.strip()


@tool("Scrape a public web page over http(s) and summarize the content. "
      "Returns an error message if the URL is invalid or has security restrictions (non-public address).",
      url="the absolute http(s) URL to fetch and summarize")
def fetch_url(url):
    """Fetch url and return its title and text, or an "ERROR: ..." string.

    Every failure path returns text rather than raising: an unreachable
    host, a blocked address, a PDF where a page was expected, and a 404
    are all things the model can react to (fix the URL, try another
    source, tell the user), which is only true if it sees them.
    """
    reason = _url_block_reason(url)
    if reason is not None:
        return f"ERROR: refused to fetch {url!r}: {reason}"

    request = urllib.request.Request(url, headers={
        "User-Agent": FETCH_USER_AGENT,
        "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1",
        # urllib does not decompress for us, so ask the server not to.
        "Accept-Encoding": "identity",
    })
    opener = urllib.request.build_opener(_GuardedRedirect)

    try:
        with opener.open(request, timeout=FETCH_TIMEOUT) as resp:
            final_url = resp.geturl()
            content_type = resp.headers.get_content_type()
            if not content_type.startswith("text/"):
                return f"ERROR: {final_url} returned {content_type}, not text"
            charset = resp.headers.get_content_charset() or "utf-8"
            raw = resp.read(FETCH_MAX_BYTES)
    except urllib.error.HTTPError as e:
        return f"ERROR: {url} returned HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"ERROR: could not fetch {url}: {e.reason}"
    except (TimeoutError, socket.timeout):
        return f"ERROR: {url} timed out after {FETCH_TIMEOUT}s"
    except OSError as e:
        return f"ERROR: could not fetch {url}: {e}"

    body = raw.decode(charset, errors="replace")
    title, text = _extract_text(body) if content_type == "text/html" else ("", body)
    if len(text) > FETCH_MAX_CHARS:
        text = text[:FETCH_MAX_CHARS] + "\n...[truncated]..."
    if not text:
        return f"Fetched {final_url} ({content_type}) but it had no readable text"
    return f"Title: {title}\nURL: {final_url}\n\n{text}"


def core_tools(workdir):
    """Build the tool set: six filesystem/shell tools sandboxed to the real
    path of workdir, plus fetch_url, which needs no workdir because its
    sandbox is an address gate rather than a path prefix."""
    root = os.path.realpath(workdir)

    def resolve(path):
        """Resolve path against root and refuse anything that escapes it."""
        full = os.path.realpath(os.path.join(root, path))
        if full != root and not full.startswith(root + os.sep):
            raise PermissionError(f"{path!r} escapes the working directory")
        return full

    @tool("Read a file's contents with line numbers", path="the file path")
    def read_file(path):
        with open(resolve(path), "r", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        shown = lines[:4000]
        numbered = "".join(f"{i + 1}\t{line}" for i, line in enumerate(shown))
        if total > 4000:
            if not numbered.endswith("\n"):
                numbered += "\n"
            numbered += f"... truncated, {total} lines total\n"
        return numbered

    @tool("Write content to a file, creating parent directories as needed",
          path="the file path", content="the text to write")
    def write_file(path, content):
        full = resolve(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {path}"

    @tool("Replace one exact snippet in a file", path="the file path",
          old="the exact text to find", new="the text to replace it with")
    def edit_file(path, old, new):
        full = resolve(path)
        with open(full, "r") as f:
            text = f.read()
        count = text.count(old)
        if count == 0:
            return "ERROR: snippet not found — read the file and copy it exactly"
        if count > 1:
            return f"ERROR: snippet appears {count} times — include more context to make it unique"
        with open(full, "w") as f:
            f.write(text.replace(old, new, 1))
        return f"Edited {path}"

    @tool("Run a shell command in the working directory", command="the shell command",
          timeout="seconds to allow before giving up, default 120")
    def bash(command, timeout="120"):
        t = int(timeout)
        try:
            proc = subprocess.run(command, shell=True, cwd=root,
                                   capture_output=True, text=True, timeout=t)
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {t}s"
        output = proc.stdout + proc.stderr
        if len(output) > 12000:
            output = output[:6000] + "\n...[truncated]...\n" + output[-6000:]
        return output if output else f"(exit {proc.returncode}, no output)"

    @tool("List files matching a glob pattern", pattern="glob pattern, e.g. **/*.py")
    def list_files(pattern="**/*"):
        # fnmatch treats "*" as matching "/" too, but it still requires a
        # literal "/" to appear where the pattern has one -- so "**/*" only
        # matches paths with a slash in them (nested files). A caller who
        # wants top-level files as well should pass pattern="*".
        matches = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            for name in filenames:
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                    matches.append(rel)
        matches.sort()
        lines = matches[:500]
        if len(matches) > 500:
            lines.append(f"... and {len(matches) - 500} more")
        return "\n".join(lines)

    @tool("Search file contents for a regex", regex="the regex to search for",
          pattern="glob limiting which files are searched, default *")
    def grep(regex, pattern="*"):
        rx = re.compile(regex)
        hits = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            for name in filenames:
                if not fnmatch.fnmatch(name, pattern):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                try:
                    with open(os.path.join(dirpath, name), "r", errors="replace") as f:
                        for lineno, line in enumerate(f, 1):
                            if rx.search(line):
                                hits.append(f"{rel}:{lineno}: {line.rstrip(chr(10))[:200]}")
                                if len(hits) >= 200:
                                    return "\n".join(hits)
                except OSError:
                    continue
        return "\n".join(hits)

    return [read_file, write_file, edit_file, bash, list_files, grep, fetch_url]
