# Astra

The smallest agent harness that is still a real one. Twelve modules, 1,430
lines, and zero dependencies — every import is the Python standard library,
including the HTTP client that talks to the API.

It is built to be read. Each module owns one idea and knows nothing about the
others; the only file that composes them is `harness.py`, and the only file
that knows the shape of the Claude API is `provider.py`. Swapping providers
means rewriting one file. Adding a tool means writing a function.

```python
from astra import Harness

Harness(workdir="./project").run("Add a --json flag to the CLI and test it")
```

## Running it

Set a key once:

```bash
export ANTHROPIC_API_KEY=sk-...        # or ASTRA_API_KEY
export ASTRA_MODEL=claude-sonnet-5  # optional; this is the default
```

Three forms:

```bash
# 1. interactive — a prompt loop, safe mode, every write asks first
python3 -m astra -d ./project

# 2. headless — one task, yolo mode, exit 0
python3 -m astra -d ./project -p "Fix the failing test in tests/test_api.py"

# 3. resume — pick up the most recent session, however it ended
python3 -m astra -d ./project --resume
```

Flags: `-p/--prompt`, `-d/--workdir`, `-m/--model`, `--mode {safe,yolo,read-only}`,
`--resume`, `--max-turns`.

The mode default depends on how you invoke it. Interactively it is `safe`,
because someone is there to answer "approve this?". With `-p` it is `yolo`,
because a prompt nobody answers is a hang. `read-only` allows `read_file`,
`list_files`, and `grep` and nothing else.

Deny patterns in `security.py` — `rm -rf /`, `sudo`, `curl | sh`, `git push
--force`, raw device writes — are checked *before* the mode, so no mode
bypasses them.

## Anatomy

| Modules | Lines | What it adds |
|---------|-------|--------------|
| `provider.py` · `loop.py` | 236 | The turn cycle, and the one file that knows the wire format |
| `tools.py` · `security.py` | 382 | Tools from function signatures, a path jail, a policy gate |
| `context.py` · `memory.py` · `skills.py` | 279 | Compaction, `ASTRA.md`, loadable skills |
| `session.py` · `subagent.py` · `harness.py` | 322 | Durable sessions, crash repair, delegation, composition |
| `cli.py` · `fleet.py` | 211 | The front door, and many agents at once |

Each component plugs into seams in the loop. `loop.py` carries
`before_turn` which is filled by compaction; `before_tool`
provides security checks via the policy gate.

## Composition

The harness takes extra tools as ordinary functions. The `@tool` decorator
reads the signature and builds the schema, so there is no second place to keep
the argument list correct:

```python
import json
import urllib.request

from astra import Harness, Policy, tool


@tool("Look up the current status of a deploy by its id",
      deploy_id="the deploy identifier, e.g. dpl_9f2a")
def deploy_status(deploy_id):
    url = f"https://deploys.internal/api/{deploy_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.dumps(json.loads(resp.read()), indent=2)


agent = Harness(
    workdir="./service",
    policy=Policy("safe", approver=lambda call, reason: input(f"{reason} ") == "y"),
    extra_tools=[deploy_status],
    system_extra="Deploys are gated on the staging soak passing.",
)

print(agent.run("Is dpl_9f2a healthy? If not, say what failed."))
```

A parameter without a default is required, one with a default is optional, and
every parameter is a string — a tool that wants an integer parses it, which is
one rule instead of six edge cases in schema generation.

## Fleets

Independent jobs across separate directories, concurrently:

```python
from astra import Harness, run_fleet

results = run_fleet(
    jobs=[
        {"name": "api", "workdir": "./api", "task": "Add request id logging"},
        {"name": "web", "workdir": "./web", "task": "Fix the flaky checkout test"},
    ],
    make_harness=lambda workdir: Harness(workdir=workdir),
    max_workers=4,
)

for r in results:
    print(f"{r['name']}: {'ok' if r['ok'] else 'FAILED'}\n{r['report']}\n")
```

Results come back in input order, and a job that raises becomes an `ok: False`
row rather than taking the fleet down with it.

## Durability

Every message is appended to `.astra/sessions/<timestamp>-<slug>.jsonl` as
it happens — not flushed at the end, because a crash is the case durability
exists for. On load, a torn final line is discarded and any tool call the
process died inside gets a synthesised result:

```
Interrupted before this ran (process restarted).
```

That is what makes the transcript legal to replay: the API rejects an assistant
turn whose `tool_use` blocks have no matching `tool_result`. The agent reads the
notice, sees what never finished, and decides whether to run it again.

## Memory and skills

`remember` appends a line to `ASTRA.md` in the working directory, and every
later run reads it back through the system prompt — so a fact learned once
survives the process that learned it.

A skill is a directory holding a `SKILL.md`:

```
skills/
  brand-voice/
    SKILL.md
```

Its front-matter `description:` goes in the catalog in the system prompt; the
body loads only when the agent calls `use_skill`. Ten skills cost ten lines of
context instead of ten documents.

## Tests

```bash
python3 tests/test_astra.py
```

Eighty-eight checks, no test runner and no dependencies — the suite is a plain
script like the demos. It exits non-zero with a named failure list, so it drops
into CI unchanged.

The whole harness runs offline. `provider.complete` is swapped for a fake that
returns scripted replies, which works precisely because every module above
`provider.py` speaks the neutral message format — so the loop, the tools, the
policy gate, compaction, sessions, and the fleet are all the real code, driven
by canned replies instead of a model. The provider boundary that makes swapping
providers a one-file job is the same seam that makes the rest testable without
a network.

It pins down the behaviour that is easy to break and quiet when broken:

- consecutive tool results merge into **one** user message, and each
  `tool_use_id` carries the signature of the call it answers — get either wrong
  and the API rejects the turn
- a deny pattern blocks `rm -rf /` even under `yolo`, and `safe` mode with no
  approver refuses rather than allows
- a blocked tool never runs, and the reason goes back to the model as a result
- a torn final line in a session is discarded, and a tool call the process died
  inside comes back with a synthesised result
- `list_files` defaults to `**/*`, which fnmatch only matches against paths
  containing a `/` — top-level files need `pattern="*"`
- a fleet job that raises becomes an `ok: False` row instead of taking the
  fleet down

That the provider is stubbed is also the limit worth knowing. A green run says
nothing about the wire format being accepted, the key being valid, or the model
being reachable — only that the harness around the call is correct. Run one of
the demos for the other half.
