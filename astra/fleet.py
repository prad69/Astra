"""astra.fleet

Concept: many agents at once. A sub-agent (subagent.py) is delegation
inside one run, sequential and reporting back into the parent's context. A
fleet is the other axis: independent jobs, usually in separate
directories, run at the same time because nothing about them needs
ordering. The whole file is a ThreadPoolExecutor and an error convention.

Design rules this file embodies:
  - Threads, not processes. The work is a socket waiting on the API, so
    the GIL is released for nearly all of it; processes would buy nothing
    here and cost pickling every harness.
  - A failed job is a result, not an exception. One job dying should not
    take the fleet with it or discard the reports that did come back, so
    exceptions are caught per job and become {"ok": False} rows -- callers
    read one list and check a flag rather than wrapping every call.
  - Results come back in input order. as_completed order is arrival order,
    which changes between runs on identical input; sorting the futures
    back into the order the caller wrote them makes the output diffable.
"""

from concurrent.futures import ThreadPoolExecutor


def run_fleet(jobs, make_harness, max_workers=4):
    """Run jobs concurrently and return one result row per job, in order.

    Each job is {"name", "workdir", "task"}; make_harness(workdir) builds
    the agent for it. Returns [{"name", "ok", "report"}], where a failed
    job carries "<ExceptionType>: <message>" as its report.
    """
    def run_one(job):
        try:
            report = make_harness(job["workdir"]).run(job["task"])
            return {"name": job["name"], "ok": True, "report": report}
        except Exception as e:
            return {"name": job["name"], "ok": False,
                    "report": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # map preserves input order, which as_completed does not.
        return list(pool.map(run_one, jobs))
