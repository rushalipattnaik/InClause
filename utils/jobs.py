import itertools
import os
import queue
import threading
import time
import uuid

TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", "1800"))
PRIORITY = {"Suspicious": 1, "Risky": 2, "Safe": 3}
MODES = ("flagged", "all", "none")


class JobManager:
    def __init__(self, retrieve_fn, explain_fn, ttl=TTL_SECONDS):
        self._retrieve = retrieve_fn
        self._explain = explain_fn
        self._ttl = ttl
        self._jobs = {}
        self._lock = threading.RLock()
        self._queue = queue.PriorityQueue()
        self._seq = itertools.count()
        self._worker = None
        self._last_cleanup = time.time()

    def create(self, filename, analysed_clauses, mode="flagged", doc_check=None):
        mode = mode if mode in MODES else "flagged"
        job_id = uuid.uuid4().hex
        clauses = []
        for i, c in enumerate(analysed_clauses):
            wanted = mode == "all" or (mode == "flagged" and c["label"] != "Safe")
            clauses.append({"idx": i, "text": c["text"], "label": c["label"], "reasons": c["reasons"],
                            "references": None, "explanation": None,
                            "explain_state": "queued" if wanted else "skipped"})
        job = {"id": job_id, "filename": filename, "clauses": clauses, "doc_check": doc_check, "version": 1,
               "retrieval_done": False, "deleted": False, "touched": time.time()}
        with self._lock:
            self._jobs[job_id] = job
        self._ensure_worker()
        self._put(0, ("retrieve", job_id, None))
        for c in sorted((c for c in clauses if c["explain_state"] == "queued"),
                        key=lambda c: (PRIORITY[c["label"]], c["idx"])):
            self._put(PRIORITY[c["label"]], ("explain", job_id, c["idx"]))
        return job_id

    def filename(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            return job["filename"] if job else ""

    def snapshot(self, job_id, since=None):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job["touched"] = time.time()
            states = [c["explain_state"] for c in job["clauses"]]
            requested = [s for s in states if s != "skipped"]
            finished = [s for s in requested if s in ("done", "error")]
            working = (not job["retrieval_done"]) or any(s in ("queued", "running") for s in states)
            snap = {"id": job_id, "version": job["version"], "status": "working" if working else "done",
                    "doc_check": job.get("doc_check"),
                    "retrieval_done": job["retrieval_done"], "total": len(job["clauses"]),
                    "counts": {lab: sum(c["label"] == lab for c in job["clauses"]) for lab in PRIORITY},
                    "progress": {"done": len(finished), "total": len(requested)}}
            if since is not None and since == job["version"]:
                snap["unchanged"] = True
            else:
                snap["clauses"] = [dict(c) for c in job["clauses"]]
            return snap

    def request_explain(self, job_id, idx):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or not (0 <= idx < len(job["clauses"])):
                return False
            clause = job["clauses"][idx]
            if clause["explain_state"] in ("skipped", "error"):
                clause["explain_state"] = "queued"
                clause["explanation"] = None
                job["version"] += 1
                self._put(0, ("explain", job_id, idx))
            return True

    def delete(self, job_id):
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job:
                job["deleted"] = True
            return job is not None

    def _put(self, priority, task):
        self._queue.put((priority, next(self._seq), task))

    def _ensure_worker(self):
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._loop, name="analysis-worker", daemon=True)
                self._worker.start()

    def _loop(self):
        while True:
            try:
                _, _, (kind, job_id, idx) = self._queue.get(timeout=5)
            except queue.Empty:
                self._cleanup()
                continue
            try:
                if kind == "retrieve":
                    self._run_retrieve(job_id)
                else:
                    self._run_explain(job_id, idx)
            except Exception as e:
                print(f"Worker error ({kind}): {type(e).__name__}")
            self._cleanup()

    def _get_live(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            return None if (job is None or job["deleted"]) else job

    def _run_retrieve(self, job_id):
        job = self._get_live(job_id)
        if not job:
            return
        for c in job["clauses"]:
            if self._get_live(job_id) is None:
                return
            if c["label"] == "Safe" and c["explain_state"] == "skipped":
                continue
            refs = self._retrieve(c["text"])
            with self._lock:
                c["references"] = refs
                if c["idx"] % 10 == 0:
                    job["version"] += 1
        with self._lock:
            job["retrieval_done"] = True
            job["version"] += 1

    def _run_explain(self, job_id, idx):
        job = self._get_live(job_id)
        if not job:
            return
        with self._lock:
            clause = job["clauses"][idx]
            if clause["explain_state"] != "queued":
                return
            clause["explain_state"] = "running"
            job["version"] += 1
            text, refs = clause["text"], clause["references"]
        if refs is None:
            refs = self._retrieve(text)
        try:
            explanation, state = self._explain(text, refs), "done"
        except RuntimeError as e:
            explanation, state = str(e), "error"
        except Exception:
            explanation, state = "Could not generate an explanation for this clause.", "error"
        with self._lock:
            if self._get_live(job_id) is None:
                return
            clause["references"] = refs
            clause["explanation"] = explanation
            clause["explain_state"] = state
            job["version"] += 1

    def _cleanup(self):
        now = time.time()
        if now - self._last_cleanup < 30:
            return
        self._last_cleanup = now
        with self._lock:
            for job_id in [j for j, job in self._jobs.items() if now - job["touched"] > self._ttl]:
                self._jobs.pop(job_id)["deleted"] = True


def build_report(snapshot, filename=""):
    counts = snapshot["counts"]
    lines = [f"# InClause review{': ' + filename if filename else ''}", "",
             f"{snapshot['total']} clauses analysed: **{counts['Suspicious']} suspicious**, "
             f"**{counts['Risky']} risky**, {counts['Safe']} safe.", "",
             "> General, AI-generated explanations for education only. Not legal advice.", ""]
    if snapshot["status"] == "working":
        lines[-1:] = ["> This report was created while explanations were still being generated, so some are missing.", ""]
    check = snapshot.get("doc_check")
    if check and check["missing"]:
        lines += [f"## Not found in this {check['type']}", "",
                  "Documents like this usually cover these topics, but no matching words were found. Check by hand.", ""]
        lines += [f"- {m}" for m in check["missing"]] + [""]
    for label in ("Suspicious", "Risky", "Safe"):
        group = [c for c in snapshot["clauses"] if c["label"] == label]
        if not group:
            continue
        lines += [f"## {label} ({len(group)})", ""]
        for c in group:
            lines.append(f"### Clause {c['idx'] + 1}")
            lines.append(f"> {c['text']}")
            if c["reasons"]:
                lines.append("")
                lines.append("**Why flagged:** " + "; ".join(f"{r['reason']} (\"{r['matched']}\")" for r in c["reasons"]))
            if c["explain_state"] == "done" and c["explanation"]:
                lines += ["", f"**What it means:** {c['explanation']}"]
            elif c["explain_state"] == "error":
                lines += ["", "**What it means:** unavailable (the local AI did not respond for this clause)."]
            if c["references"]:
                lines += ["", "**Related notes:**"]
                lines += [f"- {r['title']} ({r['source']})" for r in c["references"]]
            lines.append("")
    return "\n".join(lines)
