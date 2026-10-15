import threading
import time

from utils.jobs import JobManager, build_report


def make_clauses():
    return [
        {"text": "Safe clause number one about dates.", "label": "Safe", "reasons": []},
        {"text": "Risky clause about a late fee.", "label": "Risky",
         "reasons": [{"reason": "late fee or interest", "matched": "late fee"}]},
        {"text": "Suspicious clause: non-refundable deposit.", "label": "Suspicious",
         "reasons": [{"reason": "no refund", "matched": "non-refundable"}]},
        {"text": "Another risky clause on termination.", "label": "Risky", "reasons": []},
    ]


def wait_done(mgr, job_id, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        snap = mgr.snapshot(job_id)
        if snap["status"] == "done":
            return snap
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_flagged_mode_explains_in_priority_order_and_skips_safe():
    order = []
    mgr = JobManager(lambda t: [{"title": "T", "source": "S", "text": "x", "score": 0.9}],
                     lambda t, refs: order.append(t) or f"explained: {t[:10]}")
    snap = wait_done(mgr, mgr.create("f.txt", make_clauses(), "flagged"))
    assert order[0].startswith("Suspicious")
    assert {o.split()[0] for o in order[1:]} == {"Risky", "Another"}
    assert snap["clauses"][0]["explain_state"] == "skipped"
    assert snap["clauses"][0]["explanation"] is None
    assert snap["progress"] == {"done": 3, "total": 3}
    flagged = [c for c in snap["clauses"] if c["label"] != "Safe"]
    assert all(c["references"] for c in flagged)
    assert snap["clauses"][0]["references"] is None


def test_all_mode_explains_everything():
    mgr = JobManager(lambda t: [], lambda t, r: "ok")
    snap = wait_done(mgr, mgr.create("f.txt", make_clauses(), "all"))
    assert all(c["explain_state"] == "done" for c in snap["clauses"])


def test_none_mode_and_on_demand_explain():
    mgr = JobManager(lambda t: [], lambda t, r: "on demand answer")
    job_id = mgr.create("f.txt", make_clauses(), "none")
    snap = wait_done(mgr, job_id)
    assert all(c["explain_state"] == "skipped" for c in snap["clauses"])
    assert mgr.request_explain(job_id, 0)
    snap = wait_done(mgr, job_id)
    assert snap["clauses"][0]["explanation"] == "on demand answer"


def test_explain_failure_is_contained():
    def boom(t, r):
        raise RuntimeError("ollama down")
    mgr = JobManager(lambda t: [], boom)
    snap = wait_done(mgr, mgr.create("f.txt", make_clauses(), "flagged"))
    assert {c["explain_state"] for c in snap["clauses"] if c["label"] != "Safe"} == {"error"}
    mgr.request_explain(snap["id"], 1)


def test_unchanged_polling_and_version_bumps():
    gate = threading.Event()
    mgr = JobManager(lambda t: [], lambda t, r: (gate.wait(2), "ok")[1])
    job_id = mgr.create("f.txt", make_clauses(), "flagged")
    snap = mgr.snapshot(job_id)
    again = mgr.snapshot(job_id, since=snap["version"])
    assert again.get("unchanged") or again["version"] != snap["version"]
    gate.set()
    wait_done(mgr, job_id)


def test_delete_removes_job():
    mgr = JobManager(lambda t: [], lambda t, r: "ok")
    job_id = mgr.create("f.txt", make_clauses(), "none")
    assert mgr.delete(job_id) is True
    assert mgr.snapshot(job_id) is None
    assert mgr.request_explain(job_id, 0) is False


def test_expired_jobs_are_purged():
    mgr = JobManager(lambda t: [], lambda t, r: "ok", ttl=0)
    job_id = mgr.create("f.txt", make_clauses(), "none")
    wait_done(mgr, job_id)
    time.sleep(0.01)
    mgr._last_cleanup = 0
    mgr._cleanup()
    assert mgr.snapshot(job_id) is None


def test_report_contains_clauses_reasons_and_disclaimer():
    mgr = JobManager(lambda t: [{"title": "Deposits", "source": "Src", "text": "x", "score": 0.8}],
                     lambda t, r: "plain words")
    job_id = mgr.create("lease.pdf", make_clauses(), "flagged")
    report = build_report(wait_done(mgr, job_id), "lease.pdf")
    assert "InClause review: lease.pdf" in report and "1 suspicious" in report
    assert "no refund" in report and "plain words" in report and "Deposits (Src)" in report
    assert "Not legal advice" in report


def test_runtime_error_message_is_shown_to_the_user():
    def offline(t, r):
        raise RuntimeError("Local AI is offline. Run: ollama pull llama3.2:3b")
    mgr = JobManager(lambda t: [], offline)
    snap = wait_done(mgr, mgr.create("f.txt", make_clauses(), "flagged"))
    assert "ollama pull" in snap["clauses"][2]["explanation"]


def test_report_says_when_it_is_incomplete_or_a_clause_failed():
    gate = threading.Event()
    mgr = JobManager(lambda t: [], lambda t, r: (gate.wait(3), "late answer")[1])
    job_id = mgr.create("lease.pdf", make_clauses(), "flagged")
    early = build_report(mgr.snapshot(job_id), "lease.pdf")
    assert "still being generated" in early
    gate.set()
    wait_done(mgr, job_id)

    def offline(t, r):
        raise RuntimeError("offline")
    failing = JobManager(lambda t: [], offline)
    snap = wait_done(failing, failing.create("lease.pdf", make_clauses(), "flagged"))
    assert "unavailable" in build_report(snap, "lease.pdf")
    assert "still being generated" not in build_report(snap, "lease.pdf")


def test_document_check_is_stored_and_reported():
    check = {"type": "rental agreement", "missing": ["Lock-in period", "Rent increases"]}
    mgr = JobManager(lambda t: [], lambda t, r: "ok")
    job_id = mgr.create("lease.pdf", make_clauses(), "none", check)
    snap = wait_done(mgr, job_id)
    assert snap["doc_check"] == check
    report = build_report(snap, "lease.pdf")
    assert "Not found in this rental agreement" in report and "- Lock-in period" in report


def test_no_document_check_means_no_section():
    mgr = JobManager(lambda t: [], lambda t, r: "ok")
    snap = wait_done(mgr, mgr.create("f.txt", make_clauses(), "none"))
    assert snap["doc_check"] is None and "Not found in" not in build_report(snap)
