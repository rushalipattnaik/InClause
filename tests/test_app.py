import io
import time

import pytest

import app as app_module
from utils.jobs import JobManager

LEASE = b"""RESIDENTIAL LEASE

1.1 The security deposit of Rs. 54,000 shall be non-refundable and kept by the Landlord for any reason.
1.2 If rent is not paid on time the Tenant shall pay a late fee of Rs. 500 for each day of delay.
1.3 This lease is for a period of eleven months commencing from the first of March.
"""


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "manager", JobManager(lambda t: [], lambda t, r: "simple explanation"))
    return app_module.app.test_client()


def upload(client, content=LEASE, name="lease.txt", mode="flagged"):
    return client.post("/api/analyze", data={"document": (io.BytesIO(content), name), "explain_mode": mode},
                       content_type="multipart/form-data")


def wait(client, job_id):
    for _ in range(100):
        data = client.get(f"/api/jobs/{job_id}").get_json()
        if data["status"] == "done":
            return data
        time.sleep(0.02)
    raise AssertionError("not done")


def test_index_and_health(client):
    assert client.get("/").status_code == 200
    assert client.get("/health").get_json() == {"status": "ok"}


def test_page_loads_nothing_from_the_internet(client):
    import re
    html = client.get("/").get_data(as_text=True)
    assert not re.search(r"""(?:src|href)\s*=\s*["']\s*(?:https?:)?//""", html)


def test_full_flow(client):
    r = upload(client)
    assert r.status_code == 200
    body = r.get_json()
    assert body["total"] == 3
    data = wait(client, body["job_id"])
    labels = [c["label"] for c in data["clauses"]]
    assert labels == ["Suspicious", "Risky", "Safe"]
    assert data["clauses"][0]["explanation"] == "simple explanation"
    assert data["clauses"][2]["explain_state"] == "skipped"


def test_no_clause_limit_of_15(client):
    big = "\n".join(f"{i}.1 The party number {i} shall pay a late fee of Rs. {i} if payment is delayed." for i in range(1, 61))
    body = upload(client, big.encode(), mode="none").get_json()
    assert body["total"] == 60 and body["truncated"] is False


def test_validation_errors(client):
    assert client.post("/api/analyze", data={}, content_type="multipart/form-data").status_code == 400
    assert upload(client, b"x", "virus.exe").status_code == 400
    assert upload(client, b"too short", "a.txt").status_code == 422


def test_report_and_delete(client):
    job_id = upload(client).get_json()["job_id"]
    wait(client, job_id)
    rep = client.get(f"/api/jobs/{job_id}/report.md")
    assert rep.status_code == 200 and b"non-refundable" in rep.data
    assert "inclause-report.md" in rep.headers["Content-Disposition"]
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_responses_are_not_cacheable(client):
    job_id = upload(client).get_json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}").headers["Cache-Control"] == "no-store"


def test_upload_is_not_left_on_disk(client, tmp_path, monkeypatch):
    import os
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    upload(client)
    assert os.listdir(tmp_path) == []


def test_sample_lease_is_served_and_analysable(client):
    sample = client.get("/static/sample-lease.txt")
    assert sample.status_code == 200
    body = upload(client, sample.data, "sample-lease.txt", mode="none").get_json()
    data = wait(client, body["job_id"])
    labels = {c["label"] for c in data["clauses"]}
    assert body["total"] >= 15 and {"Suspicious", "Risky", "Safe"} <= labels
    assert data["doc_check"]["type"] == "rental agreement"
    assert "Lock-in period" in data["doc_check"]["missing"]
