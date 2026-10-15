import os
import tempfile
import uuid

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

load_dotenv()

from utils.analysis import analyze_all  # noqa: E402
from utils.clause_splitter import split_into_clauses  # noqa: E402
from utils.doc_checks import check_document  # noqa: E402
from utils.extractor import allowed_file, extract_text  # noqa: E402
from utils.jobs import MODES, JobManager, build_report  # noqa: E402
from utils.rag_engine import retrieve_reference  # noqa: E402
from utils.summarizer import summarize_clause  # noqa: E402

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "15"))
MAX_CLAUSES = int(os.environ.get("MAX_CLAUSES", "500"))

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

manager = JobManager(retrieve_reference, summarize_clause)


@app.after_request
def no_caching(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify(error=f"File is too large. The limit is {MAX_UPLOAD_MB} MB."), 413


@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_UPLOAD_MB)


@app.get("/health")
def health():
    return {"status": "ok"}, 200


@app.post("/api/analyze")
def analyze():
    file = request.files.get("document")
    mode = request.form.get("explain_mode", "flagged")
    if mode not in MODES:
        mode = "flagged"
    if not file or not file.filename:
        return jsonify(error="Please choose a file to upload."), 400
    if not allowed_file(file.filename):
        return jsonify(error="Unsupported file type. Please upload a PDF, DOCX or TXT file."), 400

    filename = secure_filename(file.filename) or "document"
    suffix = os.path.splitext(filename)[1].lower()
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=f"{uuid.uuid4().hex}_")
    os.close(fd)
    try:
        file.save(path)
        text = extract_text(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    if text.startswith("Error:"):
        return jsonify(error=text.removeprefix("Error:").strip()), 422

    clauses = split_into_clauses(text)
    if not clauses:
        return jsonify(error="No clauses could be identified in this document. Try a different file."), 422
    truncated = len(clauses) > MAX_CLAUSES
    clauses = clauses[:MAX_CLAUSES]

    analysed = [{"text": c, **result} for c, result in zip(clauses, analyze_all(clauses), strict=True)]
    job_id = manager.create(filename, analysed, mode, check_document(text))
    return jsonify(job_id=job_id, total=len(analysed), truncated=truncated, max_clauses=MAX_CLAUSES)


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    since = request.args.get("since", type=int)
    snap = manager.snapshot(job_id, since)
    if snap is None:
        return jsonify(error="This session has expired or was cleared. Please upload the document again."), 404
    return jsonify(snap)


@app.post("/api/jobs/<job_id>/explain/<int:idx>")
def explain_one(job_id, idx):
    if not manager.request_explain(job_id, idx):
        return jsonify(error="Not found."), 404
    return jsonify(ok=True)


@app.get("/api/jobs/<job_id>/report.md")
def report(job_id):
    snap = manager.snapshot(job_id)
    if snap is None:
        return jsonify(error="Not found."), 404
    body = build_report(snap, manager.filename(job_id))
    return Response(body, mimetype="text/markdown",
                    headers={"Content-Disposition": "attachment; filename=inclause-report.md"})


@app.delete("/api/jobs/<job_id>")
def clear(job_id):
    manager.delete(job_id)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
