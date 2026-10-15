(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const form = $("upload-form"), fileInput = $("file"), drop = $("drop");
  const errorEl = $("error"), results = $("results"), cardsEl = $("cards");
  const LONG_CLAUSE = 260;
  let jobId = null, version = -1, timer = null, polling = false, built = false;
  let filter = "attention", query = "", lastStatus = "working";
  const cards = new Map();

  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const showError = (msg) => { errorEl.textContent = msg || ""; errorEl.hidden = !msg; };

  function showChosenFile() {
    const file = fileInput.files && fileInput.files[0];
    $("drop-title").textContent = file ? file.name : "Drop a contract here or click to choose";
    drop.classList.toggle("has-file", Boolean(file));
  }
  fileInput.addEventListener("change", showChosenFile);
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      showChosenFile();
    }
  });

  async function dropJob() {
    if (!jobId) return;
    const old = jobId;
    jobId = null;
    stopPolling();
    try { await fetch(`/api/jobs/${old}`, { method: "DELETE" }); } catch (_) {}
  }

  function resetView() {
    cardsEl.replaceChildren();
    cards.clear();
    built = false;
    version = -1;
    lastStatus = "working";
    filter = "attention";
    query = "";
    $("search").value = "";
    syncChips();
    results.hidden = true;
  }

  function setBusy(busy, label) {
    $("submit-btn").disabled = busy;
    $("sample-btn").disabled = busy;
    $("submit-btn").textContent = busy ? label : "Analyse";
  }

  async function analyse(body) {
    showError("");
    setBusy(true, "Reading...");
    try {
      const res = await fetch("/api/analyze", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Something went wrong. Please try again.");
      await dropJob();
      resetView();
      jobId = data.job_id;
      results.hidden = false;
      $("download").href = `/api/jobs/${jobId}/report.md`;
      if (data.truncated) showError(`This document is very long: only the first ${data.max_clauses} clauses are analysed.`);
      startPolling();
    } catch (err) {
      showError(err.message);
    } finally {
      setBusy(false);
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    analyse(new FormData(form));
  });

  $("sample-btn").addEventListener("click", async () => {
    try {
      const res = await fetch("/static/sample-lease.txt");
      if (!res.ok) throw new Error();
      const body = new FormData();
      body.append("document", new File([await res.blob()], "sample-lease.txt", { type: "text/plain" }));
      body.append("explain_mode", $("mode").value);
      analyse(body);
    } catch (_) {
      showError("Could not load the sample document.");
    }
  });

  function startPolling() { stopPolling(); poll(); timer = setInterval(poll, 1500); }
  function stopPolling() { if (timer) clearInterval(timer); timer = null; }

  async function poll() {
    if (!jobId || polling) return;
    polling = true;
    try {
      const res = await fetch(`/api/jobs/${jobId}?since=${version}`);
      if (res.status === 404) {
        stopPolling(); jobId = null; resetView();
        showError("This session expired or was cleared. Please upload the document again.");
        return;
      }
      const data = await res.json();
      if (!data.unchanged) { version = data.version; render(data); }
      lastStatus = data.status;
      renderSummary(data);
      renderDocCheck(data.doc_check);
      renderProgress(data);
      if (data.status === "done") stopPolling();
    } catch (_) {
    } finally {
      polling = false;
    }
  }

  function render(data) {
    if (!built) {
      build(data.clauses);
      const attention = data.counts.Suspicious + data.counts.Risky;
      if (attention === 0) { filter = "all"; syncChips(); }
    }
    data.clauses.forEach(update);
    applyFilter();
  }

  function build(clauses) {
    clauses.forEach((c) => {
      const root = el("article", "clause");
      root.dataset.label = c.label;
      const head = el("div", "clause-head");
      head.append(el("span", `pill ${c.label}`, c.label), el("span", "", `Clause ${c.idx + 1}`));
      const text = el("p", "clause-text", c.text);
      root.append(head, text);
      if (c.text.length > LONG_CLAUSE) {
        text.classList.add("clamped");
        const toggle = el("button", "btn link", "Show full clause");
        toggle.type = "button";
        toggle.addEventListener("click", () => {
          const open = text.classList.toggle("clamped");
          toggle.textContent = open ? "Show full clause" : "Show less";
        });
        root.append(toggle);
      }
      if (c.reasons.length) {
        const tags = el("div", "tags");
        c.reasons.forEach((r) => {
          const tag = el("span", "tag");
          tag.append(document.createTextNode(`${r.reason}: `), el("b", "", `"${r.matched}"`));
          tags.append(tag);
        });
        root.append(tags);
      }
      const explain = el("div"), refs = el("div");
      root.append(explain, refs);
      cards.set(c.idx, { root, explain, refs, text: c.text, explainKey: null, refsKey: null });
      cardsEl.append(root);
    });
    built = true;
  }

  function update(c) {
    const card = cards.get(c.idx);
    if (!card) return;

    const explainKey = `${c.explain_state}|${c.explanation || ""}`;
    if (card.explainKey !== explainKey) {
      card.explainKey = explainKey;
      card.explain.replaceChildren();
      if (c.explain_state === "done") {
        const box = el("div", "explain");
        box.append(el("span", "label", "What it means"), document.createTextNode(c.explanation));
        card.explain.append(box);
      } else if (c.explain_state === "queued" || c.explain_state === "running") {
        card.explain.append(el("div", "explain pending", c.explain_state === "running" ? "Working on it..." : "Waiting in line..."));
      } else if (c.explain_state === "error") {
        card.explain.append(el("div", "explain error", c.explanation || "Could not explain this clause."));
        card.explain.append(button("Try again", () => requestExplain(c.idx)));
      } else {
        card.explain.append(button("Explain this clause", () => requestExplain(c.idx)));
      }
    }

    const refs = c.references || [];
    const refsKey = refs.map((r) => r.id).join(",");
    if (card.refsKey !== refsKey) {
      card.refsKey = refsKey;
      card.refs.replaceChildren();
      if (refs.length) {
        const det = el("details");
        det.append(el("summary", "", refs.length === 1 ? "Related legal note" : `Related legal notes (${refs.length})`));
        refs.forEach((r) => {
          const box = el("div", "ref");
          box.append(el("b", "", r.title), el("span", "", r.text),
                     el("small", "", `Source: ${r.source} | match ${Math.round(r.score * 100)}%${r.reviewed ? "" : " | not yet reviewed"}`));
          det.append(box);
        });
        card.refs.append(det);
      }
    }
  }

  function button(label, onClick) {
    const b = el("button", "btn small", label);
    b.type = "button";
    b.addEventListener("click", onClick);
    return b;
  }

  async function requestExplain(idx) {
    try {
      const res = await fetch(`/api/jobs/${jobId}/explain/${idx}`, { method: "POST" });
      if (!res.ok) throw new Error();
      version = -1;
      startPolling();
    } catch (_) {
      showError("Could not start that explanation. Please try again.");
    }
  }

  function renderSummary(d) {
    const bar = $("riskbar"), counts = $("counts");
    bar.replaceChildren();
    counts.replaceChildren(el("span", "", `${d.total} clauses`));
    ["Suspicious", "Risky", "Safe"].forEach((label) => {
      const n = d.counts[label];
      if (n) {
        const seg = el("span", `seg-${label}`);
        seg.style.width = `${(100 * n) / d.total}%`;
        bar.append(seg);
      }
      const item = el("span");
      const dot = el("span", `dot seg-${label}`);
      item.append(dot, el("b", "", String(n)), document.createTextNode(` ${label.toLowerCase()}`));
      counts.append(item);
    });
    $("n-attention").textContent = d.counts.Suspicious + d.counts.Risky;
    $("n-Suspicious").textContent = d.counts.Suspicious;
    $("n-Risky").textContent = d.counts.Risky;
    $("n-Safe").textContent = d.counts.Safe;
    $("n-all").textContent = d.total;
  }

  function renderDocCheck(check) {
    const box = $("doccheck");
    if (!check || !check.missing.length) { box.hidden = true; return; }
    box.replaceChildren(el("b", "", `This looks like a ${check.type}. Not found in the text (worth checking by hand):`));
    const list = el("ul");
    check.missing.forEach((m) => list.append(el("li", "", m)));
    box.append(list);
    box.hidden = false;
  }

  function renderProgress(d) {
    const wrap = $("progress-wrap");
    if (d.status === "done") { wrap.hidden = true; return; }
    wrap.hidden = false;
    const { done, total } = d.progress;
    $("progress-text").textContent = d.retrieval_done
      ? `Explaining clauses, most serious first: ${done} of ${total}`
      : "Finding related legal notes...";
    $("bar-fill").style.width = (total ? Math.round((100 * done) / total) : 0) + "%";
  }

  function matchesFilter(label) {
    if (filter === "all") return true;
    if (filter === "attention") return label !== "Safe";
    return label === filter;
  }

  function applyFilter() {
    let shown = 0;
    cards.forEach((card) => {
      const visible = matchesFilter(card.root.dataset.label) && (!query || card.text.toLowerCase().includes(query));
      card.root.hidden = !visible;
      if (visible) shown += 1;
    });
    $("empty").hidden = !built || shown > 0;
  }

  function syncChips() {
    document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c.dataset.filter === filter));
  }

  $("filters").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    filter = chip.dataset.filter;
    syncChips();
    applyFilter();
  });
  $("search").addEventListener("input", (e) => { query = e.target.value.trim().toLowerCase(); applyFilter(); });
  $("download").addEventListener("click", (e) => {
    if (lastStatus !== "done" && !window.confirm("InClause is still explaining clauses, so the report will be incomplete. Download it anyway?")) {
      e.preventDefault();
    }
  });
  $("print").addEventListener("click", () => window.print());
  $("clear").addEventListener("click", async () => {
    await dropJob();
    resetView();
    showError("");
    form.reset();
    showChosenFile();
  });
})();
