const keywordsInput = document.getElementById("keywords");
const companiesInput = document.getElementById("companies");
const quickPicksEl = document.getElementById("quick-picks");
const form = document.getElementById("search-form");
const searchBtn = document.getElementById("search-btn");
const statusLine = document.getElementById("status-line");
const feed = document.getElementById("feed");
const disclaimerEl = document.getElementById("disclaimer");

async function loadQuickPicks() {
  try {
    const res = await fetch("/quick-picks");
    const data = await res.json();
    quickPicksEl.innerHTML = "";
    for (const pick of data.quick_picks) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "quick-pick";
      btn.textContent = pick;
      btn.addEventListener("click", () => {
        keywordsInput.value = pick;
        keywordsInput.focus();
      });
      quickPicksEl.appendChild(btn);
    }
  } catch (err) {
    // Quick-picks are a convenience only — fail silently if unavailable.
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function renderJobRow(job, index) {
  const row = document.createElement("article");
  row.className = "job-row";
  row.style.animationDelay = `${Math.min(index, 10) * 45}ms`;

  const badge = job.is_direct_apply
    ? '<span class="badge-direct">Direct</span>'
    : "";

  const summary = job.summary
    ? `<p class="job-summary">${escapeHtml(job.summary)}</p>`
    : "";

  const metaParts = [
    job.location || "Location not specified",
    job.experience_level || "Experience not specified",
    job.posted || "Posted date not shown",
    job.source ? `via ${job.source}` : "",
  ].filter(Boolean);

  row.innerHTML = `
    <div class="job-main">
      <div class="job-title-line">
        <span class="job-title">${escapeHtml(job.title || "Untitled role")}</span>
        <span class="job-company">${escapeHtml(job.company || "")}</span>
        ${badge}
      </div>
      <div class="job-meta">${metaParts.map(escapeHtml).join("  ·  ")}</div>
      ${summary}
    </div>
  `;

  const applyWrap = document.createElement("div");
  if (job.apply_url) {
    const a = document.createElement("a");
    a.className = "job-apply";
    a.href = job.apply_url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = "Apply";
    applyWrap.appendChild(a);
  }
  row.appendChild(applyWrap);

  return row;
}

function renderListing(listing) {
  feed.innerHTML = "";

  const jobs = listing.jobs || [];
  const noResults = listing.companies_with_no_results || [];

  if (listing.notes) {
    const note = document.createElement("div");
    note.className = "feed-note";
    note.innerHTML = `<strong>Note:</strong> ${escapeHtml(listing.notes)}`;
    feed.appendChild(note);
  }

  if (noResults.length) {
    const note = document.createElement("div");
    note.className = "feed-note";
    note.innerHTML = `<strong>No current openings found at:</strong> ${escapeHtml(
      noResults.join(", ")
    )}`;
    feed.appendChild(note);
  }

  if (jobs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "feed-empty";
    empty.textContent = "No open postings found for this search.";
    feed.appendChild(empty);
    return;
  }

  jobs.forEach((job, i) => feed.appendChild(renderJobRow(job, i)));
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const keywords = keywordsInput.value.trim();
  if (!keywords) {
    keywordsInput.focus();
    return;
  }

  const payload = {
    keywords,
    companies: companiesInput.value
      .split(",")
      .map((c) => c.trim())
      .filter(Boolean),
  };

  searchBtn.disabled = true;
  searchBtn.textContent = "Searching…";
  statusLine.classList.remove("is-error");
  statusLine.textContent = "Searching live career pages and ATS boards — this can take 15-30s…";
  feed.innerHTML = "";
  disclaimerEl.hidden = true;

  try {
    const res = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    const jobCount = (data.listing.jobs || []).length;

    statusLine.textContent = `Searched ${data.companies_searched.join(
      ", "
    )} for "${data.query}" — found ${jobCount} posting${
      jobCount === 1 ? "" : "s"
    } (${data.latency_ms}ms)`;

    renderListing(data.listing);

    if (data.listing.disclaimer) {
      disclaimerEl.textContent = data.listing.disclaimer;
      disclaimerEl.hidden = false;
    }
  } catch (err) {
    statusLine.classList.add("is-error");
    statusLine.textContent = err.message || "Something went wrong.";
  } finally {
    searchBtn.disabled = false;
    searchBtn.textContent = "Search";
  }
});

loadQuickPicks();
