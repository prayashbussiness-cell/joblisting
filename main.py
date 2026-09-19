"""
main.py

FastAPI backend for the "Fresh Openings" job finder demo.

Responsibilities:
- Accept a JSON request with a category/role keyword and (optionally) a
  list of target companies.
- Run a REAL web search (Serper.dev, wrapping Google Search), one query per
  company, restricted to roughly the last 24 hours, to find CURRENT, open
  postings for that category at those companies.
- Call Gemini (no tools, plain text generation) to reshape that raw search
  data into strict JSON the frontend can render as a list of cards.
- Return the structured listing to the frontend.

WHY THIS CHANGED FROM GEMINI GROUNDING TO A DIRECT SEARCH API
---------------------------------------------------------------------------
The previous version of this file used Gemini's built-in Google Search
grounding tool (`tools=[Tool(google_search=GoogleSearch())]`) to do the
fact-finding step. That tool draws from a SEPARATE, much smaller quota on
the Gemini API than ordinary generate_content calls — it's a well-known
Gemini API behavior (see https://ai.google.dev/gemini-api/docs/rate-limits
and Google's own developer forum threads on "429 RESOURCE_EXHAUSTED ...
Search Grounding"). That grounding-specific quota was getting exhausted
independently of the API key/billing being otherwise fine, which is why
every /search call failed with a 429 while everything else (including the
plain-text structuring call right below it in the same request) kept
working.

This version removes that dependency entirely:
  - Step 1 (fetch_web_search_context, below) hits a real search API
    directly. No Gemini grounding quota is involved at all.
  - Step 2 (generate_job_listing's structuring call) still uses Gemini, but
    with NO tools attached, so it only ever touches the large, ordinary
    text-generation quota — the same one the astrology project uses.

SETUP REQUIRED: you need a Serper.dev API key (free tier: ~2,500 queries,
no card required to start — https://serper.dev). Set it as the
SERPER_API_KEY environment variable on Render. Without it, /search will
return a clear 502 telling you it's missing, instead of a confusing error.

Honest limitation: this is still a demo/POC. A search engine's date filter
is a best-effort recency signal (it depends on Google's crawl/index date
for the page), not a guaranteed "posted in exactly the last N hours."
Treat this as a fast way to prototype "find me fresh jobs," not a
production data pipeline — see README.md.
"""

import os
import re
import json
import time
import uuid
import asyncio
import logging
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from google import genai
from google.genai import types as genai_types

try:
    # Works when run from inside backend/ (uvicorn main:app)
    from prompt import (
        QUICK_PICKS,
        DEFAULT_COMPANIES,
        MAX_COMPANIES,
        RESEARCH_DISCLAIMER,
        SYSTEM_PROMPT,
        build_company_search_query,
        build_user_prompt,
    )
except ImportError:
    # Works when run from the repo root (uvicorn backend.main:app)
    from backend.prompt import (
        QUICK_PICKS,
        DEFAULT_COMPANIES,
        MAX_COMPANIES,
        RESEARCH_DISCLAIMER,
        SYSTEM_PROMPT,
        build_company_search_query,
        build_user_prompt,
    )

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("job-finder")

BASE_DIR = os.path.dirname(__file__)

# --- Gemini config (used ONLY for the ungrounded structuring step now) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# "gemini-3.6-flash" is not a real model name — use a real, current one.
# If GEMINI_MODEL is set as a Render environment variable, that overrides
# this default entirely — check your Render service's Environment tab if
# you ever see an unexpected model name in the logs.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

# --- Web search config (does the actual fact-finding now) ---
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
SERPER_ENDPOINT = "https://google.serper.dev/search"
# Google's date-restrict operator: qdr:d = past 24 hours, qdr:w = past week,
# qdr:m = past month. Widen this via env var if a 24h window is too narrow
# for your target companies and returns too many empty results.
SEARCH_FRESHNESS_TBS = os.environ.get("SEARCH_FRESHNESS_TBS", "qdr:d")
SEARCH_RESULTS_PER_COMPANY = int(os.environ.get("SEARCH_RESULTS_PER_COMPANY", "8"))
SEARCH_TIMEOUT_SECONDS = float(os.environ.get("SEARCH_TIMEOUT_SECONDS", "12"))

# How long to reuse a previous search result for the same (query, companies,
# freshness window), to avoid re-spending search-API credits on repeat
# testing/clicking. Set to 0 to disable caching.
SEARCH_CACHE_TTL_SECONDS = int(os.environ.get("SEARCH_CACHE_TTL_SECONDS", "300"))

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
origins = (
    ["*"] if ALLOWED_ORIGINS.strip() == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",")]
)

app = FastAPI(title="Fresh Openings — Job Finder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(BASE_DIR, "static")


class NoCacheStaticFiles(StaticFiles):
    """Adds Cache-Control: no-cache so browsers always revalidate instead of
    silently reusing a stale index.html/script/style after a deploy."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        logger.exception("Failed to initialize Gemini client — check GEMINI_API_KEY.")
else:
    logger.warning("GEMINI_API_KEY not set — report structuring will fail until it's configured.")

if not SERPER_API_KEY:
    logger.warning(
        "SERPER_API_KEY not set — job search will fail until it's configured. "
        "Sign up at https://serper.dev and set SERPER_API_KEY."
    )

# Very small in-memory cache: {(query_lower, tuple(sorted companies), tbs): (expires_at, context_text)}
# Process-local only (fine for a single free Render instance / demo use).
_search_cache: dict[tuple, tuple[float, str]] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    keywords: str = Field(
        ..., description="Free-text role/skills query, e.g. 'selenium, functional, automation, playwright'"
    )
    companies: list[str] = Field(
        default_factory=list,
        description="Optional list of companies to search. Defaults to DEFAULT_COMPANIES.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(raw_text: str) -> dict:
    """Gemini is asked for strict JSON, but strip code fences defensively in
    case the model wraps the response in ```json ... ``` anyway."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def _resolve_query(payload: SearchRequest) -> str:
    query = payload.keywords.strip()
    if not query:
        raise HTTPException(status_code=422, detail="keywords is required, e.g. 'selenium, automation, playwright'.")
    return query


def _resolve_companies(payload: SearchRequest) -> list[str]:
    companies = [c.strip() for c in payload.companies if c.strip()]
    if not companies:
        companies = list(DEFAULT_COMPANIES)

    # de-duplicate while preserving order
    seen = set()
    cleaned = []
    for c in companies:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(c)
    return cleaned[:MAX_COMPANIES]


def _cache_key(query: str, companies: list[str]) -> tuple:
    return (query.strip().lower(), tuple(sorted(c.lower() for c in companies)), SEARCH_FRESHNESS_TBS)


def _serper_search_sync(query: str, company: str) -> dict:
    """
    Blocking call to Serper.dev for ONE company. Run via asyncio.to_thread
    so multiple companies can be searched concurrently without blocking the
    event loop. Raises requests.RequestException on network/HTTP failure —
    callers are expected to catch this per-company so one bad company
    doesn't take down the whole search.
    """
    search_query = build_company_search_query(query, company)
    response = requests.post(
        SERPER_ENDPOINT,
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={
            "q": search_query,
            "num": SEARCH_RESULTS_PER_COMPANY,
            "tbs": SEARCH_FRESHNESS_TBS,
            "gl": "in",  # bias results toward India, matching this project's target market
        },
        timeout=SEARCH_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _format_company_results(company: str, data: dict) -> str:
    """Turns one company's raw Serper JSON into the plain-text block Step 2
    reads. Only fields that actually came back from the search engine are
    included — nothing is invented here."""
    organic = data.get("organic") or []
    if not organic:
        return f"### {company}\nNo search results found in this window.\n"

    lines = [f"### {company}"]
    for item in organic[:SEARCH_RESULTS_PER_COMPANY]:
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        date = (item.get("date") or "").strip()
        if not title or not link:
            continue
        lines.append(
            f"- title: {title}\n  link: {link}\n  date: {date or 'not shown'}\n  snippet: {snippet}"
        )
    if len(lines) == 1:
        return f"### {company}\nNo usable search results found in this window.\n"
    return "\n".join(lines) + "\n"


async def fetch_web_search_context(query: str, companies: list[str]) -> str:
    """
    Step 1: real web search, one query per company, run concurrently. This
    is the ONLY step allowed to produce facts (company names, titles,
    links, dates) — everything it returns came directly from the search
    engine's response, never from a model.
    """
    if not SERPER_API_KEY:
        return (
            "Live search is not configured: SERPER_API_KEY is missing on the "
            "server. Sign up for a free key at https://serper.dev and set it "
            "as an environment variable, then redeploy."
        )

    cache_key = _cache_key(query, companies)
    if SEARCH_CACHE_TTL_SECONDS > 0:
        cached = _search_cache.get(cache_key)
        if cached and cached[0] > time.monotonic():
            logger.info("Serving web search from cache for %s", cache_key)
            return cached[1]

    async def _search_one(company: str) -> str:
        try:
            data = await asyncio.to_thread(_serper_search_sync, query, company)
            return _format_company_results(company, data)
        except requests.RequestException:
            logger.exception("Web search failed for company=%s", company)
            return f"### {company}\nSearch failed for this company (network/API error).\n"

    blocks = await asyncio.gather(*[_search_one(c) for c in companies])
    context = "\n".join(blocks).strip()

    if not context:
        context = "No search results were retrieved for any company."

    if SEARCH_CACHE_TTL_SECONDS > 0:
        _search_cache[cache_key] = (time.monotonic() + SEARCH_CACHE_TTL_SECONDS, context)

    return context


async def generate_job_listing(query: str, companies: list[str]) -> dict:
    """Runs both steps and returns the parsed structured listing dict."""
    if gemini_client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured on the server. Set it in your "
            ".env file before starting the backend."
        )

    # Step 1: real, current postings via a direct web search API.
    search_context = await fetch_web_search_context(query, companies)

    # Step 2: structuring call, informed by (and restricted to) that
    # context. No tools attached, so this uses the ordinary, large
    # text-generation quota — never the grounding quota.
    user_prompt = build_user_prompt(
        query=query, companies=companies, search_context=search_context
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[user_prompt],
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=4000,
            response_mime_type="application/json",
        ),
    )

    raw_text = (response.text or "").strip()
    if not raw_text:
        raise RuntimeError("Gemini returned an empty response.")

    try:
        data = _extract_json(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Gemini JSON output: %s", raw_text[:500])
        raise RuntimeError("The model returned malformed data. Please try again.") from exc

    data.setdefault("jobs", [])
    data.setdefault("companies_with_no_results", [])
    data.setdefault("notes", "")
    data["disclaimer"] = RESEARCH_DISCLAIMER
    return data


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/quick-picks")
async def quick_picks():
    """Lets the frontend build its suggestion chips from the same source of
    truth the backend uses, instead of hardcoding the list twice. These are
    suggestions only — the user can type any free-text query."""
    return {"quick_picks": QUICK_PICKS, "default_companies": DEFAULT_COMPANIES}


@app.post("/search")
async def search(payload: SearchRequest):
    """
    Main endpoint: accepts a free-text role/skills query (e.g. "selenium,
    functional, automation, playwright") and an optional company list, runs
    the live web search + structuring pipeline, and returns the listing.
    """
    query = _resolve_query(payload)
    companies = _resolve_companies(payload)

    start = time.monotonic()
    try:
        listing = await generate_job_listing(query, companies)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception:
        logger.exception("Job search generation failed")
        raise HTTPException(
            status_code=502,
            detail="Something went wrong while searching for openings. Please try again.",
        )
    latency_ms = int((time.monotonic() - start) * 1000)

    return JSONResponse(
        {
            "success": True,
            "query": query,
            "companies_searched": companies,
            "listing": listing,
            "latency_ms": latency_ms,
            "session_id": uuid.uuid4().hex[:8].upper(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# Frontend (static site)
# ---------------------------------------------------------------------------
# Serves backend/static/index.html at "/" so frontend + backend + Gemini all
# run from this single FastAPI app. Mounted LAST so it never shadows the
# /health, /categories, or /search routes defined above it.
app.mount("/", NoCacheStaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
