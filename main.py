"""
main.py

FastAPI backend for the "Fresh Openings" job finder demo.

Responsibilities:
- Accept a JSON request with a category/role keyword and (optionally) a
  list of target companies.
- Run a web-search-grounded Gemini call to find CURRENT, open postings for
  that category at those companies (real search, not model memory).
- Call Gemini again (no tools) to reshape that raw research into strict
  JSON the frontend can render as a list of cards.
- Return the structured listing to the frontend.

This intentionally follows the same two-step "grounded search -> strict
JSON structuring" pattern as the reference stock-research project: one call
is responsible for facts (with Google Search grounding), the other is
responsible for shape only, and is explicitly told not to invent data.

Honest limitation: this is a demo/POC. Google Search grounding is a fast
way to prototype "find me fresh jobs," but it's less reliable than pulling
directly from each company's own ATS API or career page (see the
discussion in README.md). Don't treat this as a production data pipeline.

Quota note (read this if /search keeps returning "no postings"):
Grounding with Google Search has its OWN quota on the Gemini API, separate
from (and much smaller than) the normal plain-text generation quota. A
free/low-tier key can exhaust the grounding quota after a handful of
searches even though ordinary (non-grounded) generate_content calls keep
working fine. That's a real account/billing limit, not something any
client-side code change can bypass — check https://ai.dev/usage and
https://ai.google.dev/gemini-api/docs/rate-limits for your key's current
grounding quota. What this file DOES do to cope with that:
  1. Retries a 429 on the grounded call a couple of times with backoff,
     which helps with short per-minute limits.
  2. Surfaces a specific "grounding quota exceeded" message instead of a
     generic "search tool error", so it's obvious what's going on.
  3. Caches identical (query, companies) searches in memory for a few
     minutes, so repeated testing/clicking doesn't burn quota you don't
     have to spend.
"""

import os
import re
import json
import time
import uuid
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors

try:
    # Works when run from inside backend/ (uvicorn main:app)
    from prompt import (
        QUICK_PICKS,
        DEFAULT_COMPANIES,
        MAX_COMPANIES,
        RESEARCH_DISCLAIMER,
        GROUNDED_SEARCH_SYSTEM_PROMPT,
        SYSTEM_PROMPT,
        build_grounded_search_prompt,
        build_user_prompt,
    )
except ImportError:
    # Works when run from the repo root (uvicorn backend.main:app)
    from backend.prompt import (
        QUICK_PICKS,
        DEFAULT_COMPANIES,
        MAX_COMPANIES,
        RESEARCH_DISCLAIMER,
        GROUNDED_SEARCH_SYSTEM_PROMPT,
        SYSTEM_PROMPT,
        build_grounded_search_prompt,
        build_user_prompt,
    )

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("job-finder")

BASE_DIR = os.path.dirname(__file__)

# --- Gemini config ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# "gemini-3.6-flash" is not a real model name — use a real, current one.
# NOTE: if GEMINI_MODEL / GEMINI_SEARCH_MODEL are set as Render environment
# variables, those override this default entirely. Check your Render
# service's Environment tab if you still see an unexpected model in logs.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_SEARCH_MODEL = os.environ.get("GEMINI_SEARCH_MODEL", GEMINI_MODEL)

# How many times to retry the grounded search call if Gemini returns a 429,
# and how long to wait between attempts (seconds). Short/cheap by design —
# this is meant to smooth over brief per-minute rate limits, not to wait out
# a fully exhausted daily/monthly grounding quota.
GROUNDING_RETRY_ATTEMPTS = int(os.environ.get("GROUNDING_RETRY_ATTEMPTS", "2"))
GROUNDING_RETRY_DELAY_SECONDS = float(os.environ.get("GROUNDING_RETRY_DELAY_SECONDS", "3"))

# How long to reuse a previous grounded search result for the same
# (query, companies) pair, to avoid re-spending grounding quota on repeat
# testing. Set to 0 to disable caching.
GROUNDING_CACHE_TTL_SECONDS = int(os.environ.get("GROUNDING_CACHE_TTL_SECONDS", "300"))

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
    logger.warning("GEMINI_API_KEY not set — job search will fail until it's configured.")

# Very small in-memory cache: {(query_lower, tuple(sorted companies)): (expires_at, text)}
# Process-local only (fine for a single free Render instance / demo use).
_grounded_cache: dict[tuple, tuple[float, str]] = {}


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


def _is_quota_exhausted_error(exc: Exception) -> bool:
    """True for a 429 RESOURCE_EXHAUSTED from the Gemini API specifically."""
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    # Fall back to string sniffing in case the SDK version doesn't expose
    # status_code on this exception type.
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def _cache_key(query: str, companies: list[str]) -> tuple:
    return (query.strip().lower(), tuple(sorted(c.lower() for c in companies)))


async def fetch_grounded_jobs(query: str, companies: list[str]) -> str:
    """
    Step 1: a SEPARATE Gemini call with Google Search grounding enabled.
    This is the only step allowed to produce facts (company names, titles,
    links, dates).

    Grounding has its own, much smaller quota than plain generate_content
    calls — a 429 here does NOT mean the API key or the rest of the app is
    broken; it means the grounding-specific quota is temporarily or fully
    exhausted. See the module docstring for details.
    """
    if gemini_client is None:
        return "Live search unavailable: Gemini is not configured."

    cache_key = _cache_key(query, companies)
    if GROUNDING_CACHE_TTL_SECONDS > 0:
        cached = _grounded_cache.get(cache_key)
        if cached and cached[0] > time.monotonic():
            logger.info("Serving grounded search from cache for %s", cache_key)
            return cached[1]

    last_error: Exception | None = None
    for attempt in range(1, GROUNDING_RETRY_ATTEMPTS + 2):  # +1 initial try, +N retries
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_SEARCH_MODEL,
                contents=[build_grounded_search_prompt(query, companies)],
                config=genai_types.GenerateContentConfig(
                    system_instruction=GROUNDED_SEARCH_SYSTEM_PROMPT,
                    max_output_tokens=3500,
                    tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                ),
            )
            text = (response.text or "").strip()
            if not text:
                return "Live search returned no results for this category/company list."

            if GROUNDING_CACHE_TTL_SECONDS > 0:
                _grounded_cache[cache_key] = (
                    time.monotonic() + GROUNDING_CACHE_TTL_SECONDS,
                    text,
                )
            return text

        except genai_errors.ClientError as exc:
            last_error = exc
            if _is_quota_exhausted_error(exc) and attempt <= GROUNDING_RETRY_ATTEMPTS:
                logger.warning(
                    "Grounded search hit 429 (attempt %s/%s), retrying in %ss",
                    attempt, GROUNDING_RETRY_ATTEMPTS + 1, GROUNDING_RETRY_DELAY_SECONDS,
                )
                time.sleep(GROUNDING_RETRY_DELAY_SECONDS)
                continue
            break
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a best-effort step
            last_error = exc
            break

    if last_error is not None and _is_quota_exhausted_error(last_error):
        logger.error("Grounded search (Google Search tool) quota exhausted: %s", last_error)
        return (
            "Live search is temporarily unavailable: the Google Search grounding "
            "quota for this Gemini API key has been used up (this is a separate, "
            "smaller quota than normal text generation). It will reset on Google's "
            "usual schedule, or you can raise it by enabling billing on the "
            "project — see https://ai.google.dev/gemini-api/docs/rate-limits. "
            "No postings could be verified this time."
        )

    logger.exception("Grounded search (Google Search tool) failed", exc_info=last_error)
    return (
        "Live search was unavailable for this request (search tool error). "
        "No postings could be verified."
    )


async def generate_job_listing(query: str, companies: list[str]) -> dict:
    """Runs both steps and returns the parsed structured listing dict."""
    if gemini_client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured on the server. Set it in your "
            ".env file before starting the backend."
        )

    # Step 1: real, current postings via Google Search grounding.
    grounded_context = await fetch_grounded_jobs(query, companies)

    # Step 2: structuring call, informed by (and restricted to) that context.
    # No tools attached here, so this call is NOT subject to the grounding
    # quota — it uses the ordinary, much larger plain-text quota.
    user_prompt = build_user_prompt(
        query=query, companies=companies, grounded_context=grounded_context
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
    the grounded search + structuring pipeline, and returns the listing.
    """
    query = _resolve_query(payload)
    companies = _resolve_companies(payload)

    start = time.monotonic()
    try:
        listing = await generate_job_listing(query, companies)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception:
        logger.exception("Gemini generation failed")
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
