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
# FIX: the previous default here was "gemini-3.6-flash", which is not a
# real Gemini model name. That caused every generate_content() call to
# fail (model not found) whenever GEMINI_MODEL wasn't explicitly set in
# the environment, which in turn made fetch_grounded_jobs() silently fall
# back to its "search unavailable" string. Use a real, current model.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_SEARCH_MODEL = os.environ.get("GEMINI_SEARCH_MODEL", GEMINI_MODEL)

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


async def fetch_grounded_jobs(query: str, companies: list[str]) -> str:
    """
    Step 1: a SEPARATE Gemini call with Google Search grounding enabled.
    This is the only step allowed to produce facts (company names, titles,
    links, dates). Best-effort: if grounding fails, returns a clear
    fallback string so the structuring step reports the failure honestly
    instead of the caller crashing.
    """
    if gemini_client is None:
        return "Live search unavailable: Gemini is not configured."

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
        return text
    except Exception:
        logger.exception("Grounded search (Google Search tool) failed")
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
