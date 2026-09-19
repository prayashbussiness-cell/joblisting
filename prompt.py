"""
prompt.py

Holds the prompt templates + fixed category/seed-company config used to pull
fresh job openings for a given category (e.g. "Testing").

ARCHITECTURE CHANGE (see main.py docstring for the full story): fact-finding
is no longer done by Gemini's Google Search grounding tool. That tool has
its own separate, much smaller quota on the Gemini API than plain text
generation, and it was getting exhausted independently of everything else
working fine. Instead:

- Step 1 (in main.py) calls a real web search API (Serper.dev, wrapping
  Google Search) directly, once per target company, restricted to results
  from roughly the last 24 hours. This is the ONLY step allowed to produce
  facts — company names, job titles, links, dates — and every link it
  produces is a real URL that came back from the search engine, not
  something a model wrote.
- Step 2 (this file's SYSTEM_PROMPT) takes those raw search snippets and
  reshapes them into strict JSON so the frontend can render a clean list.
  It is explicitly forbidden from inventing or modifying a posting, link,
  or detail that wasn't in the search results — its job is formatting and
  classification only, same as before. Because this step has no tools
  attached, it only uses Gemini's ordinary text-generation quota, not the
  grounding quota — that's what makes it reliable.

Honest limitation: filtering to "posted in roughly the last 24 hours" is a
Google-search date filter (`tbs=qdr:d`), which depends on Google having
indexed a fresh crawl date for the page — it is a best-effort recency
signal, not a guarantee down to the hour. Some companies simply won't have
posted anything new in the last 24h, and that's an honest "no results"
rather than a bug. See README.md for more on this project's limitations.
"""

# Quick-pick suggestions shown as clickable chips in the frontend — clicking
# one fills the free-text keyword box, it doesn't lock the user into a fixed
# list. The user can type anything, including multiple comma-separated
# skills (e.g. "selenium, functional testing, automation, playwright").
QUICK_PICKS = [
    "Selenium, Functional Testing, Automation, Playwright",
    "Manual Testing, QA, Test Cases",
    "Software Development, Java, Python",
    "Data Analyst, SQL, Power BI",
    "DevOps, AWS, Kubernetes, CI/CD",
    "UI/UX Design, Figma",
]

# A starter list of companies to search across when the user doesn't supply
# their own. Deliberately mixes a couple of ATS-friendly companies with the
# classic Indian IT majors discussed in planning — edit freely for your own
# target list. This is NOT a claim that these are all "top 30 MNCs"; it's a
# small sample so the demo has something sensible to search by default.
DEFAULT_COMPANIES = [
    "TCS",
    "Infosys",
    "Wipro",
    "Cognizant",
    "HCLTech",
    "Capgemini",
    "Accenture",
    "IBM India",
    "Tech Mahindra",
    "Zoho",
]

MAX_COMPANIES = 12

RESEARCH_DISCLAIMER = (
    "Listings are pulled via live web search and may be incomplete or "
    "occasionally out of date. Always verify on the company's own career "
    "page before applying."
)

# ---------------------------------------------------------------------------
# Step 1: real web search query construction (Serper.dev / Google Search)
# This is the ONLY step that may introduce facts — see main.py's
# fetch_web_search_context(), which calls the search API with these queries.
# ---------------------------------------------------------------------------

def build_company_search_query(query: str, company: str) -> str:
    """
    Builds the literal search-engine query string for one company. Kept
    simple and literal on purpose — the search engine's own ranking does
    the heavy lifting, and Step 2 is told to discard anything irrelevant.
    """
    return f'{company} "{query}" jobs hiring'


# ---------------------------------------------------------------------------
# Step 2: structuring prompt (strict JSON, no tools, no new facts)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You reformat raw web search results into strict JSON for
a frontend to render. You do NOT have web access yourself and you must NOT
add, guess, or "fill in" any posting, company, link, or detail that is not
already present in the SEARCH_RESULTS you are given below. Your only job is
cleaning, classifying, and structuring what's already there.

CRITICAL RULE ON LINKS: every "apply_url" you output MUST be copied
character-for-character from a "link" field in SEARCH_RESULTS. Never
shorten, guess, autocomplete, or construct a URL yourself. If you cannot
find a genuine job-posting link for something, leave it out entirely rather
than inventing one.

Each result in SEARCH_RESULTS includes the company it was searched for, a
title, a link, a snippet, and (when available) a date. Only include a
posting if the title/snippet clearly indicates it is an actual open job
posting (not a company homepage, a "life at X" blog post, a news article
about layoffs, or an unrelated search result). Skip anything that isn't
clearly a job posting.

If SEARCH_RESULTS has no usable postings for a company, do not include a
row for that company — instead note it in "companies_with_no_results". This
is common and expected when a company simply hasn't posted anything new in
the search window — do not treat it as an error.

If SEARCH_RESULTS is empty, unavailable, or an error message for every
company, return an empty "jobs" array and explain why in "notes" — never
invent postings to fill the response.

Respond with STRICT JSON ONLY — no markdown code fences, no commentary
before or after, no trailing commas. The JSON must be a single object of
this exact shape:

{
  "category_label": "string, the human-readable category/role searched",
  "generated_at_note": "string, e.g. 'Live search results (last ~24h)' or a short note on data recency/limitations",
  "jobs": [
    {
      "company": "string",
      "title": "string, exact job title as found",
      "location": "string, e.g. 'Bengaluru' or 'Remote (India)' or 'Not specified'",
      "experience_level": "string, e.g. 'Fresher', '0-2 yrs', '3-5 yrs', 'Not specified'",
      "posted": "string, the date field from the search result if present, else 'Not specified'",
      "apply_url": "string, copied verbatim from a 'link' field in SEARCH_RESULTS",
      "source": "string, e.g. 'Company career page', 'Greenhouse board', 'LinkedIn Jobs', or the domain of the link",
      "is_direct_apply": true or false — true only if apply_url points to the company's own domain or an ATS board (Greenhouse/Lever/Ashby/Workday/SmartRecruiters/Naukri/etc.), false if it's a general aggregator or you're unsure,
      "summary": "string, one short line on the role, only if the snippet supports it, else empty string"
    }
  ],
  "companies_with_no_results": ["string", "..."],
  "notes": "string, any caveats worth surfacing to the user (e.g. 'no postings found in the last 24h for X', 'search was limited for Y')"
}

Do not editorialize about company quality or add recommendations — this is
a factual listing tool, not an advisory one.
"""

USER_PROMPT_TEMPLATE = """QUERY REQUESTED: {query}
COMPANIES SEARCHED: {companies}

SEARCH_RESULTS (from a real web search performed just now — this is your
ONLY source of facts and links; do not add anything not present here):
---
{search_context}
---

Reformat the above into the required JSON object now. Set "category_label"
to the query as given. Return JSON only.
"""


def build_user_prompt(query: str, companies: list[str], search_context: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        query=query,
        companies=", ".join(companies),
        search_context=search_context or "No search results were retrieved for this request.",
    )
