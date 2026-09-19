"""
prompt.py

Holds the prompt templates + fixed category/seed-company config used to pull
fresh job openings for a given category (e.g. "Testing") via Gemini, using
the same two-step pattern as the stock research terminal this project is
modeled on:

Design notes (mirrors the reference project's approach):
- Step 1 is a SEPARATE Gemini call with Google Search grounding enabled.
  It does the actual "go find current job postings" work. This is the ONLY
  step allowed to produce facts — company names, job titles, links, dates.
- Step 2 takes that raw grounded text and reshapes it into strict JSON so
  the frontend can render a clean list. Step 2 is explicitly forbidden from
  inventing or adding any posting that wasn't in the Step 1 output — its
  job is formatting/classification, not sourcing.
- This mirrors why the reference project splits mutual-fund/news data (must
  be grounded) from the rest of the report (can use model knowledge): here,
  ALL of the output must be grounded, since a job listing is worthless if
  it's not real.

Honest limitation (see README): Google Search grounding is a fast way to
prototype this, but it is inherently less reliable than pulling directly
from each company's official ATS API (Greenhouse/Lever/etc.) or career
page, because the model can still misread a stale search snippet. Treat
this as a demo/POC, not a production data pipeline — see README.md.
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
    "Listings are pulled via AI web search and may be incomplete, delayed, "
    "or occasionally inaccurate. Always verify on the company's own career "
    "page before applying."
)

# ---------------------------------------------------------------------------
# Step 1: web-search-grounded prompt (separate call, Google Search tool)
# This is the ONLY step that may introduce facts.
# ---------------------------------------------------------------------------

GROUNDED_SEARCH_SYSTEM_PROMPT = """You are a job-search research assistant
with live web search access. Given a role/skills query and a list of target
companies, search for CURRENT, OPEN job postings matching that query at
those companies.

The query may be a single role title (e.g. "Software Testing / QA") OR a
comma-separated list of specific skills/tools (e.g. "selenium, functional
testing, automation, playwright"). When it's a skills list, treat it as
one combined profile — search for postings whose title or description
mentions ANY of those skills/tools, not only ones that mention all of them,
and prefer postings that match more of the listed skills over ones that
match only one.

Prioritize, in this order:
1. The company's own official careers page or ATS-hosted board (Greenhouse,
   Lever, Ashby, Workday, SmartRecruiters, or the company's own
   careers.[company].com domain).
2. The company's official LinkedIn Jobs listing for that specific posting.
3. Reputable listings only as a last resort — and if you use one, say so.

For EACH posting you find, report:
- Company name
- Exact job title
- Location (city/remote)
- Experience level if stated (e.g. Fresher, 0-2 yrs, 3-5 yrs, Senior)
- Posted date or "time ago" if the source shows one, else say "not shown"
- The DIRECT application URL (the actual posting page, not a homepage)
- Which source it came from (company career page / company ATS board /
  LinkedIn / other — name the actual source)

Do not invent, guess, or fill in a posting you are not reasonably confident
is currently live. If you find NO current postings for a company in this
category, say so explicitly for that company rather than fabricating one.
If a company's career site could not be searched (blocked, no results,
etc.), say so plainly.

Be concise and factual — this is raw research material for a formatting
step, not a final answer. Organize your findings company by company.
"""


def build_grounded_search_prompt(query: str, companies: list[str]) -> str:
    company_list = ", ".join(companies)
    return (
        f"Search for current open job postings matching this role/skills "
        f'query: "{query}", at each of these companies: {company_list}.\n\n'
        f"For each company, find as many genuinely open postings matching "
        f"the query as you can (aim for up to 3 per company), following the "
        f"source priority and reporting format in your instructions."
    )


# ---------------------------------------------------------------------------
# Step 2: structuring prompt (strict JSON, no tools, no new facts)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You reformat raw job-search research notes into strict
JSON for a frontend to render. You do NOT have web access and you must NOT
add, guess, or "fill in" any posting, company, link, or detail that is not
already present in the GROUNDED_CONTEXT you are given below. Your only job
is cleaning, classifying, and structuring what's already there.

If GROUNDED_CONTEXT says no postings were found for a company, do not
include a row for that company — instead note it in "companies_with_no_results".

If GROUNDED_CONTEXT is empty, unavailable, or an error message, return an
empty "jobs" array and explain why in "notes" — never invent postings to
fill the response.

Respond with STRICT JSON ONLY — no markdown code fences, no commentary
before or after, no trailing commas. The JSON must be a single object of
this exact shape:

{
  "category_label": "string, the human-readable category/role searched",
  "generated_at_note": "string, e.g. 'Live search results' or a short note on data recency/limitations",
  "jobs": [
    {
      "company": "string",
      "title": "string, exact job title as found",
      "location": "string, e.g. 'Bengaluru' or 'Remote (India)' or 'Not specified'",
      "experience_level": "string, e.g. 'Fresher', '0-2 yrs', '3-5 yrs', 'Not specified'",
      "posted": "string, e.g. '2 days ago', '2026-09-05', or 'Not specified'",
      "apply_url": "string, the direct posting URL from the research notes",
      "source": "string, e.g. 'Company career page', 'Greenhouse board', 'LinkedIn Jobs'",
      "is_direct_apply": true or false — true only if apply_url points to the company's own domain or an ATS board (Greenhouse/Lever/Ashby/Workday/SmartRecruiters/etc.), false if it's a general aggregator or you're unsure,
      "summary": "string, one short line on the role, only if the research notes support it, else empty string"
    }
  ],
  "companies_with_no_results": ["string", "..."],
  "notes": "string, any caveats worth surfacing to the user (e.g. 'search was limited for X', 'no postings found in this category right now')"
}

Do not editorialize about company quality or add recommendations — this is
a factual listing tool, not an advisory one.
"""

USER_PROMPT_TEMPLATE = """QUERY REQUESTED: {query}
COMPANIES SEARCHED: {companies}

GROUNDED_CONTEXT (from a real web search performed just now — this is your
ONLY source of facts; do not add anything not present here):
---
{grounded_context}
---

Reformat the above into the required JSON object now. Set "category_label"
to the query as given. Return JSON only.
"""


def build_user_prompt(query: str, companies: list[str], grounded_context: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        query=query,
        companies=", ".join(companies),
        grounded_context=grounded_context or "No grounded context was retrieved for this request.",
    )
