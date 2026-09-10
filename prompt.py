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

# (id, display_label) — the category presets shown in the frontend dropdown.
# "custom" lets the user type any keyword instead (e.g. "embedded systems").
CATEGORIES = [
    ("testing", "Software Testing / QA"),
    ("software_dev", "Software Development"),
    ("data", "Data / Analytics"),
    ("devops", "DevOps / Cloud / SRE"),
    ("support", "Technical / IT Support"),
    ("design", "UI/UX Design"),
    ("product", "Product Management"),
    ("custom", "Custom keyword"),
]
CATEGORY_LABELS = {cid: label for cid, label in CATEGORIES}

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
with live web search access. Given a job category/role keyword and a list
of target companies, search for CURRENT, OPEN job postings matching that
role at those companies.

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


def build_grounded_search_prompt(category_label: str, companies: list[str]) -> str:
    company_list = ", ".join(companies)
    return (
        f"Search for current open job postings for the role/category "
        f'"{category_label}" at each of these companies: {company_list}.\n\n'
        f"For each company, find as many genuinely open postings in this "
        f"category as you can (aim for up to 3 per company), following the "
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

USER_PROMPT_TEMPLATE = """CATEGORY REQUESTED: {category_label}
COMPANIES SEARCHED: {companies}

GROUNDED_CONTEXT (from a real web search performed just now — this is your
ONLY source of facts; do not add anything not present here):
---
{grounded_context}
---

Reformat the above into the required JSON object now. Return JSON only.
"""


def build_user_prompt(category_label: str, companies: list[str], grounded_context: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        category_label=category_label,
        companies=", ".join(companies),
        grounded_context=grounded_context or "No grounded context was retrieved for this request.",
    )
