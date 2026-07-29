"""Sportsbet Brownlow Medal odds: live fetch + pure HTML parsing.

The odds page is a JS-rendered SPA, so a plain ``requests.get`` (what the rest
of this project uses) returns an empty shell with no odds data. ``fetch_odds_page_html``
drives a real headless browser via Playwright to render it. ``parse_brownlow_odds``
is a pure BeautifulSoup parser over the rendered HTML and is the unit-tested part
(the live fetcher is not unit-tested, consistent with ``brownlow/http.py``'s
``fetch_url``). Playwright is imported lazily inside the fetcher so the parser and
this module stay importable without the browser dependency installed.
"""

import re

from bs4 import BeautifulSoup

from brownlow.names import normalize_player_name

SPORTSBET_BROWNLOW_URL = (
    "https://www.sportsbet.com.au/betting/australian-rules/afl-brownlow-medal"
)

# The name and odds spans for one outcome share a numeric id prefix, e.g.
# data-automation-id="1232681781-list-outcome-name" pairs with
# data-automation-id="1232681781-list-outcome-text".
_NAME_ID_RE = re.compile(r"^(\d+)-list-outcome-name$")
_TEXT_ID_RE = re.compile(r"^(\d+)-list-outcome-text$")


def fetch_odds_page_html(url: str = SPORTSBET_BROWNLOW_URL) -> str:
    """Render the SPA with headless chromium and return its HTML.

    Uses ``wait_until="load"`` (``networkidle`` never settles on this page --
    it polls in the background -- and times out), then waits a few real seconds
    for the odds table to finish populating client-side before snapshotting.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, timeout=45000, wait_until="load")
            page.wait_for_timeout(5000)
            return page.content()
        finally:
            browser.close()


def parse_brownlow_odds(html: str) -> list[dict]:
    """Parse rendered Sportsbet HTML into ``[{"player", "decimal_odds"}, ...]``.

    Player names are normalized via ``normalize_player_name`` to the canonical
    "F. Surname" leaderboard join key. Name and odds spans are paired by their
    shared numeric id prefix, so an outcome that is missing either half (e.g. a
    truncated trailing entry) is dropped rather than mis-paired. Results are
    returned in document order (first appearance of each id).
    """
    soup = BeautifulSoup(html, "html.parser")

    names: dict[str, str] = {}
    odds: dict[str, float] = {}
    order: list[str] = []

    for span in soup.find_all("span", attrs={"data-automation-id": True}):
        auto_id = span["data-automation-id"]

        name_match = _NAME_ID_RE.match(auto_id)
        if name_match:
            outcome_id = name_match.group(1)
            names[outcome_id] = span.get_text(strip=True)
            if outcome_id not in order:
                order.append(outcome_id)
            continue

        text_match = _TEXT_ID_RE.match(auto_id)
        if text_match:
            outcome_id = text_match.group(1)
            raw = span.get_text(strip=True)
            try:
                odds[outcome_id] = float(raw)
            except ValueError:
                # Non-numeric odds text (e.g. a suspended market placeholder) --
                # skip rather than crash, matching the pipeline's default-on-
                # missing-data style.
                continue

    results = []
    for outcome_id in order:
        if outcome_id in names and outcome_id in odds:
            results.append(
                {
                    "player": normalize_player_name(names[outcome_id]),
                    "decimal_odds": odds[outcome_id],
                }
            )
    return results


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to an implied probability percentage.

    e.g. ``implied_probability(1.20) == 83.33...``. Rounding for display happens
    at the render site, not here.
    """
    return 1.0 / decimal_odds * 100.0
