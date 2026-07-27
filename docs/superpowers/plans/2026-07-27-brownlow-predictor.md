# Brownlow Predictor 2026 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Model note:** per the user's stated preference, use the Opus model for all coding/implementation work on this plan.

**Goal:** Build a pipeline that scrapes historical + weekly AFL player stats and Brownlow votes, trains a LightGBM ranking model, and publishes a live-updating top-20 Brownlow leaderboard for the 2026 season as a static GitHub Pages site.

**Architecture:** A small `brownlow/` Python package holds scraping, joining, modeling, and dashboard-rendering logic as pure, independently testable functions. Three thin CLI scripts (`backfill_data.py`, `train_model.py`, `weekly_update.py`) orchestrate that package for one-time backfill, one-time training, and the recurring GitHub Actions cron job, respectively — mirroring the sibling `afl-tipster` project's single-repo, static-site-plus-cron pattern.

**Tech Stack:** Python 3.11, `requests`, `beautifulsoup4` + `lxml`, `pandas`, `pyarrow` (Parquet I/O), `lightgbm`, `pytest`. No JS framework — the dashboard is a hand-rendered static `index.html`.

## Global Constraints

- Training window: seasons 2012–present (first season with all 12 afltables-verified advanced-stat columns populated).
- Feature set is exactly these 14 columns, in this order, everywhere they appear in code — `kicks, handballs, disposals, marks, goals, behinds, hitouts, tackles, clearances, contested_possessions, contested_marks, goal_assists, score_involvements, intercepts`. Kick-ins, intercept marks (specifically), and spoils are out of scope for v1 (documented gap in the spec).
- `brownlow_votes` (0-3 int) is the training label, sourced from afltables' `BR` column only.
- All HTTP fetches must rate-limit politely (minimum 1 second between requests to the same host) and set a descriptive `User-Agent` — no formal ToS exists for afltables.com/footywire.com, but treat them like any third-party site the sibling `afl-tipster` project depends on.
- Player names must be normalized to canonical `"F. Surname"` form (see Task 6) before any cross-source join or lookup — this is the only reliable join key between afltables and footywire naming conventions.
- No network calls in tests — every parser/harvester is tested against the static fixture files in `tests/fixtures/`, which are real (not fabricated) HTML captured from live pages and already committed to the repo.
- Follow `afl-tipster`'s file convention: flat top-level scripts for entry points, a small internal package for shared logic, single static `index.html` for the UI.

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `brownlow/__init__.py`
- Create: `CLAUDE.md`
- Test: none (scaffolding only)

**Interfaces:**
- Produces: the `brownlow` package import root that every later task's modules live under.

- [ ] **Step 1: Create `requirements.txt`**

```
requests==2.32.3
beautifulsoup4==4.12.3
lxml==5.3.0
pandas==2.2.3
pyarrow==17.0.0
lightgbm==4.5.0
pytest==8.3.3
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.venv/
*.egg-info/
```

- [ ] **Step 3: Create `brownlow/__init__.py`** (empty package marker)

```python
```

- [ ] **Step 4: Create `CLAUDE.md`**

```markdown
# Brownlow Predictor 2026 — Project Context

## Overview

Predicts the top 20 finishers in the 2026 AFL Brownlow Medal count. A
LightGBM ranking model trained on 2012+ historical data, scored weekly
against the current season and published as a static leaderboard —
sibling project to `afl-tipster`, same single-repo + GitHub Actions
cron pattern.

## Files

- `brownlow/` — scraping, joining, modeling, and dashboard logic
- `backfill_data.py` — one-time historical data scrape → `data/training_data.parquet`
- `train_model.py` — trains and backtests the model → `model.txt`
- `weekly_update.py` — GitHub Actions cron entry point; scores the latest
  round and regenerates `index.html`

## Data Sources

| Data | Source | Notes |
|------|--------|-------|
| Brownlow votes + 12 stats | afltables.com match pages | `afl/stats/games/{year}/{id}.html`, enumerated via `afl/brownlow/brownlow{year}rbr.html` |
| Score involvements + intercepts | footywire.com advanced stats | `ft_match_statistics?mid=...&advv=Y`, match IDs enumerated via `ft_match_list?year=...` |

See `docs/superpowers/specs/2026-07-27-brownlow-predictor-design.md` for
full data source research and rationale.

## Design docs

- Spec: `docs/superpowers/specs/2026-07-27-brownlow-predictor-design.md`
- Plan: `docs/superpowers/plans/2026-07-27-brownlow-predictor.md`
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore brownlow/__init__.py CLAUDE.md
git commit -m "Scaffold brownlow-predictor-2026 project"
```

---

### Task 2: HTTP fetch helper

**Files:**
- Create: `brownlow/http.py`
- Test: `tests/test_http.py`

**Interfaces:**
- Produces: `fetch_url(url: str, min_interval_seconds: float = 1.0) -> str` — used by every harvester/scraper task as the sole network entry point, so it's the only thing that needs mocking in higher-level tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_http.py
import time
from unittest.mock import patch, MagicMock
from brownlow.http import fetch_url


def test_fetch_url_returns_response_text():
    mock_response = MagicMock()
    mock_response.text = "<html>ok</html>"
    mock_response.raise_for_status = MagicMock()
    with patch("brownlow.http.requests.get", return_value=mock_response) as mock_get:
        result = fetch_url("https://example.com/page.html")
    assert result == "<html>ok</html>"
    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs["headers"]["User-Agent"]


def test_fetch_url_rate_limits_same_host():
    mock_response = MagicMock()
    mock_response.text = "<html>ok</html>"
    mock_response.raise_for_status = MagicMock()
    with patch("brownlow.http.requests.get", return_value=mock_response):
        with patch("brownlow.http.time.sleep") as mock_sleep:
            fetch_url("https://example.com/a.html", min_interval_seconds=2.0)
            fetch_url("https://example.com/b.html", min_interval_seconds=2.0)
    mock_sleep.assert_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_http.py -v`
Expected: FAIL with "No module named 'brownlow.http'"

- [ ] **Step 3: Write minimal implementation**

```python
# brownlow/http.py
import time
from urllib.parse import urlparse

import requests

_USER_AGENT = "brownlow-predictor-2026/1.0 (+https://github.com/nq2bui/browlow-predictor-2026)"
_last_request_time: dict[str, float] = {}


def fetch_url(url: str, min_interval_seconds: float = 1.0) -> str:
    host = urlparse(url).netloc
    last = _last_request_time.get(host)
    if last is not None:
        elapsed = time.time() - last
        if elapsed < min_interval_seconds:
            time.sleep(min_interval_seconds - elapsed)
    response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
    response.raise_for_status()
    _last_request_time[host] = time.time()
    return response.text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_http.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add brownlow/http.py tests/test_http.py
git commit -m "Add rate-limited HTTP fetch helper"
```

---

### Task 3: afltables match-page parser

**Files:**
- Create: `brownlow/afltables.py`
- Test: `tests/test_afltables.py`
- Uses fixture: `tests/fixtures/afltables_match_sample.html` (real page: Richmond v Carlton, Round 1 2023, trimmed to 3 players per team — already in the repo)

**Interfaces:**
- Produces: `parse_match_header(html: str) -> dict` with keys `round: str, date: str ("YYYY-MM-DD"), home_team: str, away_team: str`.
- Produces: `parse_match_page(html: str) -> list[dict]` — one dict per player with keys `team: str, player: str` (raw `"Surname, First"` form — normalization happens in Task 6) plus the 12 afltables stat columns (`kicks, handballs, disposals, marks, goals, behinds, hitouts, tackles, clearances, contested_possessions, contested_marks, goal_assists` — each `int`, `0` if the cell was blank) and `brownlow_votes: int` (`0` if blank).
- Consumes: nothing (pure HTML-string-in, dict-out functions).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_afltables.py
from pathlib import Path
from brownlow.afltables import parse_match_header, parse_match_page

FIXTURE = Path("tests/fixtures/afltables_match_sample.html").read_text()


def test_parse_match_header():
    header = parse_match_header(FIXTURE)
    assert header == {
        "round": "1",
        "date": "2023-03-16",
        "home_team": "Richmond",
        "away_team": "Carlton",
    }


def test_parse_match_page_returns_rows_for_both_teams():
    rows = parse_match_page(FIXTURE)
    assert len(rows) == 5  # 3 Richmond + 2 Carlton in the trimmed fixture
    teams = {row["team"] for row in rows}
    assert teams == {"Richmond", "Carlton"}


def test_parse_match_page_stat_values_are_correct():
    rows = parse_match_page(FIXTURE)
    bolton = next(r for r in rows if r["player"] == "Bolton, Shai")
    assert bolton["team"] == "Richmond"
    assert bolton["kicks"] == 15
    assert bolton["handballs"] == 3
    assert bolton["disposals"] == 18
    assert bolton["marks"] == 6
    assert bolton["goals"] == 1
    assert bolton["behinds"] == 1
    assert bolton["hitouts"] == 0
    assert bolton["tackles"] == 4
    assert bolton["clearances"] == 3
    assert bolton["contested_possessions"] == 8
    assert bolton["contested_marks"] == 1
    assert bolton["goal_assists"] == 1
    assert bolton["brownlow_votes"] == 3


def test_parse_match_page_blank_cells_become_zero():
    rows = parse_match_page(FIXTURE)
    baker = next(r for r in rows if r["player"] == "Baker, Liam")
    assert baker["brownlow_votes"] == 0
    assert baker["hitouts"] == 0
    assert baker["clearances"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_afltables.py -v`
Expected: FAIL with "No module named 'brownlow.afltables'"

- [ ] **Step 3: Write minimal implementation**

```python
# brownlow/afltables.py
import re
from datetime import datetime

from bs4 import BeautifulSoup

_COLUMN_TO_FIELD = {
    "KI": "kicks",
    "HB": "handballs",
    "DI": "disposals",
    "MK": "marks",
    "GL": "goals",
    "BH": "behinds",
    "HO": "hitouts",
    "TK": "tackles",
    "CL": "clearances",
    "CP": "contested_possessions",
    "CM": "contested_marks",
    "GA": "goal_assists",
    "BR": "brownlow_votes",
}

SEASON_INDEX_URL_TEMPLATE = "https://afltables.com/afl/brownlow/brownlow{year}rbr.html"


def parse_match_header(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    summary_table = soup.find("table")
    text = summary_table.get_text(" ", strip=True)

    round_match = re.search(r"Round:\s*(\S+)", text)
    date_match = re.search(r"Date:\s*\w+,\s*(\d{1,2}-\w{3}-\d{4})", text)
    parsed_date = datetime.strptime(date_match.group(1), "%d-%b-%Y").strftime("%Y-%m-%d")

    team_links = [
        a.get_text(strip=True)
        for a in summary_table.find_all("a")
        if "teams/" in a.get("href", "")
    ]

    return {
        "round": round_match.group(1),
        "date": parsed_date,
        "home_team": team_links[0],
        "away_team": team_links[1],
    }


def parse_match_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for table in soup.find_all("table", class_="sortable"):
        header_th = table.find("th", colspan="25")
        team_name = header_th.get_text(strip=True).split(" Match Statistics")[0]
        header_cells = [th.get_text(strip=True) for th in table.find_all("tr")[1].find_all("th")]

        for tr in table.find("tbody").find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            raw_row = dict(zip(header_cells, cells))

            parsed = {"team": team_name, "player": raw_row["Player"]}
            for abbr, field in _COLUMN_TO_FIELD.items():
                value = raw_row.get(abbr, "")
                parsed[field] = int(value) if value.isdigit() else 0
            rows.append(parsed)
    return rows


def list_season_match_urls(index_html: str) -> list[str]:
    hrefs = re.findall(r'href="(\.\./stats/games/[^"]+)"', index_html)
    seen = []
    for href in hrefs:
        path = href.split("stats/games/")[1]
        url = f"https://afltables.com/afl/stats/games/{path}"
        if url not in seen:
            seen.append(url)
    return seen
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_afltables.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add brownlow/afltables.py tests/test_afltables.py
git commit -m "Add afltables match-page and header parser"
```

---

### Task 4: afltables season match-URL harvester test

**Files:**
- Modify: `tests/test_afltables.py`
- Uses fixture: `tests/fixtures/afltables_season_index_sample.html` (already in repo)

**Interfaces:**
- Consumes: `list_season_match_urls` from Task 3 (already implemented there since it's a small, related function — this task only adds its test coverage using the season-index fixture).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_afltables.py
from brownlow.afltables import list_season_match_urls

SEASON_INDEX_FIXTURE = Path("tests/fixtures/afltables_season_index_sample.html").read_text()


def test_list_season_match_urls():
    urls = list_season_match_urls(SEASON_INDEX_FIXTURE)
    assert urls == [
        "https://afltables.com/afl/stats/games/2023/031420230316.html",
        "https://afltables.com/afl/stats/games/2023/040920230317.html",
        "https://afltables.com/afl/stats/games/2023/121820230318.html",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_afltables.py::test_list_season_match_urls -v`
Expected: FAIL — actually this should PASS immediately since `list_season_match_urls` was already implemented in Task 3. Run it to confirm: if it unexpectedly fails, the Task 3 implementation has a bug — fix `list_season_match_urls` in `brownlow/afltables.py` before proceeding.

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_afltables.py -v`
Expected: PASS (5 passed)

- [ ] **Step 4: Commit**

```bash
git add tests/test_afltables.py
git commit -m "Add season match-URL harvester test coverage"
```

---

### Task 5: footywire advanced-stats parser

**Files:**
- Create: `brownlow/footywire.py`
- Test: `tests/test_footywire.py`
- Uses fixtures: `tests/fixtures/footywire_advanced_sample.html`, `tests/fixtures/footywire_match_list_sample.html` (both already in repo, real captured pages)

**Interfaces:**
- Produces: `parse_advanced_stats_page(html: str) -> list[dict]` — one dict per player with keys `team: str, player: str` (raw `"F Surname"` form) plus `score_involvements: int, intercepts: int`.
- Produces: `list_season_match_ids(match_list_html: str) -> list[dict]` — one dict per match with keys `mid: int, home_team: str, away_team: str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_footywire.py
from pathlib import Path
from brownlow.footywire import parse_advanced_stats_page, list_season_match_ids

ADV_FIXTURE = Path("tests/fixtures/footywire_advanced_sample.html").read_text()
MATCH_LIST_FIXTURE = Path("tests/fixtures/footywire_match_list_sample.html").read_text()


def test_parse_advanced_stats_page():
    rows = parse_advanced_stats_page(ADV_FIXTURE)
    assert len(rows) == 6  # 3 Sydney + 3 Geelong in the fixture
    teams = {row["team"] for row in rows}
    assert teams == {"Sydney", "Geelong"}

    florent = next(r for r in rows if r["player"] == "O Florent")
    assert florent["team"] == "Sydney"
    assert florent["score_involvements"] == 7
    assert florent["intercepts"] == 2

    selwood = next(r for r in rows if r["player"] == "J Selwood")
    assert selwood["team"] == "Geelong"
    assert selwood["score_involvements"] == 9
    assert selwood["intercepts"] == 7


def test_list_season_match_ids():
    matches = list_season_match_ids(MATCH_LIST_FIXTURE)
    assert matches == [
        {"mid": 10751, "home_team": "Richmond", "away_team": "Carlton"},
        {"mid": 10752, "home_team": "Collingwood", "away_team": "Sydney"},
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_footywire.py -v`
Expected: FAIL with "No module named 'brownlow.footywire'"

- [ ] **Step 3: Write minimal implementation**

```python
# brownlow/footywire.py
import re

from bs4 import BeautifulSoup

MATCH_STATS_URL_TEMPLATE = "https://www.footywire.com/afl/footy/ft_match_statistics?mid={mid}&advv=Y"
SEASON_MATCH_LIST_URL_TEMPLATE = "https://www.footywire.com/afl/footy/ft_match_list?year={year}"


def parse_advanced_stats_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    teams = []
    for title_td in soup.find_all("td", class_="innertbtitle"):
        text = title_td.get_text(strip=True)
        if "Match Statistics" in text:
            teams.append(text.split(" Match Statistics")[0])

    candidate_tables = [t for t in soup.find_all("table") if t.find("span", attrs={"title": "Player"})]
    stats_tables = [
        t for t in candidate_tables
        if not any(t is not other and other in t.find_all("table") for other in candidate_tables)
    ]

    rows = []
    for team, table in zip(teams, stats_tables):
        headers = [s.get_text(strip=True) for s in table.find_all("span", class_="sortByAjaxLink")]
        si_index = headers.index("SI") - 1  # -1 because headers includes "Player" but tds after the player cell don't
        itc_index = headers.index("ITC") - 1

        for tr in table.find_all("tr", class_=["darkcolor", "lightcolor"]):
            tds = tr.find_all("td")
            player = tds[0].get_text(strip=True)
            values = [td.get_text(strip=True) for td in tds[1:]]
            rows.append({
                "team": team,
                "player": player,
                "score_involvements": int(values[si_index]) if values[si_index].replace(".", "").isdigit() else 0,
                "intercepts": int(values[itc_index]) if values[itc_index].replace(".", "").isdigit() else 0,
            })
    return rows


def list_season_match_ids(match_list_html: str) -> list[dict]:
    soup = BeautifulSoup(match_list_html, "html.parser")
    matches = []
    for tr in soup.find_all("tr", class_=["darkcolor", "lightcolor"]):
        tds = tr.find_all("td")
        team_links = tds[1].find_all("a")
        if len(team_links) != 2:
            continue
        mid_link = tds[3].find("a")
        if mid_link is None:
            continue
        mid_match = re.search(r"mid=(\d+)", mid_link["href"])
        if mid_match is None:
            continue
        matches.append({
            "mid": int(mid_match.group(1)),
            "home_team": team_links[0].get_text(strip=True),
            "away_team": team_links[1].get_text(strip=True),
        })
    return matches
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_footywire.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add brownlow/footywire.py tests/test_footywire.py
git commit -m "Add footywire advanced-stats parser and match-list harvester"
```

---

### Task 6: Player name normalization

**Files:**
- Create: `brownlow/names.py`
- Test: `tests/test_names.py`

**Interfaces:**
- Produces: `normalize_player_name(raw: str) -> str` — used by Task 7's join logic to reconcile afltables' `"Surname, First"` form with footywire's `"F Surname"` form into one canonical `"F. Surname"` key.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_names.py
from brownlow.names import normalize_player_name


def test_normalize_afltables_style_name():
    assert normalize_player_name("Baker, Liam") == "L. Baker"
    assert normalize_player_name("Bolton, Shai") == "S. Bolton"


def test_normalize_footywire_style_name():
    assert normalize_player_name("O Florent") == "O. Florent"
    assert normalize_player_name("J Selwood") == "J. Selwood"


def test_normalize_multi_word_surname():
    assert normalize_player_name("Van Berlo, Jack") == "J. Van Berlo"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_names.py -v`
Expected: FAIL with "No module named 'brownlow.names'"

- [ ] **Step 3: Write minimal implementation**

```python
# brownlow/names.py
def normalize_player_name(raw: str) -> str:
    raw = raw.strip()
    if "," in raw:
        surname, first = (part.strip() for part in raw.split(",", 1))
        first_initial = first[0] if first else ""
    else:
        parts = raw.split()
        first_initial = parts[0][0] if parts else ""
        surname = " ".join(parts[1:]) if len(parts) > 1 else raw
    return f"{first_initial}. {surname}".strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_names.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add brownlow/names.py tests/test_names.py
git commit -m "Add player name normalization for cross-source joins"
```

---

### Task 7: Dataset assembly and cross-source join

**Files:**
- Create: `brownlow/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `parse_match_header`, `parse_match_page` (Task 3); `parse_advanced_stats_page` (Task 5); `normalize_player_name` (Task 6).
- Produces: `STAT_COLUMNS: list[str]` (the 14 canonical feature names, in order) — used by Task 9 (training) and Task 12 (weekly scoring).
- Produces: `assemble_match_records(season: int, match_id: str, afltables_html: str, footywire_html: str | None) -> list[dict]` — returns one dict per player with keys `season, round, date, match_id, team, player` (normalized) plus all of `STAT_COLUMNS` (int, `0` if unavailable) and `brownlow_votes` (int).
- Produces: `rows_to_dataframe(rows: list[dict]) -> "pandas.DataFrame"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
from pathlib import Path
from brownlow.dataset import assemble_match_records, rows_to_dataframe, STAT_COLUMNS

AFLTABLES_FIXTURE = Path("tests/fixtures/afltables_match_sample.html").read_text()
FOOTYWIRE_FIXTURE = Path("tests/fixtures/footywire_advanced_sample.html").read_text()


def test_stat_columns_has_14_entries():
    assert len(STAT_COLUMNS) == 14
    assert STAT_COLUMNS[:8] == [
        "kicks", "handballs", "disposals", "marks",
        "goals", "behinds", "hitouts", "tackles",
    ]
    assert STAT_COLUMNS[-2:] == ["score_involvements", "intercepts"]


def test_assemble_match_records_joins_footywire_stats():
    records = assemble_match_records(
        season=2023,
        match_id="031420230316",
        afltables_html=AFLTABLES_FIXTURE,
        footywire_html=FOOTYWIRE_FIXTURE,  # Sydney v Geelong fixture, doesn't share players
    )
    assert len(records) == 5
    bolton = next(r for r in records if r["player"] == "S. Bolton")
    assert bolton["season"] == 2023
    assert bolton["round"] == "1"
    assert bolton["date"] == "2023-03-16"
    assert bolton["match_id"] == "031420230316"
    assert bolton["team"] == "Richmond"
    assert bolton["kicks"] == 15
    assert bolton["brownlow_votes"] == 3
    # no matching footywire player in this fixture -> defaults to 0, not a crash
    assert bolton["score_involvements"] == 0
    assert bolton["intercepts"] == 0


def test_assemble_match_records_without_footywire_data():
    records = assemble_match_records(
        season=2023,
        match_id="031420230316",
        afltables_html=AFLTABLES_FIXTURE,
        footywire_html=None,
    )
    assert len(records) == 5
    assert all(r["score_involvements"] == 0 and r["intercepts"] == 0 for r in records)


def test_rows_to_dataframe_has_expected_columns():
    records = assemble_match_records(2023, "031420230316", AFLTABLES_FIXTURE, None)
    df = rows_to_dataframe(records)
    for col in STAT_COLUMNS + ["season", "round", "date", "match_id", "team", "player", "brownlow_votes"]:
        assert col in df.columns
    assert len(df) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL with "No module named 'brownlow.dataset'"

- [ ] **Step 3: Write minimal implementation**

```python
# brownlow/dataset.py
import pandas as pd

from brownlow.afltables import parse_match_header, parse_match_page
from brownlow.footywire import parse_advanced_stats_page
from brownlow.names import normalize_player_name

STAT_COLUMNS = [
    "kicks", "handballs", "disposals", "marks", "goals", "behinds", "hitouts", "tackles",
    "clearances", "contested_possessions", "contested_marks", "goal_assists",
    "score_involvements", "intercepts",
]

_FOOTYWIRE_ONLY_COLUMNS = ("score_involvements", "intercepts")


def assemble_match_records(
    season: int,
    match_id: str,
    afltables_html: str,
    footywire_html: str | None,
) -> list[dict]:
    header = parse_match_header(afltables_html)
    afltables_rows = parse_match_page(afltables_html)

    footywire_lookup = {}
    if footywire_html:
        for row in parse_advanced_stats_page(footywire_html):
            key = (row["team"], normalize_player_name(row["player"]))
            footywire_lookup[key] = row

    records = []
    for row in afltables_rows:
        normalized_name = normalize_player_name(row["player"])
        footywire_row = footywire_lookup.get((row["team"], normalized_name))

        record = {
            "season": season,
            "round": header["round"],
            "date": header["date"],
            "match_id": match_id,
            "team": row["team"],
            "player": normalized_name,
            "brownlow_votes": row["brownlow_votes"],
        }
        for col in STAT_COLUMNS:
            if col in _FOOTYWIRE_ONLY_COLUMNS:
                record[col] = footywire_row[col] if footywire_row else 0
            else:
                record[col] = row[col]
        records.append(record)
    return records


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add brownlow/dataset.py tests/test_dataset.py
git commit -m "Add dataset assembly and afltables/footywire join logic"
```

---

### Task 8: Unmatched-name logging

**Files:**
- Modify: `brownlow/dataset.py`
- Modify: `tests/test_dataset.py`

**Interfaces:**
- Consumes/modifies: `assemble_match_records` from Task 7 — adds logging when a footywire row can't be matched to any afltables player in the same match, per the spec's error-handling requirement ("unmatched names are flagged to a log rather than silently dropped").

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dataset.py
import logging


def test_unmatched_footywire_player_is_logged(caplog):
    footywire_html_with_extra_player = FOOTYWIRE_FIXTURE.replace(
        'title="Oliver Florent">O Florent</a>',
        'title="Oliver Florent">O Florento</a>',  # simulate a name mismatch
    )
    with caplog.at_level(logging.WARNING):
        assemble_match_records(2023, "031420230316", AFLTABLES_FIXTURE, footywire_html_with_extra_player)
    assert any("O. Florento" in record.message for record in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py::test_unmatched_footywire_player_is_logged -v`
Expected: FAIL — no warning logged yet (current implementation only looks up matches from the afltables side, never checks for footywire rows with no afltables counterpart)

- [ ] **Step 3: Write minimal implementation**

```python
# brownlow/dataset.py — add near the top
import logging

logger = logging.getLogger(__name__)
```

```python
# brownlow/dataset.py — modify assemble_match_records body, after building afltables_rows'
# normalized-name set and before the return statement
def assemble_match_records(
    season: int,
    match_id: str,
    afltables_html: str,
    footywire_html: str | None,
) -> list[dict]:
    header = parse_match_header(afltables_html)
    afltables_rows = parse_match_page(afltables_html)

    footywire_lookup = {}
    if footywire_html:
        for row in parse_advanced_stats_page(footywire_html):
            key = (row["team"], normalize_player_name(row["player"]))
            footywire_lookup[key] = row

    afltables_keys = {(row["team"], normalize_player_name(row["player"])) for row in afltables_rows}
    for key in footywire_lookup:
        if key not in afltables_keys:
            logger.warning("footywire player %s (%s) not found in afltables match %s", key[1], key[0], match_id)

    records = []
    for row in afltables_rows:
        normalized_name = normalize_player_name(row["player"])
        footywire_row = footywire_lookup.get((row["team"], normalized_name))

        record = {
            "season": season,
            "round": header["round"],
            "date": header["date"],
            "match_id": match_id,
            "team": row["team"],
            "player": normalized_name,
            "brownlow_votes": row["brownlow_votes"],
        }
        for col in STAT_COLUMNS:
            if col in _FOOTYWIRE_ONLY_COLUMNS:
                record[col] = footywire_row[col] if footywire_row else 0
            else:
                record[col] = row[col]
        records.append(record)
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add brownlow/dataset.py tests/test_dataset.py
git commit -m "Log unmatched footywire players instead of silently dropping them"
```

---

### Task 9: `backfill_data.py` CLI

**Files:**
- Create: `backfill_data.py`
- Test: `tests/test_backfill_data.py`

**Interfaces:**
- Consumes: `fetch_url` (Task 2); `list_season_match_urls`, `parse_match_header` (Task 3); `list_season_match_ids`, `MATCH_STATS_URL_TEMPLATE`, `SEASON_MATCH_LIST_URL_TEMPLATE` (Task 5); `assemble_match_records`, `rows_to_dataframe`, `STAT_COLUMNS` (Task 7); afltables' `SEASON_INDEX_URL_TEMPLATE` (Task 3).
- Produces: `backfill_seasons(start_season: int, end_season: int, fetch=fetch_url) -> "pandas.DataFrame"` (the `fetch` parameter defaults to the real network fetcher but accepts an injected fake for testing — this is the seam that keeps this test network-free) and a CLI (`python backfill_data.py --start-season 2012 --end-season 2025 --output data/training_data.parquet`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backfill_data.py
from pathlib import Path
from backfill_data import backfill_seasons
from brownlow.afltables import SEASON_INDEX_URL_TEMPLATE
from brownlow.footywire import SEASON_MATCH_LIST_URL_TEMPLATE, MATCH_STATS_URL_TEMPLATE

AFLTABLES_MATCH = Path("tests/fixtures/afltables_match_sample.html").read_text()
AFLTABLES_INDEX = Path("tests/fixtures/afltables_season_index_sample.html").read_text()
FOOTYWIRE_ADV = Path("tests/fixtures/footywire_advanced_sample.html").read_text()
FOOTYWIRE_MATCH_LIST = Path("tests/fixtures/footywire_match_list_sample.html").read_text()

MATCH_URL = "https://afltables.com/afl/stats/games/2023/031420230316.html"


def fake_fetch(url: str) -> str:
    if url == SEASON_INDEX_URL_TEMPLATE.format(year=2023):
        return AFLTABLES_INDEX
    if url == MATCH_URL:
        return AFLTABLES_MATCH
    if url == SEASON_MATCH_LIST_URL_TEMPLATE.format(year=2023):
        return FOOTYWIRE_MATCH_LIST
    if url == MATCH_STATS_URL_TEMPLATE.format(mid=10751):
        return FOOTYWIRE_ADV
    raise AssertionError(f"unexpected URL fetched in test: {url}")


def test_backfill_seasons_builds_combined_dataframe():
    df = backfill_seasons(2023, 2023, fetch=fake_fetch)
    # 3 afltables match URLs in the index fixture; only the first (031420230316)
    # has a fake_fetch response wired up for its match page, the others 404 in
    # fake_fetch and should be skipped with a warning, not crash the run
    assert len(df) == 5  # 5 players in the one successfully-fetched match
    assert set(df["match_id"]) == {"031420230316"}
    assert df["season"].iloc[0] == 2023
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backfill_data.py -v`
Expected: FAIL with "No module named 'backfill_data'"

- [ ] **Step 3: Write minimal implementation**

```python
# backfill_data.py
import argparse
import logging
import re

import pandas as pd

from brownlow.afltables import SEASON_INDEX_URL_TEMPLATE, list_season_match_urls, parse_match_header
from brownlow.footywire import (
    SEASON_MATCH_LIST_URL_TEMPLATE,
    MATCH_STATS_URL_TEMPLATE,
    list_season_match_ids,
)
from brownlow.dataset import assemble_match_records
from brownlow.http import fetch_url

logger = logging.getLogger(__name__)


def _match_id_from_url(url: str) -> str:
    return re.search(r"/(\w+)\.html$", url).group(1)


def backfill_seasons(start_season: int, end_season: int, fetch=fetch_url) -> pd.DataFrame:
    all_records = []
    for season in range(start_season, end_season + 1):
        try:
            index_html = fetch(SEASON_INDEX_URL_TEMPLATE.format(year=season))
        except Exception:
            logger.warning("could not fetch season index for %s, skipping season", season)
            continue

        match_urls = list_season_match_urls(index_html)

        try:
            footywire_matches = list_season_match_ids(
                fetch(SEASON_MATCH_LIST_URL_TEMPLATE.format(year=season))
            )
        except Exception:
            logger.warning("could not fetch footywire match list for %s, continuing without it", season)
            footywire_matches = []
        footywire_html_by_teams = {}
        for match in footywire_matches:
            try:
                footywire_html_by_teams[(match["home_team"], match["away_team"])] = fetch(
                    MATCH_STATS_URL_TEMPLATE.format(mid=match["mid"])
                )
            except Exception:
                logger.warning("could not fetch footywire stats for mid=%s", match["mid"])

        for match_url in match_urls:
            match_id = _match_id_from_url(match_url)
            try:
                afltables_html = fetch(match_url)
            except Exception:
                logger.warning("could not fetch afltables match %s, skipping", match_url)
                continue

            header = parse_match_header(afltables_html)
            footywire_html = footywire_html_by_teams.get((header["home_team"], header["away_team"]))

            all_records.extend(
                assemble_match_records(season, match_id, afltables_html, footywire_html)
            )
    return pd.DataFrame(all_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    df = backfill_seasons(args.start_season, args.end_season)
    df.to_parquet(args.output, index=False)
    logger.info("wrote %d rows to %s", len(df), args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backfill_data.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add backfill_data.py tests/test_backfill_data.py
git commit -m "Add backfill_data.py CLI orchestrating full historical scrape"
```

---

### Task 10: Model training

**Files:**
- Create: `brownlow/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `STAT_COLUMNS` (Task 7).
- Produces: `train_ranker(df: "pandas.DataFrame") -> "lightgbm.LGBMRanker"`, `save_model(model, path: str) -> None`, `load_model(path: str) -> "lightgbm.LGBMRanker"`, `predict_match_votes(model, df: "pandas.DataFrame") -> "pandas.Series"` — used by Task 11 (backtest) and Task 12 (weekly scoring).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import pandas as pd
from brownlow.dataset import STAT_COLUMNS
from brownlow.model import train_ranker, save_model, load_model, predict_match_votes


def _fake_training_data() -> pd.DataFrame:
    rows = []
    for match_id in ["m1", "m2", "m3"]:
        for i, votes in enumerate([3, 2, 1, 0]):
            row = {col: (i + 1) * 5 for col in STAT_COLUMNS}
            row["match_id"] = match_id
            row["brownlow_votes"] = votes
            row["player"] = f"{match_id}-player{i}"
            rows.append(row)
    return pd.DataFrame(rows)


def test_train_ranker_returns_fitted_model():
    df = _fake_training_data()
    model = train_ranker(df)
    predictions = predict_match_votes(model, df[df["match_id"] == "m1"])
    assert len(predictions) == 4


def test_save_and_load_model_roundtrip(tmp_path):
    df = _fake_training_data()
    model = train_ranker(df)
    model_path = str(tmp_path / "model.txt")
    save_model(model, model_path)
    loaded = load_model(model_path)
    predictions = predict_match_votes(loaded, df[df["match_id"] == "m1"])
    assert len(predictions) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with "No module named 'brownlow.model'"

- [ ] **Step 3: Write minimal implementation**

```python
# brownlow/model.py
import lightgbm as lgb
import pandas as pd

from brownlow.dataset import STAT_COLUMNS


def train_ranker(df: pd.DataFrame) -> lgb.LGBMRanker:
    df_sorted = df.sort_values("match_id").reset_index(drop=True)
    group_sizes = df_sorted.groupby("match_id", sort=False).size().tolist()

    model = lgb.LGBMRanker(objective="lambdarank", min_child_samples=1)
    model.fit(
        df_sorted[STAT_COLUMNS],
        df_sorted["brownlow_votes"],
        group=group_sizes,
    )
    return model


def save_model(model: lgb.LGBMRanker, path: str) -> None:
    model.booster_.save_model(path)


def load_model(path: str) -> lgb.Booster:
    return lgb.Booster(model_file=path)


def predict_match_votes(model, df: pd.DataFrame) -> pd.Series:
    predictions = model.predict(df[STAT_COLUMNS])
    return pd.Series(predictions, index=df.index)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add brownlow/model.py tests/test_model.py
git commit -m "Add LightGBM ranker training, save/load, and scoring"
```

---

### Task 11: Backtest evaluation and `train_model.py` CLI

**Files:**
- Create: `brownlow/backtest.py`
- Create: `train_model.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `train_ranker`, `predict_match_votes` (Task 10); `STAT_COLUMNS` (Task 7).
- Produces: `top20_hit_rate(model, season_df: "pandas.DataFrame") -> float` — fraction of the season's actual top-20 vote-getters that also appear in the model's predicted top 20 for that season.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest.py
import pandas as pd
from brownlow.dataset import STAT_COLUMNS
from brownlow.model import train_ranker
from brownlow.backtest import top20_hit_rate


def _season_df_with_known_top_scorer() -> pd.DataFrame:
    rows = []
    for match_num in range(25):
        match_id = f"m{match_num}"
        # "Star" gets 3 votes almost every match with high stats; others vary
        star_row = {col: 20 for col in STAT_COLUMNS}
        star_row.update({"match_id": match_id, "brownlow_votes": 3, "player": "Star"})
        others_row = {col: 5 for col in STAT_COLUMNS}
        others_row.update({"match_id": match_id, "brownlow_votes": 0, "player": f"Other{match_num}"})
        rows.extend([star_row, others_row])
    return pd.DataFrame(rows)


def test_top20_hit_rate_finds_the_obvious_top_scorer():
    df = _season_df_with_known_top_scorer()
    model = train_ranker(df)
    hit_rate = top20_hit_rate(model, df)
    assert 0.0 <= hit_rate <= 1.0
    assert hit_rate > 0  # "Star" should be found since it's the only real vote-getter
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest.py -v`
Expected: FAIL with "No module named 'brownlow.backtest'"

- [ ] **Step 3: Write minimal implementation**

```python
# brownlow/backtest.py
import pandas as pd

from brownlow.model import predict_match_votes


def top20_hit_rate(model, season_df: pd.DataFrame) -> float:
    df = season_df.copy()
    df["predicted_votes"] = predict_match_votes(model, df)

    actual_totals = df.groupby("player")["brownlow_votes"].sum().sort_values(ascending=False)
    predicted_totals = df.groupby("player")["predicted_votes"].sum().sort_values(ascending=False)

    actual_top20 = set(actual_totals.head(20).index)
    predicted_top20 = set(predicted_totals.head(20).index)

    if not actual_top20:
        return 0.0
    return len(actual_top20 & predicted_top20) / len(actual_top20)
```

```python
# train_model.py
import argparse
import logging

import pandas as pd

from brownlow.model import train_ranker, save_model
from brownlow.backtest import top20_hit_rate

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model-out", type=str, required=True)
    parser.add_argument("--holdout-seasons", type=int, nargs="+", default=[2024, 2025])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    df = pd.read_parquet(args.data)

    train_df = df[~df["season"].isin(args.holdout_seasons)]
    model = train_ranker(train_df)

    for season in args.holdout_seasons:
        season_df = df[df["season"] == season]
        if season_df.empty:
            continue
        hit_rate = top20_hit_rate(model, season_df)
        logger.info("season %s top-20 hit rate: %.2f", season, hit_rate)

    final_model = train_ranker(df)
    save_model(final_model, args.model_out)
    logger.info("saved model to %s", args.model_out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add brownlow/backtest.py train_model.py tests/test_backtest.py
git commit -m "Add backtest evaluation and train_model.py CLI"
```

---

### Task 12: Weekly season-to-date scoring

**Files:**
- Create: `brownlow/weekly.py`
- Test: `tests/test_weekly.py`

**Interfaces:**
- Consumes: `predict_match_votes` (Task 10); `STAT_COLUMNS` (Task 7).
- Produces: `accumulate_season_votes(model, season_df: "pandas.DataFrame") -> "pandas.DataFrame"` — returns a DataFrame with columns `player, team, predicted_season_votes`, sorted descending, used by Task 13's dashboard renderer and by `weekly_update.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_weekly.py
import pandas as pd
from brownlow.dataset import STAT_COLUMNS
from brownlow.model import train_ranker
from brownlow.weekly import accumulate_season_votes


def _season_df() -> pd.DataFrame:
    rows = []
    for match_num in range(3):
        match_id = f"m{match_num}"
        star_row = {col: 20 for col in STAT_COLUMNS}
        star_row.update({"match_id": match_id, "brownlow_votes": 3, "player": "Star", "team": "Richmond"})
        other_row = {col: 5 for col in STAT_COLUMNS}
        other_row.update({"match_id": match_id, "brownlow_votes": 0, "player": "Other", "team": "Carlton"})
        rows.extend([star_row, other_row])
    return pd.DataFrame(rows)


def test_accumulate_season_votes_ranks_players_by_total_predicted_votes():
    df = _season_df()
    model = train_ranker(df)
    leaderboard = accumulate_season_votes(model, df)

    assert list(leaderboard.columns) == ["player", "team", "predicted_season_votes"]
    assert leaderboard.iloc[0]["player"] == "Star"
    assert leaderboard["predicted_season_votes"].is_monotonic_decreasing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_weekly.py -v`
Expected: FAIL with "No module named 'brownlow.weekly'"

- [ ] **Step 3: Write minimal implementation**

```python
# brownlow/weekly.py
import pandas as pd

from brownlow.model import predict_match_votes


def accumulate_season_votes(model, season_df: pd.DataFrame) -> pd.DataFrame:
    df = season_df.copy()
    df["predicted_votes"] = predict_match_votes(model, df)

    leaderboard = (
        df.groupby(["player", "team"])["predicted_votes"]
        .sum()
        .reset_index()
        .rename(columns={"predicted_votes": "predicted_season_votes"})
        .sort_values("predicted_season_votes", ascending=False)
        .reset_index(drop=True)
    )
    return leaderboard[["player", "team", "predicted_season_votes"]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_weekly.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add brownlow/weekly.py tests/test_weekly.py
git commit -m "Add season-to-date vote accumulation for the weekly leaderboard"
```

---

### Task 13: Dashboard renderer

**Files:**
- Create: `brownlow/dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: the `leaderboard` DataFrame shape produced by Task 12's `accumulate_season_votes` (`player, team, predicted_season_votes` columns).
- Produces: `render_leaderboard(leaderboard: "pandas.DataFrame", output_path: str) -> None` — writes a static `index.html`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard.py
import pandas as pd
from brownlow.dashboard import render_leaderboard


def test_render_leaderboard_writes_top_20_table(tmp_path):
    leaderboard = pd.DataFrame({
        "player": [f"Player{i}" for i in range(25)],
        "team": ["Richmond"] * 25,
        "predicted_season_votes": [25 - i for i in range(25)],
    })
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, str(output_path))

    html = output_path.read_text()
    assert "Player0" in html
    assert "Player19" in html
    assert "Player20" not in html  # only top 20 shown
    assert "<table" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard.py -v`
Expected: FAIL with "No module named 'brownlow.dashboard'"

- [ ] **Step 3: Write minimal implementation**

```python
# brownlow/dashboard.py
from datetime import datetime, timezone

import pandas as pd

_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Brownlow Predictor 2026</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #ddd; }}
  th {{ color: #555; font-weight: 600; }}
  .rank {{ color: #999; width: 32px; }}
  .votes {{ text-align: right; font-variant-numeric: tabular-nums; }}
  footer {{ margin-top: 24px; color: #999; font-size: 13px; }}
</style>
</head>
<body>
<h1>Brownlow Predictor 2026</h1>
<p>Predicted top 20, updated after each round.</p>
<table>
<thead><tr><th class="rank">#</th><th>Player</th><th>Team</th><th class="votes">Predicted votes</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
<footer>Last updated {timestamp}</footer>
</body>
</html>
"""

_ROW_TEMPLATE = '<tr><td class="rank">{rank}</td><td>{player}</td><td>{team}</td><td class="votes">{votes:.1f}</td></tr>'


def render_leaderboard(leaderboard: pd.DataFrame, output_path: str) -> None:
    top20 = leaderboard.head(20)
    rows_html = "\n".join(
        _ROW_TEMPLATE.format(rank=i + 1, player=row.player, team=row.team, votes=row.predicted_season_votes)
        for i, row in enumerate(top20.itertuples())
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = _PAGE_TEMPLATE.format(rows=rows_html, timestamp=timestamp)
    with open(output_path, "w") as f:
        f.write(html)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dashboard.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add brownlow/dashboard.py tests/test_dashboard.py
git commit -m "Add static leaderboard dashboard renderer"
```

---

### Task 14: `weekly_update.py` CLI and GitHub Actions cron

**Files:**
- Create: `weekly_update.py`
- Create: `.github/workflows/weekly_update.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `list_season_match_urls`, `SEASON_INDEX_URL_TEMPLATE`, `parse_match_header` (Task 3); `list_season_match_ids`, `SEASON_MATCH_LIST_URL_TEMPLATE`, `MATCH_STATS_URL_TEMPLATE` (Task 5); `assemble_match_records` (Task 7); `load_model` (Task 10); `accumulate_season_votes` (Task 12); `render_leaderboard` (Task 13).
- No test file — this script is a thin orchestrator wired entirely from already-tested functions (each of which has its own unit tests from Tasks 3, 5, 7, 10, 12, 13); its correctness is exercised by running it against real 2026 data once the season starts, same as `afl-tipster`'s `update_simulator.py` has no dedicated test suite either.

- [ ] **Step 1: Write `weekly_update.py`**

```python
# weekly_update.py
import logging

import pandas as pd

from brownlow.afltables import SEASON_INDEX_URL_TEMPLATE, list_season_match_urls, parse_match_header
from brownlow.footywire import SEASON_MATCH_LIST_URL_TEMPLATE, MATCH_STATS_URL_TEMPLATE, list_season_match_ids
from brownlow.dataset import assemble_match_records
from brownlow.model import load_model
from brownlow.weekly import accumulate_season_votes
from brownlow.dashboard import render_leaderboard
from brownlow.http import fetch_url

logger = logging.getLogger(__name__)

CURRENT_SEASON = 2026
MODEL_PATH = "model.txt"
OUTPUT_PATH = "index.html"


def build_current_season_dataframe(season: int, fetch=fetch_url) -> pd.DataFrame:
    try:
        index_html = fetch(SEASON_INDEX_URL_TEMPLATE.format(year=season))
    except Exception:
        logger.error("could not fetch season index for %s, aborting weekly update", season)
        return pd.DataFrame()

    match_urls = list_season_match_urls(index_html)

    try:
        footywire_matches = list_season_match_ids(fetch(SEASON_MATCH_LIST_URL_TEMPLATE.format(year=season)))
    except Exception:
        logger.warning("could not fetch footywire match list for %s, continuing without it", season)
        footywire_matches = []
    footywire_html_by_teams = {}
    for match in footywire_matches:
        try:
            footywire_html_by_teams[(match["home_team"], match["away_team"])] = fetch(
                MATCH_STATS_URL_TEMPLATE.format(mid=match["mid"])
            )
        except Exception:
            logger.warning("could not fetch footywire stats for mid=%s", match["mid"])

    all_records = []
    for match_url in match_urls:
        match_id = match_url.rsplit("/", 1)[1].replace(".html", "")
        try:
            afltables_html = fetch(match_url)
        except Exception:
            logger.warning("could not fetch afltables match %s, skipping", match_url)
            continue
        header = parse_match_header(afltables_html)
        footywire_html = footywire_html_by_teams.get((header["home_team"], header["away_team"]))
        all_records.extend(assemble_match_records(season, match_id, afltables_html, footywire_html))

    return pd.DataFrame(all_records)


def main():
    logging.basicConfig(level=logging.INFO)

    season_df = build_current_season_dataframe(CURRENT_SEASON)
    if season_df.empty:
        logger.error("no 2026 match data available yet, leaving existing index.html untouched")
        return

    model = load_model(MODEL_PATH)
    leaderboard = accumulate_season_votes(model, season_df)
    render_leaderboard(leaderboard, OUTPUT_PATH)
    logger.info("wrote updated leaderboard to %s (%d players)", OUTPUT_PATH, len(leaderboard))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `.github/workflows/weekly_update.yml`**

```yaml
name: Update Brownlow Predictor Weekly

on:
  schedule:
    # Tuesday 9am AEST (11pm UTC Monday) — after all weekend/Monday matches are final on both sources
    - cron: '0 23 * * 1'
  workflow_dispatch:

permissions:
  contents: write

jobs:
  update-leaderboard:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4.3.0

      - name: Set up Python
        uses: actions/setup-python@v5.6.0
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run weekly update
        run: python weekly_update.py

      - name: Commit and push changes
        run: |
          git config user.name "Brownlow Predictor Bot"
          git config user.email "bot@browlow-predictor-2026"
          git add index.html
          git diff --staged --quiet || git commit -m "Auto-update: leaderboard $(date +'%d %b %Y')"
          git pull --no-rebase -X ours origin main --no-edit
          git push origin main
```

- [ ] **Step 3: Update `README.md`**

```markdown
# Brownlow Predictor 2026

Predicts the top 20 players to finish highest in the 2026 AFL Brownlow
Medal count, updated weekly via a LightGBM ranking model trained on
2012–2025 match data.

See `CLAUDE.md` for project structure and data sources, and
`docs/superpowers/specs/2026-07-27-brownlow-predictor-design.md` for the
full design rationale.

## Running locally

```bash
pip install -r requirements.txt
python backfill_data.py --start-season 2012 --end-season 2025 --output data/training_data.parquet
python train_model.py --data data/training_data.parquet --model-out model.txt
python weekly_update.py
```
```

- [ ] **Step 4: Verify the whole test suite still passes**

Run: `pytest -v`
Expected: PASS (all tests from Tasks 1–13)

- [ ] **Step 5: Commit**

```bash
git add weekly_update.py .github/workflows/weekly_update.yml README.md
git commit -m "Add weekly_update.py CLI and GitHub Actions cron"
```
