from pathlib import Path

from brownlow.afltables import (
    list_season_match_urls,
    parse_match_header,
    parse_match_page,
)

FIXTURE = Path("tests/fixtures/afltables_match_sample.html").read_text()
SEASON_INDEX_FIXTURE = Path(
    "tests/fixtures/afltables_season_index_sample.html"
).read_text()
# Built programmatically from a real captured live match page (Adelaide v
# Hawthorn, round 22 2025) which contains FOUR sortable tables: two "Match
# Statistics" tables and two "Player Details" tables. See
# .superpowers/sdd/debug_real_2025_match.html.
MATCH_WITH_PLAYER_DETAILS_FIXTURE = Path(
    "tests/fixtures/afltables_match_with_player_details_sample.html"
).read_text()
# Built programmatically from a real fetch of the general season-results page
# https://afltables.com/afl/seas/2026.html (the in-progress 2026 season). Its
# per-match markup differs from the Brownlow round-by-round page, but carries
# the same href="../stats/games/..." links, so list_season_match_urls must
# handle it too -- this is the fallback source weekly_update.py/backfill_data.py
# use when the round-by-round page 404s mid-season.
SEASON_RESULTS_FIXTURE = Path(
    "tests/fixtures/afltables_season_results_sample.html"
).read_text()


def test_parse_match_header():
    header = parse_match_header(FIXTURE)
    assert header == {
        "round": "1",
        "date": "2023-03-16",
        "home_team": "Richmond",
        "away_team": "Carlton",
        # Richmond v Carlton R1 2023 was a real draw: both sides 8.10.58.
        "home_score": 58,
        "away_score": 58,
    }


def test_parse_match_header_extracts_home_and_away_scores():
    # Real captured page: Adelaide (home) v Hawthorn (away), R22 2025.
    # Adelaide's last quarter cell final total is 15.11.101, Hawthorn's 13.9.87.
    # Using a non-draw match proves home_score and away_score are read from the
    # correct rows (home != away), which the drawn fixture above cannot show.
    header = parse_match_header(MATCH_WITH_PLAYER_DETAILS_FIXTURE)
    assert header["home_team"] == "Adelaide"
    assert header["away_team"] == "Hawthorn"
    assert header["home_score"] == 101
    assert header["away_score"] == 87


def test_parse_match_page_returns_rows_for_both_teams():
    rows = parse_match_page(FIXTURE)
    assert len(rows) == 6  # 3 Richmond + 3 Carlton in the trimmed fixture
    # NOTE: brief asserted 5 ("2 Carlton"), but the real fixture actually
    # contains 3 Carlton players (Acres, Cerra, Cowan). Corrected to match
    # the real captured data; fixture left untouched. See task-3-report.md.
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


def test_parse_match_page_handles_non_25_colspan():
    # Robustness: an older/newer afltables season may have a different total
    # column count, so the team-name header th could carry any colspan (here
    # 20, not 25). Team name + stat header must still be located by content.
    html = """
    <html><body>
    <table class="sortable">
      <thead>
        <tr><th colspan=20>Geelong Match Statistics [<a href="#">Season</a>][<a href="#">Game by Game</a>]</th></tr>
        <tr><th>#</th><th>Player</th><th>KI</th><th>MK</th><th>HB</th><th>DI</th><th>GL</th><th>BR</th></tr>
      </thead>
      <tbody>
        <tr><td>1</td><td>Dangerfield, Patrick</td><td>12</td><td>4</td><td>8</td><td>20</td><td>2</td><td>3</td></tr>
      </tbody>
    </table>
    </body></html>
    """
    rows = parse_match_page(html)
    assert len(rows) == 1
    row = rows[0]
    assert row["team"] == "Geelong"
    assert row["player"] == "Dangerfield, Patrick"
    assert row["kicks"] == 12
    assert row["handballs"] == 8
    assert row["disposals"] == 20
    assert row["goals"] == 2
    assert row["brownlow_votes"] == 3


def test_parse_match_page_skips_player_details_table():
    # Real live afltables match pages carry "Player Details" sortable tables
    # (career-bio columns) alongside the "Match Statistics" tables. Those have
    # no "Match Statistics" header th; the parser must skip them rather than
    # crash (previously raised StopIteration) or emit spurious rows.
    rows = parse_match_page(MATCH_WITH_PLAYER_DETAILS_FIXTURE)

    # Only the Match Statistics table contributes rows; the Player Details
    # table (which has real rows in the fixture) contributes zero.
    assert len(rows) == 3
    assert all(row["team"] == "Adelaide" for row in rows)
    players = {row["player"] for row in rows}
    assert players == {"Berry, Sam", "Bond, Hugh", "Cook, Brayden"}

    # Real stat values come from the Match Statistics table, not the bio table.
    berry = next(r for r in rows if r["player"] == "Berry, Sam")
    assert berry["kicks"] == 9
    assert berry["handballs"] == 4
    assert berry["disposals"] == 13
    assert berry["marks"] == 1
    assert berry["tackles"] == 9
    assert berry["clearances"] == 2
    assert berry["contested_possessions"] == 5
    assert berry["brownlow_votes"] == 0

    bond = next(r for r in rows if r["player"] == "Bond, Hugh")
    assert bond["kicks"] == 3
    assert bond["handballs"] == 11
    assert bond["disposals"] == 14


def test_list_season_match_urls():
    urls = list_season_match_urls(SEASON_INDEX_FIXTURE)
    assert urls == [
        "https://afltables.com/afl/stats/games/2023/031420230316.html",
        "https://afltables.com/afl/stats/games/2023/040920230317.html",
        "https://afltables.com/afl/stats/games/2023/121820230318.html",
    ]


def test_list_season_match_urls_parses_season_results_page():
    # Regression guard: list_season_match_urls must parse the general
    # season-results page (afl/seas/{year}.html) format too, not just the
    # Brownlow round-by-round page. weekly_update.py and backfill_data.py fall
    # back to this page for in-progress seasons whose round-by-round page 404s.
    # If someone tightens the regex to the round-by-round layout, this fails.
    urls = list_season_match_urls(SEASON_RESULTS_FIXTURE)
    assert urls == [
        "https://afltables.com/afl/stats/games/2026/031620260305.html",
        "https://afltables.com/afl/stats/games/2026/092020260306.html",
        "https://afltables.com/afl/stats/games/2026/102120260307.html",
    ]
