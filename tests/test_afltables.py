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
