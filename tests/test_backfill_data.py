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
    # fake_fetch and should be skipped with a warning, not crash the run.
    # The match fixture holds 6 players (3 Richmond + 3 Carlton) -- the brief
    # text said 5, but the real fixture has 6, so the assertion is corrected.
    assert len(df) == 6  # 6 players in the one successfully-fetched match
    assert set(df["match_id"]) == {"031420230316"}
    assert df["season"].iloc[0] == 2023
