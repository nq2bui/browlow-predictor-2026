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


# A footywire fixture whose team + player are relabelled to line up with a real
# afltables fixture player. The unmodified fixture is "Sydney v Geelong" with
# "O Florent" (score_involvements=7, intercepts=2 in the SI/ITC columns), which
# never matches the "Richmond v Carlton" afltables match, so every join misses
# and falls back to 0. Here we relabel Sydney -> Richmond and O Florent ->
# S Bolton so the footywire row aligns with afltables' "Bolton, Shai"
# (normalized to "S. Bolton") on both team AND normalized name.
FOOTYWIRE_ADV_ALIGNED = FOOTYWIRE_ADV.replace(
    "Sydney Match Statistics", "Richmond Match Statistics"
).replace(">O Florent<", ">S Bolton<")


def fake_fetch_aligned(url: str) -> str:
    if url == MATCH_STATS_URL_TEMPLATE.format(mid=10751):
        return FOOTYWIRE_ADV_ALIGNED
    return fake_fetch(url)


def test_backfill_joins_footywire_stats_into_matching_player():
    # Prove the "successful join" path: a footywire stat actually flows into the
    # matching afltables player's row with a nonzero value. mid=10751 maps to the
    # (Richmond, Carlton) match in the footywire match list, so the aligned
    # footywire HTML (team relabelled to Richmond, player to S Bolton) joins onto
    # afltables' Richmond "Bolton, Shai" row.
    df = backfill_seasons(2023, 2023, fetch=fake_fetch_aligned)

    bolton = df[(df["team"] == "Richmond") & (df["player"] == "S. Bolton")]
    assert len(bolton) == 1
    # These come from the O Florent row of the fixture (SI=7, ITC=2), now
    # relabelled onto S Bolton -- nonzero footywire values flowing through join.
    assert bolton["score_involvements"].iloc[0] == 7
    assert bolton["intercepts"].iloc[0] == 2

    # And a Richmond player with no matching footywire row still falls back to 0.
    baker = df[(df["team"] == "Richmond") & (df["player"] == "L. Baker")]
    assert len(baker) == 1
    assert baker["score_involvements"].iloc[0] == 0
    assert baker["intercepts"].iloc[0] == 0
