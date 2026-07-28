from pathlib import Path

from brownlow.footywire import parse_advanced_stats_page, list_season_match_ids

ADV_FIXTURE = Path("tests/fixtures/footywire_advanced_sample.html").read_text()
MATCH_LIST_FIXTURE = Path("tests/fixtures/footywire_match_list_sample.html").read_text()
# The base match-list fixture plus two real "noise" rows (Change Password /
# Update Settings account widgets) extracted from a real captured live page
# (.superpowers/sdd/debug_real_footywire_matchlist_2025.html). These reuse the
# darkcolor/lightcolor classes but have a single colspan cell, not match cells.
MATCH_LIST_WITH_NOISE_FIXTURE = Path(
    "tests/fixtures/footywire_match_list_with_noise_sample.html"
).read_text()
# Real match rows sliced verbatim from a live captured page
# (.superpowers/sdd/debug_real_footywire_matchlist_2021.html). Genuine
# completed-match rows have 7 <td> cells: [0] date, [1] team names (2 links),
# [2] venue, [3] attendance (plain number, no link), [4] result (the
# ft_match_statistics?mid=... link lives HERE, not in [3]), [5] leading
# disposals, [6] leading goals. The old parser looked for the link in tds[3]
# (attendance, no <a>) and therefore returned zero matches on real pages.
MATCH_LIST_REAL_2021_FIXTURE = Path(
    "tests/fixtures/footywire_match_list_real_2021_sample.html"
).read_text()
# Real advanced-stats page rows sliced verbatim from a live captured page
# (.superpowers/sdd/debug_real_footywire_unused_substitute.html, Collingwood v
# Western Bulldogs, mid=10328, 2021 round 1). Every genuine player row has 18
# <td> cells (Player + 17 stat columns), but a player named as an
# emergency/substitute who never took the field gets an "Unused Substitute" row
# with only 2 <td> cells: the name and a colspan="17" "Unused Substitute" cell.
# The old parser indexed values[si_index] (~11) on that 1-element values list
# and raised IndexError, crashing the whole backfill.
ADV_WITH_UNUSED_SUB_FIXTURE = Path(
    "tests/fixtures/footywire_advanced_stats_with_unused_substitute_sample.html"
).read_text()
# Real advanced-stats page rows sliced verbatim from a live captured 2012 page
# (.superpowers/sdd/debug_real_footywire_2012_no_si_itc.html, GWS v Sydney,
# mid=5343). footywire's older advanced-stats pages track a SMALLER column set:
# ['Player', 'CP', 'UP', 'ED', 'DE%', 'CM', 'GA', 'MI5', '1%', 'BO', 'TOG%'] --
# with no 'SI' (Score Involvements) or 'ITC' (Intercepts) columns at all (those
# were added by footywire sometime after 2012). The old parser did
# headers.index("SI"), which raised ValueError: 'SI' is not in list and crashed
# the whole match's parse (~25% of 2012-2017 matches in the real backfill).
ADV_NO_SI_ITC_FIXTURE = Path(
    "tests/fixtures/footywire_advanced_stats_no_si_itc_sample.html"
).read_text()


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


def test_parse_advanced_stats_page_skips_unused_substitute_rows():
    # A player named as an emergency but who never played gets an "Unused
    # Substitute" row with only 2 <td> cells instead of the usual 18. The old
    # parser raised IndexError reaching for values[si_index] on that short row.
    # The fix skips any row too short to contain the SI/ITC values, so the
    # unused substitutes (C Brown, R West) are omitted entirely -- matching how
    # the pipeline already treats a player it has no data for -- while the
    # genuine player rows are still parsed with their real SI/ITC values.
    rows = parse_advanced_stats_page(ADV_WITH_UNUSED_SUB_FIXTURE)

    players = {r["player"] for r in rows}
    assert "C Brown" not in players  # Collingwood unused substitute
    assert "R West" not in players  # Western Bulldogs unused substitute

    pendlebury = next(r for r in rows if r["player"] == "S Pendlebury")
    assert pendlebury["team"] == "Collingwood"
    assert pendlebury["score_involvements"] == 7
    assert pendlebury["intercepts"] == 7

    smith = next(r for r in rows if r["player"] == "B Smith")
    assert smith["team"] == "Western Bulldogs"
    assert smith["score_involvements"] == 7
    assert smith["intercepts"] == 5

    # Only the genuine players survive: 2 per team in the sliced fixture.
    assert len(rows) == 4


def test_parse_advanced_stats_page_older_page_without_si_itc_columns():
    # footywire's older (2012-era) advanced-stats pages have a smaller column
    # set that lacks the 'SI' and 'ITC' columns entirely. The old parser did
    # headers.index("SI") and raised ValueError: 'SI' is not in list, crashing
    # the whole match's parse. The fix defaults score_involvements/intercepts to
    # 0 when those columns are absent, rather than crashing -- matching how the
    # pipeline already treats data it can't find. These rows are sliced verbatim
    # from a real captured 2012 GWS v Sydney page (mid=5343).
    rows = parse_advanced_stats_page(ADV_NO_SI_ITC_FIXTURE)

    # 2 real player rows per team were sliced into the fixture.
    assert len(rows) == 4
    teams = {r["team"] for r in rows}
    assert teams == {"GWS", "Sydney"}

    kennedy = next(r for r in rows if r["player"] == "Adam Kennedy")
    assert kennedy["team"] == "GWS"
    # These columns don't exist on the older page -> default to 0, no crash.
    assert kennedy["score_involvements"] == 0
    assert kennedy["intercepts"] == 0

    jack = next(r for r in rows if r["player"] == "Kieren Jack")
    assert jack["team"] == "Sydney"
    assert jack["score_involvements"] == 0
    assert jack["intercepts"] == 0


def test_list_season_match_ids():
    matches = list_season_match_ids(MATCH_LIST_FIXTURE)
    assert matches == [
        {"mid": 10751, "home_team": "Richmond", "away_team": "Carlton"},
        {"mid": 10752, "home_team": "Collingwood", "away_team": "Sydney"},
    ]


def test_list_season_match_ids_skips_non_match_noise_rows():
    # Real footywire pages carry account-settings widgets ("Change Password" /
    # "Update Settings") that reuse the darkcolor/lightcolor classes but have a
    # single colspan cell, not the 4 cells a real match row has. The parser
    # must skip them (previously raised IndexError on tds[1]) and return only
    # the genuine matches -- identical to the noise-free fixture's output.
    matches = list_season_match_ids(MATCH_LIST_WITH_NOISE_FIXTURE)
    assert matches == [
        {"mid": 10751, "home_team": "Richmond", "away_team": "Carlton"},
        {"mid": 10752, "home_team": "Collingwood", "away_team": "Sydney"},
    ]


def test_list_season_match_ids_real_2021_seven_column_rows():
    # Real footywire match rows have 7 columns and the ft_match_statistics?mid=
    # link lives in the RESULT column (index 4), not the attendance column
    # (index 3). The old parser hardcoded tds[3], found no <a> there, and
    # returned an empty list for every genuine match. These rows are sliced
    # verbatim from a real captured 2021 page.
    matches = list_season_match_ids(MATCH_LIST_REAL_2021_FIXTURE)
    assert matches == [
        {"mid": 10327, "home_team": "Richmond", "away_team": "Carlton"},
        {"mid": 10328, "home_team": "Collingwood", "away_team": "Western Bulldogs"},
        {"mid": 10329, "home_team": "Melbourne", "away_team": "Fremantle"},
    ]
