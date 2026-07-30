from pathlib import Path

from brownlow.footywire import parse_advanced_stats_page, list_season_match_ids, parse_team_roster

ADV_FIXTURE = Path("tests/fixtures/footywire_advanced_sample.html").read_text()
# Real team-roster page rows sliced verbatim from a live captured 2015 Richmond
# page (.superpowers/sdd/debug_footywire_roster_2015.html, tp-richmond-tigers?
# year=2015). Position is the LAST <td class="data"> cell of each darkcolor/
# lightcolor row. The fixture deliberately spans 5 distinct real position values
# -- Midfield, Defender, Forward, the combo "MidfieldForward", and Ruck -- and
# the first row (Arnot, Matthew) carries the extra <span class="playerflag"
# title="Rookie">R</span> after the name link, to prove that span does not break
# name or position parsing. footywire's roster table also has a stray extra
# </tr> after every row that html.parser mis-nests (dropping all but the first
# row), so parse_team_roster uses lxml, which recovers the intended structure.
TEAM_ROSTER_FIXTURE = Path("tests/fixtures/footywire_team_roster_sample.html").read_text()
# Real footywire pattern: a substituted player's name carries a trailing arrow
# marker directly appended with no separator -- U+2197 (↗, subbed ON) or U+2199
# (↙, subbed OFF). We reproduce it by suffixing the arrow onto a real player row
# in the base fixture, mirroring the exact live-page shape (e.g. "N Vlastuin↗").
FOOTYWIRE_SUB_ARROW_FIXTURE_ON = ADV_FIXTURE.replace(">O Florent<", ">N Vlastuin↗<")
FOOTYWIRE_SUB_ARROW_FIXTURE_OFF = ADV_FIXTURE.replace(">J Lloyd<", ">J Ross↙<")
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


def test_parse_advanced_stats_page_strips_substitution_arrow_markers():
    # footywire suffixes a player's name with an arrow when they were
    # substituted during the match: U+2197 (↗, subbed ON) or U+2199 (↙, subbed
    # OFF), appended directly to the name with no separator (e.g. "N Vlastuin↗").
    # afltables carries no such marker, so these names never matched on the join
    # key and silently lost their score_involvements/intercepts. The parser must
    # strip the trailing arrow (and surrounding whitespace) BEFORE the name is
    # used, so "N Vlastuin↗" -> "N Vlastuin" normalizes identically to
    # afltables' "Vlastuin, Nathan" -> "N. Vlastuin".
    subbed_on = FOOTYWIRE_SUB_ARROW_FIXTURE_ON  # "O Florent" renamed "N Vlastuin↗"
    rows = parse_advanced_stats_page(subbed_on)
    players = {r["player"] for r in rows}
    assert "N Vlastuin" in players  # arrow stripped
    assert "N Vlastuin↗" not in players  # raw arrow-suffixed name gone
    vlastuin = next(r for r in rows if r["player"] == "N Vlastuin")
    # Stats still parse correctly for the stripped row (from the O Florent row).
    assert vlastuin["score_involvements"] == 7
    assert vlastuin["intercepts"] == 2

    subbed_off = FOOTYWIRE_SUB_ARROW_FIXTURE_OFF  # "J Lloyd" renamed "J Ross↙"
    rows_off = parse_advanced_stats_page(subbed_off)
    players_off = {r["player"] for r in rows_off}
    assert "J Ross" in players_off  # ↙ stripped too
    assert "J Ross↙" not in players_off


def test_parse_team_roster():
    roster = parse_team_roster(TEAM_ROSTER_FIXTURE)

    # 5 real player rows sliced into the fixture; every row yields an entry.
    assert len(roster) == 5

    by_player = {r["player"]: r["position"] for r in roster}

    # Name is normalized to the canonical "F. Surname" join key via
    # normalize_player_name, from footywire's "Surname, First" format.
    assert by_player["D. Astbury"] == "Defender"
    assert by_player["D. Butler"] == "Forward"
    assert by_player["S. Hampson"] == "Ruck"
    # Combo positions occur in real data and are returned verbatim (raw string);
    # bucketing into one-hot columns happens later in the dataset layer.
    assert by_player["T. Cotchin"] == "MidfieldForward"

    # The row with the extra <span class="playerflag" title="Rookie">R</span>
    # after the name link still parses cleanly: name is not polluted by the "R"
    # flag, and the last data cell ("Midfield") is still read as the position.
    assert by_player["M. Arnot"] == "Midfield"

    # Position text is whitespace-stripped (the real cell is newline-padded).
    assert all(p == p.strip() for p in by_player.values())


def test_parse_team_roster_missing_div_returns_empty():
    # Graceful degradation: a page without the team-players-div (e.g. an error
    # page or an unexpected layout) yields an empty roster, not a crash.
    assert parse_team_roster("<html><body>no roster here</body></html>") == []


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
