import logging
from pathlib import Path

import pandas as pd

from brownlow.dataset import (
    assemble_match_records,
    rows_to_dataframe,
    STAT_COLUMNS,
    POSITION_COLUMNS,
    normalize_position,
    build_position_lookup,
    add_position_features,
)
from brownlow.footywire import TEAM_ROSTER_URL_TEMPLATE

AFLTABLES_FIXTURE = Path("tests/fixtures/afltables_match_sample.html").read_text()
TEAM_ROSTER_FIXTURE = Path("tests/fixtures/footywire_team_roster_sample.html").read_text()
FOOTYWIRE_FIXTURE = Path("tests/fixtures/footywire_advanced_sample.html").read_text()
# Real captured Adelaide (home, 101) v Hawthorn (away, 87) R22 2025 match page,
# used to verify team_margin on a non-draw where home_score != away_score.
ADELAIDE_HAWTHORN_FIXTURE = Path(
    "tests/fixtures/afltables_match_with_player_details_sample.html"
).read_text()


def test_stat_columns_has_19_entries():
    assert len(STAT_COLUMNS) == 19
    assert STAT_COLUMNS[:8] == [
        "kicks", "handballs", "disposals", "marks",
        "goals", "behinds", "hitouts", "tackles",
    ]
    # The 4 one-hot position columns are appended last so the existing 15
    # columns keep their ordering (score_involvements, intercepts, team_margin
    # precede them).
    assert POSITION_COLUMNS == [
        "position_forward", "position_midfield", "position_defender", "position_ruck",
    ]
    assert STAT_COLUMNS[-4:] == POSITION_COLUMNS
    assert STAT_COLUMNS[-7:-4] == ["score_involvements", "intercepts", "team_margin"]


def test_normalize_position_buckets_raw_strings():
    # The 4 primary categories map straight through, case-insensitively.
    assert normalize_position("Forward") == "forward"
    assert normalize_position("midfield") == "midfield"
    assert normalize_position("  DEFENDER  ") == "defender"
    assert normalize_position("Ruck") == "ruck"
    # Common abbreviations normalize too.
    assert normalize_position("FWD") == "forward"
    # Combo positions resolve to the PRIMARY (first-listed) role: footywire lists
    # the primary role first, so "MidfieldForward" -> midfield, "ForwardRuck" ->
    # forward. This keeps the encoding strictly one-hot.
    assert normalize_position("MidfieldForward") == "midfield"
    assert normalize_position("ForwardRuck") == "forward"
    # Genuinely unrecognized / empty / missing -> None (all-zero one-hot).
    assert normalize_position("Substitute") is None
    assert normalize_position("") is None
    assert normalize_position(None) is None


def test_add_position_features_one_hot_encoding():
    df = pd.DataFrame([
        {"season": 2015, "team": "Richmond", "player": "T. Cotchin"},   # MidfieldForward
        {"season": 2015, "team": "Richmond", "player": "D. Astbury"},   # Defender
        {"season": 2015, "team": "Richmond", "player": "S. Hampson"},   # Ruck
        {"season": 2015, "team": "Richmond", "player": "N. Obody"},     # not in lookup
    ])
    lookup = {
        (2015, "Richmond", "T. Cotchin"): "MidfieldForward",
        (2015, "Richmond", "D. Astbury"): "Defender",
        (2015, "Richmond", "S. Hampson"): "Ruck",
    }

    out = add_position_features(df, lookup)

    # All 4 one-hot columns exist regardless of match outcome.
    for col in POSITION_COLUMNS:
        assert col in out.columns

    cotchin = out[out["player"] == "T. Cotchin"].iloc[0]
    assert cotchin["position_midfield"] == 1  # combo -> primary role = midfield
    assert cotchin["position_forward"] == 0
    assert cotchin["position_defender"] == 0
    assert cotchin["position_ruck"] == 0

    astbury = out[out["player"] == "D. Astbury"].iloc[0]
    assert astbury["position_defender"] == 1
    assert astbury[["position_forward", "position_midfield", "position_ruck"]].sum() == 0

    hampson = out[out["player"] == "S. Hampson"].iloc[0]
    assert hampson["position_ruck"] == 1

    # A player absent from the lookup gets all-zero one-hots (the "unknown"
    # encoding) rather than an error.
    obody = out[out["player"] == "N. Obody"].iloc[0]
    assert obody[POSITION_COLUMNS].sum() == 0


def test_add_position_features_empty_dataframe():
    # An empty season DataFrame (no matches yet) still gains the 4 columns so
    # downstream model.predict(df[STAT_COLUMNS]) never KeyErrors.
    df = pd.DataFrame(columns=["season", "team", "player"])
    out = add_position_features(df, {})
    for col in POSITION_COLUMNS:
        assert col in out.columns
    assert len(out) == 0


def _roster_fake_fetch(url: str) -> str:
    # Only Richmond 2015 has a wired response; every other (season, team) URL
    # "fails", exercising the graceful per-team skip in build_position_lookup.
    if url == TEAM_ROSTER_URL_TEMPLATE.format(slug="richmond-tigers", year=2015):
        return TEAM_ROSTER_FIXTURE
    raise AssertionError(f"no roster wired for {url}")


def test_build_position_lookup_uses_real_slug_and_skips_failures():
    lookup = build_position_lookup(2015, 2015, fetch=_roster_fake_fetch)

    # Richmond 2015 players are keyed by (season, canonical team name, norm name).
    assert lookup[(2015, "Richmond", "T. Cotchin")] == "MidfieldForward"
    assert lookup[(2015, "Richmond", "D. Astbury")] == "Defender"
    assert lookup[(2015, "Richmond", "S. Hampson")] == "Ruck"

    # Every key carries the Richmond team name; the other 17 teams' fetches
    # "failed" and were skipped gracefully rather than crashing the whole build.
    assert {team for (_, team, _) in lookup} == {"Richmond"}
    assert len(lookup) == 5


def test_assemble_match_records_computes_team_margin_for_drawn_fixture():
    # Richmond (home) v Carlton (away) R1 2023 was a real draw, both sides 58.
    # So every player's team_margin is 0, and home/away branches must agree.
    records = assemble_match_records(2023, "031420230316", AFLTABLES_FIXTURE, None)
    bolton = next(r for r in records if r["player"] == "S. Bolton")  # Richmond (home)
    cerra = next(r for r in records if r["player"] == "A. Cerra")  # Carlton (away)
    assert bolton["team"] == "Richmond"
    assert bolton["team_margin"] == 0  # 58 - 58
    assert cerra["team"] == "Carlton"
    assert cerra["team_margin"] == 0  # 58 - 58


# A minimal, real Hawthorn Match Statistics table (Karl Amon, KI=15, from the
# same real R22 2025 page). The captured fixture only kept Adelaide's stats
# table, so we append this real away-team row to exercise the away (negative)
# margin branch on a non-draw -- which the drawn Richmond/Carlton fixture and an
# Adelaide-only (home) assertion cannot: it catches a hypothetical bug that
# always computes home_score - away_score regardless of the player's team.
_HAWTHORN_STATS_TABLE = """
<table class="sortable"><thead>
<tr><th colspan=25>Hawthorn Match Statistics</th></tr>
<tr><th>#</th><th>Player</th><th>KI</th><th>MK</th><th>HB</th><th>DI</th><th>GL</th>
<th>BH</th><th>HO</th><th>TK</th><th>RB</th><th>IF</th><th>CL</th><th>CG</th><th>FF</th>
<th>FA</th><th>BR</th><th>CP</th><th>UP</th><th>CM</th><th>MI</th><th>1%</th><th>BO</th>
<th>GA</th><th>%P</th></tr></thead><tbody>
<tr><td>18</td><td><a href="../../players/K/Karl_Amon.html">Amon, Karl</a></td>
<td>15</td><td>3</td><td>7</td><td>22</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td>
<td>2</td><td>3</td><td>1</td><td>1</td><td>2</td><td>&nbsp;</td><td>&nbsp;</td>
<td>&nbsp;</td><td>5</td><td>17</td><td>&nbsp;</td><td>&nbsp;</td><td>2</td><td>&nbsp;</td>
<td>&nbsp;</td><td>82</td></tr>
</tbody></table>
"""
ADELAIDE_HAWTHORN_BOTH_TEAMS = ADELAIDE_HAWTHORN_FIXTURE + _HAWTHORN_STATS_TABLE


def test_assemble_match_records_computes_signed_team_margin():
    # Non-draw real match: Adelaide (home) 101, Hawthorn (away) 87.
    # A home player's margin is home_score - away_score = +14; an away player's
    # is away_score - home_score = -14. This proves the sign follows the team.
    records = assemble_match_records(
        2025, "011020250801", ADELAIDE_HAWTHORN_BOTH_TEAMS, None
    )
    adelaide = next(r for r in records if r["team"] == "Adelaide")
    hawthorn = next(r for r in records if r["team"] == "Hawthorn")
    assert adelaide["team_margin"] == 14
    assert hawthorn["team_margin"] == -14


def test_assemble_match_records_joins_footywire_stats():
    records = assemble_match_records(
        season=2023,
        match_id="031420230316",
        afltables_html=AFLTABLES_FIXTURE,
        footywire_html=FOOTYWIRE_FIXTURE,  # Sydney v Geelong fixture, doesn't share players
    )
    assert len(records) == 6
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
    assert len(records) == 6
    assert all(r["score_involvements"] == 0 and r["intercepts"] == 0 for r in records)


def test_assemble_match_records_joins_across_team_name_alias():
    # afltables spells the club "Brisbane Lions"; footywire spells it "Brisbane".
    # The per-player join keys on (team, normalized_name), so without team-name
    # canonicalization a Brisbane Lions player never matches their footywire row
    # and loses score_involvements/intercepts. Here we (a) retitle the footywire
    # block to the footywire alias "Brisbane" and rename its top player to line
    # up with an afltables player, then (b) retitle the afltables team to
    # "Brisbane Lions" so the two sources disagree exactly as the real sites do.
    # After canonicalization the footywire "Brisbane" row joins onto afltables'
    # "Brisbane Lions" player with the real nonzero SI/ITC.
    afltables_html = AFLTABLES_FIXTURE.replace("Richmond", "Brisbane Lions")
    footywire_html = FOOTYWIRE_FIXTURE.replace(
        "<a name=t1></a>Sydney Match Statistics",
        "<a name=t1></a>Brisbane Match Statistics",  # footywire's alias spelling
    ).replace(
        'title="Oliver Florent">O Florent</a>',
        'title="Shai Bolton">S Bolton</a>',  # -> "S. Bolton", an afltables player
    )

    records = assemble_match_records(2023, "031420230316", afltables_html, footywire_html)

    bolton = next(r for r in records if r["player"] == "S. Bolton")
    assert bolton["team"] == "Brisbane Lions"  # canonical afltables spelling kept
    # These flow from the footywire "Brisbane" O-Florent row (SI=7, ITC=2) that
    # only joins because "Brisbane" was canonicalized to "Brisbane Lions".
    assert bolton["score_involvements"] == 7
    assert bolton["intercepts"] == 2


def test_rows_to_dataframe_has_expected_columns():
    records = assemble_match_records(2023, "031420230316", AFLTABLES_FIXTURE, None)
    df = rows_to_dataframe(records)
    # assemble_match_records populates every STAT_COLUMN EXCEPT the 4 one-hot
    # position columns, which are added later as a whole-DataFrame join step
    # (add_position_features) so assemble stays a pure per-match afltables+
    # footywire function that needs no season/team-level roster lookup.
    non_position_stats = [c for c in STAT_COLUMNS if c not in POSITION_COLUMNS]
    for col in non_position_stats + ["season", "round", "date", "match_id", "team", "player", "brownlow_votes"]:
        assert col in df.columns
    for col in POSITION_COLUMNS:
        assert col not in df.columns  # only add_position_features adds these
    assert len(df) == 6


def test_unmatched_footywire_player_is_logged(caplog):
    # Build a footywire fixture where ONE player is made to line up exactly with a
    # real afltables player on the join key of (team, normalized_name), and another
    # is left as a genuine mismatch. This proves the warning is *selective*: it must
    # fire for the mismatch but stay silent for the player that matches -- otherwise
    # the test can't distinguish "logs unmatched names" from "logs every row".
    #
    # afltables has ("Richmond", "S. Bolton") (Bolton, Shai). We retitle the first
    # footywire team block to "Richmond" and rename its top player "O Florent" to
    # "S Bolton" so it normalizes to "S. Bolton" and joins cleanly. The other players
    # in that block (e.g. "J Lloyd" -> "J. Lloyd") have no Richmond counterpart in
    # afltables and remain genuine mismatches.
    footywire_html = FOOTYWIRE_FIXTURE.replace(
        "<a name=t1></a>Sydney Match Statistics",
        "<a name=t1></a>Richmond Match Statistics",
    ).replace(
        'title="Oliver Florent">O Florent</a>',
        'title="Oliver Florent">S Bolton</a>',  # now matches afltables ("Richmond", "S. Bolton")
    )

    with caplog.at_level(logging.WARNING):
        assemble_match_records(2023, "031420230316", AFLTABLES_FIXTURE, footywire_html)

    warning_messages = [record.message for record in caplog.records]

    # 1. The genuinely mismatched player IS logged.
    assert any("J. Lloyd" in message for message in warning_messages), warning_messages

    # 2. The player made to match "S. Bolton" is NOT logged, even though it was
    #    present in the footywire lookup and processed alongside the mismatches.
    #    This is what proves the warning is selective rather than fired per-row.
    assert not any("S. Bolton" in message for message in warning_messages), warning_messages
