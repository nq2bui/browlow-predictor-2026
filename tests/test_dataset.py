import logging
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
    for col in STAT_COLUMNS + ["season", "round", "date", "match_id", "team", "player", "brownlow_votes"]:
        assert col in df.columns
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
