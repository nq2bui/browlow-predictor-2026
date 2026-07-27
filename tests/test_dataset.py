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


def test_rows_to_dataframe_has_expected_columns():
    records = assemble_match_records(2023, "031420230316", AFLTABLES_FIXTURE, None)
    df = rows_to_dataframe(records)
    for col in STAT_COLUMNS + ["season", "round", "date", "match_id", "team", "player", "brownlow_votes"]:
        assert col in df.columns
    assert len(df) == 6


def test_unmatched_footywire_player_is_logged(caplog):
    footywire_html_with_extra_player = FOOTYWIRE_FIXTURE.replace(
        'title="Oliver Florent">O Florent</a>',
        'title="Oliver Florent">O Florento</a>',  # simulate a name mismatch
    )
    with caplog.at_level(logging.WARNING):
        assemble_match_records(2023, "031420230316", AFLTABLES_FIXTURE, footywire_html_with_extra_player)
    assert any("O. Florento" in record.message for record in caplog.records)
