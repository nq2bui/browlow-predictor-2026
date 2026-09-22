import pandas as pd

from backtest_2026 import build_comparison_table


def _season_df():
    """3 rounds, 2 matches each, engineered so actual and predicted top-20 sets
    diverge by exactly one player (A misses actual top-20, F misses predicted
    top-20), which build_comparison_table's union logic must surface as rows
    for BOTH sides even though they only appear in one ranking's top 20."""
    rows = []
    # 19 players who dominate BOTH rankings (in both top-20s -> "hit").
    for i in range(19):
        player = f"P{i:02d}"
        rows.append({"player": player, "team": "Richmond", "match_id": "m0", "brownlow_votes": 20 - i})
    # Player A: barely misses the real top 20 (21st by actual votes) but is the
    # model's #20 pick.
    rows.append({"player": "A", "team": "Carlton", "match_id": "m0", "brownlow_votes": 0})
    # Player F: makes the real top 20 (20th by actual votes) but the model never
    # ranks it in its own top 20.
    rows.append({"player": "F", "team": "Collingwood", "match_id": "m0", "brownlow_votes": 1})
    return pd.DataFrame(rows)


def _predicted_df():
    predicted = [{"player": f"P{i:02d}", "team": "Richmond", "predicted_season_votes": 20 - i} for i in range(19)]
    predicted.append({"player": "A", "team": "Carlton", "predicted_season_votes": 0.5})
    predicted.append({"player": "F", "team": "Collingwood", "predicted_season_votes": 0.0})
    return pd.DataFrame(predicted)


def test_build_comparison_table_includes_union_of_both_top20s():
    table = build_comparison_table(_season_df(), _predicted_df())

    # 19 shared "hit" players + A (predicted-only) + F (actual-only) = 21 rows.
    assert len(table) == 21
    assert set(table["player"]) == {f"P{i:02d}" for i in range(19)} | {"A", "F"}


def test_build_comparison_table_flags_top20_membership_per_side():
    table = build_comparison_table(_season_df(), _predicted_df())
    by_player = table.set_index("player")

    # A: predicted top-20 (rank 20) but NOT actual top-20 (rank 21).
    assert by_player.loc["A", "in_predicted_top20"]
    assert not by_player.loc["A", "in_actual_top20"]

    # F: actual top-20 (rank 20, tied-but-ordered last with 1 vote) but NOT
    # predicted top-20 (predicted_season_votes=0.0 ranks it last, rank 21).
    assert by_player.loc["F", "in_actual_top20"]
    assert not by_player.loc["F", "in_predicted_top20"]

    # A shared player is in both.
    assert by_player.loc["P00", "in_actual_top20"]
    assert by_player.loc["P00", "in_predicted_top20"]


def test_build_comparison_table_fills_team_for_every_row():
    # Regression: team must be populated even for a player only present in one
    # side's top 20 (was NaN before the team_lookup fix).
    table = build_comparison_table(_season_df(), _predicted_df())
    assert table["team"].isna().sum() == 0
    assert table.set_index("player").loc["F", "team"] == "Collingwood"
