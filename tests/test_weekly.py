import pandas as pd
from brownlow.dataset import STAT_COLUMNS
from brownlow.model import train_ranker
from brownlow.weekly import (
    accumulate_season_votes,
    assign_discrete_match_votes,
    per_round_votes,
    round_sort_key,
)


def _season_df() -> pd.DataFrame:
    rows = []
    for match_num in range(3):
        match_id = f"m{match_num}"
        star_row = {col: 20 for col in STAT_COLUMNS}
        star_row.update({"match_id": match_id, "brownlow_votes": 3, "player": "Star", "team": "Richmond"})
        other_row = {col: 5 for col in STAT_COLUMNS}
        other_row.update({"match_id": match_id, "brownlow_votes": 0, "player": "Other", "team": "Carlton"})
        rows.extend([star_row, other_row])
    return pd.DataFrame(rows)


class _FakeModel:
    """Model stub whose predict() returns a caller-supplied score per row,
    keyed off the first STAT_COLUMN so tests control ranking deterministically."""

    def __init__(self, scores_by_kicks):
        self._scores_by_kicks = scores_by_kicks

    def predict(self, features: pd.DataFrame):
        return [self._scores_by_kicks[k] for k in features["kicks"]]


def _match_row(player, kicks):
    row = {col: 0 for col in STAT_COLUMNS}
    row["kicks"] = kicks
    row.update({"match_id": "mX", "player": player, "team": "Richmond"})
    return row


def test_assign_discrete_match_votes_awards_3_2_1_to_top_three():
    # Five players in one match; predicted scores set via the fake model so we
    # know the exact ranking. kicks value is used purely as a lookup key.
    match_df = pd.DataFrame([
        _match_row("A", 10),
        _match_row("B", 40),
        _match_row("C", 30),
        _match_row("D", 20),
        _match_row("E", 5),
    ])
    scores = {10: 1.0, 40: 9.0, 30: 7.0, 20: 3.0, 5: 0.5}
    model = _FakeModel(scores)

    votes = assign_discrete_match_votes(model, match_df)

    # B highest -> 3, C -> 2, D -> 1, everyone else 0.
    expected = pd.Series([0, 3, 2, 1, 0], index=match_df.index)
    pd.testing.assert_series_equal(votes, expected, check_names=False)
    assert (votes >= 0).all()


def test_assign_discrete_match_votes_ties_broken_by_input_order():
    # A and B tie on score; deterministic stable sort keeps the earlier input
    # row (A) ranked ahead of the later one (B).
    match_df = pd.DataFrame([
        _match_row("A", 10),
        _match_row("B", 20),
        _match_row("C", 30),
    ])
    scores = {10: 5.0, 20: 5.0, 30: 9.0}
    model = _FakeModel(scores)

    votes = assign_discrete_match_votes(model, match_df)

    # C wins (3). A and B tie at 5.0; A comes first in input so gets 2, B gets 1.
    expected = pd.Series([2, 1, 3], index=match_df.index)
    pd.testing.assert_series_equal(votes, expected, check_names=False)


def test_assign_discrete_match_votes_preserves_original_index():
    match_df = pd.DataFrame(
        [_match_row("A", 10), _match_row("B", 20), _match_row("C", 30), _match_row("D", 40)],
        index=[100, 101, 102, 103],
    )
    scores = {10: 1.0, 20: 2.0, 30: 3.0, 40: 4.0}
    votes = assign_discrete_match_votes(_FakeModel(scores), match_df)
    assert list(votes.index) == [100, 101, 102, 103]
    assert votes.loc[103] == 3 and votes.loc[102] == 2 and votes.loc[101] == 1 and votes.loc[100] == 0


def test_accumulate_season_votes_ranks_players_by_total_predicted_votes():
    df = _season_df()
    model = train_ranker(df)
    leaderboard = accumulate_season_votes(model, df)

    assert list(leaderboard.columns) == ["player", "team", "predicted_season_votes"]
    assert leaderboard.iloc[0]["player"] == "Star"
    assert leaderboard["predicted_season_votes"].is_monotonic_decreasing


def test_accumulate_season_votes_returns_discrete_3_2_1_sums():
    # Two matches, four players each. Model scores are controlled via kicks so
    # the per-match top-3 ordering is known exactly. Star should top both
    # matches (3+3=6); Mid places 2nd both (2+2=4); Low places 3rd both (1+1=2);
    # Fringe never places top-3 -> 0.
    rows = []
    for match_num in range(2):
        mid = f"m{match_num}"
        rows.append({**{c: 0 for c in STAT_COLUMNS}, "kicks": 40, "match_id": mid, "player": "Star", "team": "Rich"})
        rows.append({**{c: 0 for c in STAT_COLUMNS}, "kicks": 30, "match_id": mid, "player": "Mid", "team": "Carl"})
        rows.append({**{c: 0 for c in STAT_COLUMNS}, "kicks": 20, "match_id": mid, "player": "Low", "team": "Geel"})
        rows.append({**{c: 0 for c in STAT_COLUMNS}, "kicks": 10, "match_id": mid, "player": "Fringe", "team": "Hawt"})
    season_df = pd.DataFrame(rows)
    scores = {40: 9.0, 30: 6.0, 20: 3.0, 10: 1.0}
    model = _FakeModel(scores)

    leaderboard = accumulate_season_votes(model, season_df)

    votes_by_player = dict(zip(leaderboard["player"], leaderboard["predicted_season_votes"]))
    assert votes_by_player == {"Star": 6, "Mid": 4, "Low": 2, "Fringe": 0}
    assert (leaderboard["predicted_season_votes"] >= 0).all()
    assert list(leaderboard.columns) == ["player", "team", "predicted_season_votes"]


def test_round_sort_key_orders_numeric_ascending_then_finals():
    # Convention: plain-integer rounds sort ascending by numeric value; finals
    # labels sort AFTER all numeric rounds, in the order QF, SF, PF, GF.
    unsorted = ["10", "2", "QF", "1", "GF", "23", "SF", "PF"]
    ordered = sorted(unsorted, key=round_sort_key)
    assert ordered == ["1", "2", "10", "23", "QF", "SF", "PF", "GF"]


def test_round_sort_key_naive_string_sort_would_be_wrong():
    # Guard the specific numeric-vs-string trap: naive sort puts "10" before "2".
    assert sorted(["10", "2", "1"], key=round_sort_key) == ["1", "2", "10"]


def _multi_round_season_df():
    # 3 rounds, one match each; 4 players per match with controlled kick scores
    # so per-match 3-2-1 allocation is known exactly.
    rows = []
    for rnd in ["1", "2", "10"]:
        mid = f"m{rnd}"
        rows.append({**{c: 0 for c in STAT_COLUMNS}, "kicks": 40, "match_id": mid, "round": rnd, "player": "Star", "team": "Rich"})
        rows.append({**{c: 0 for c in STAT_COLUMNS}, "kicks": 30, "match_id": mid, "round": rnd, "player": "Mid", "team": "Carl"})
        rows.append({**{c: 0 for c in STAT_COLUMNS}, "kicks": 20, "match_id": mid, "round": rnd, "player": "Low", "team": "Geel"})
        rows.append({**{c: 0 for c in STAT_COLUMNS}, "kicks": 10, "match_id": mid, "round": rnd, "player": "Fringe", "team": "Hawt"})
    return pd.DataFrame(rows)


def test_per_round_votes_returns_player_round_votes_scoped_to_players():
    season_df = _multi_round_season_df()
    scores = {40: 9.0, 30: 6.0, 20: 3.0, 10: 1.0}
    model = _FakeModel(scores)

    result = per_round_votes(model, season_df, ["Star", "Mid"])

    assert list(result.columns) == ["player", "round", "votes"]
    # Only requested players appear (scoping filter).
    assert set(result["player"]) == {"Star", "Mid"}
    # Star tops every match -> 3 votes each round; Mid is 2nd -> 2 each round.
    star = result[result["player"] == "Star"]
    assert list(star["round"]) == ["1", "2", "10"]  # numeric round order
    assert list(star["votes"]) == [3, 3, 3]
    mid = result[result["player"] == "Mid"]
    assert list(mid["votes"]) == [2, 2, 2]


def test_per_round_votes_orders_finals_after_numeric_rounds():
    rows = []
    for rnd in ["2", "GF", "1", "QF"]:
        mid = f"m{rnd}"
        rows.append({**{c: 0 for c in STAT_COLUMNS}, "kicks": 40, "match_id": mid, "round": rnd, "player": "Star", "team": "Rich"})
        rows.append({**{c: 0 for c in STAT_COLUMNS}, "kicks": 10, "match_id": mid, "round": rnd, "player": "Other", "team": "Carl"})
    season_df = pd.DataFrame(rows)
    model = _FakeModel({40: 9.0, 10: 1.0})

    result = per_round_votes(model, season_df, ["Star"])

    assert list(result["round"]) == ["1", "2", "QF", "GF"]


def test_per_round_votes_sums_duplicate_player_round_defensively():
    # A player somehow has two rows in the same round (two matches same round).
    rows = []
    for mid in ["mA", "mB"]:
        rows.append({**{c: 0 for c in STAT_COLUMNS}, "kicks": 40, "match_id": mid, "round": "5", "player": "Star", "team": "Rich"})
        rows.append({**{c: 0 for c in STAT_COLUMNS}, "kicks": 10, "match_id": mid, "round": "5", "player": "Other", "team": "Carl"})
    season_df = pd.DataFrame(rows)
    model = _FakeModel({40: 9.0, 10: 1.0})

    result = per_round_votes(model, season_df, ["Star"])

    # One combined row for round 5, votes summed (3 + 3 = 6), not a crash/dup.
    assert len(result) == 1
    assert result.iloc[0]["round"] == "5"
    assert result.iloc[0]["votes"] == 6
