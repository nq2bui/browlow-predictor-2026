import pandas as pd
from brownlow.dataset import STAT_COLUMNS
from brownlow.model import train_ranker
from brownlow.weekly import accumulate_season_votes


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


def test_accumulate_season_votes_ranks_players_by_total_predicted_votes():
    df = _season_df()
    model = train_ranker(df)
    leaderboard = accumulate_season_votes(model, df)

    assert list(leaderboard.columns) == ["player", "team", "predicted_season_votes"]
    assert leaderboard.iloc[0]["player"] == "Star"
    assert leaderboard["predicted_season_votes"].is_monotonic_decreasing
