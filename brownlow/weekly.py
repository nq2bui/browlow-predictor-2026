import pandas as pd

from brownlow.model import predict_match_votes


def accumulate_season_votes(model, season_df: pd.DataFrame) -> pd.DataFrame:
    df = season_df.copy()
    df["predicted_votes"] = predict_match_votes(model, df)

    leaderboard = (
        df.groupby(["player", "team"])["predicted_votes"]
        .sum()
        .reset_index()
        .rename(columns={"predicted_votes": "predicted_season_votes"})
        .sort_values("predicted_season_votes", ascending=False)
        .reset_index(drop=True)
    )
    return leaderboard[["player", "team", "predicted_season_votes"]]
