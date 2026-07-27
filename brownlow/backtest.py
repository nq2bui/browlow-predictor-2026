import pandas as pd

from brownlow.model import predict_match_votes


def top20_hit_rate(model, season_df: pd.DataFrame) -> float:
    df = season_df.copy()
    df["predicted_votes"] = predict_match_votes(model, df)

    actual_totals = df.groupby("player")["brownlow_votes"].sum().sort_values(ascending=False)
    predicted_totals = df.groupby("player")["predicted_votes"].sum().sort_values(ascending=False)

    actual_top20 = set(actual_totals.head(20).index)
    predicted_top20 = set(predicted_totals.head(20).index)

    if not actual_top20:
        return 0.0
    return len(actual_top20 & predicted_top20) / len(actual_top20)
