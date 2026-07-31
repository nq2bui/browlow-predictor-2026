import pandas as pd

from brownlow.model import predict_match_votes


def top20_hit_rate(model, season_df: pd.DataFrame) -> float:
    """Top-20 backtest hit rate using SUMMED RAW per-match relevance scores.

    This measures the RAW-score path only: it sums the raw ``predict_match_votes``
    LightGBM scores per player, ranks, and measures the top-20 overlap against the
    real historical ``brownlow_votes`` top 20. It is NOT how the production
    leaderboard scores (production first converts each match's scores to discrete
    3-2-1 votes via ``assign_discrete_match_votes`` before summing), and it is no
    longer what ``train_model.py`` reports -- that now uses
    ``top20_hit_rate_with_scheme`` with the production 3-2-1 assigner. This raw
    variant is retained as a baseline and for the per-player-sum aggregation
    regression test; for the production/scheme-parameterized version see
    ``top20_hit_rate_with_scheme``.
    """
    df = season_df.copy()
    df["predicted_votes"] = predict_match_votes(model, df)

    actual_totals = df.groupby("player")["brownlow_votes"].sum().sort_values(ascending=False)
    predicted_totals = df.groupby("player")["predicted_votes"].sum().sort_values(ascending=False)

    actual_top20 = set(actual_totals.head(20).index)
    predicted_top20 = set(predicted_totals.head(20).index)

    if not actual_top20:
        return 0.0
    return len(actual_top20 & predicted_top20) / len(actual_top20)


def top20_hit_rate_with_scheme(model, season_df: pd.DataFrame, vote_assigner) -> float:
    """Top-20 hit rate where the PREDICTED side is built by a per-match vote scheme.

    ``vote_assigner`` is a callable with the same contract as
    ``brownlow.weekly.assign_discrete_match_votes`` /
    ``assign_espn_style_votes`` -- it takes ``(model, match_df)`` for ONE match and
    returns a per-match vote ``pd.Series`` aligned to that match's index. This lets
    the 3-2-1 (V1) and ESPN fractional (V2) schemes be compared on equal footing.

    Methodology mirrors ``top20_hit_rate`` exactly, only the predicted side
    changes: group ``season_df`` by ``match_id``, apply ``vote_assigner`` to each
    match to get predicted per-match votes, sum per player into predicted season
    totals, take the predicted top 20, and compare against the REAL top 20 (summed
    historical ``brownlow_votes`` -- the only real 3-2-1 ground truth, unchanged
    for both schemes). Returns the overlap fraction.
    """
    df = season_df.copy()

    per_match_votes = [
        vote_assigner(model, match_df)
        for _, match_df in df.groupby("match_id", sort=False)
    ]
    df["predicted_votes"] = pd.concat(per_match_votes)

    actual_totals = df.groupby("player")["brownlow_votes"].sum().sort_values(ascending=False)
    predicted_totals = df.groupby("player")["predicted_votes"].sum().sort_values(ascending=False)

    actual_top20 = set(actual_totals.head(20).index)
    predicted_top20 = set(predicted_totals.head(20).index)

    if not actual_top20:
        return 0.0
    return len(actual_top20 & predicted_top20) / len(actual_top20)
