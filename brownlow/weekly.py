import pandas as pd

from brownlow.model import predict_match_votes


def assign_discrete_match_votes(model, match_df: pd.DataFrame) -> pd.Series:
    """Assign real-Brownlow-style discrete 3-2-1 votes for a SINGLE match.

    The model's raw ``predict_match_votes`` output (LightGBM lambdarank) is an
    arbitrary real-valued relevance score meaningful only for RELATIVE ordering
    within a match. This turns that ordering into the actual Brownlow voting
    mechanic: within the match, rank players by predicted score and award 3 to
    the top-ranked player, 2 to the second, 1 to the third, and 0 to everyone
    else. The result is aligned to ``match_df``'s index.

    Tie-breaking is deterministic (not "fair"): a stable descending sort is used,
    so when two players share the exact same predicted score the one appearing
    EARLIER in the input's row order is ranked higher. This guarantees a single,
    reproducible allocation rather than an ambiguous one.
    """
    scores = predict_match_votes(model, match_df)

    # Stable sort keeps input order for equal scores -> deterministic ties.
    ranked_index = scores.sort_values(ascending=False, kind="stable").index

    votes = pd.Series(0, index=match_df.index, dtype=int)
    for points, idx in zip((3, 2, 1), ranked_index[:3]):
        votes.loc[idx] = points
    return votes


def accumulate_season_votes(model, season_df: pd.DataFrame) -> pd.DataFrame:
    df = season_df.copy()

    # Assign discrete 3-2-1 votes per match, then sum across the season, so the
    # published "predicted_season_votes" mirrors the real Brownlow tally (always
    # a non-negative total in a realistic range) instead of summed raw scores.
    per_match_votes = [
        assign_discrete_match_votes(model, match_df)
        for _, match_df in df.groupby("match_id", sort=False)
    ]
    df["predicted_votes"] = pd.concat(per_match_votes)

    leaderboard = (
        df.groupby(["player", "team"])["predicted_votes"]
        .sum()
        .reset_index()
        .rename(columns={"predicted_votes": "predicted_season_votes"})
        .sort_values("predicted_season_votes", ascending=False)
        .reset_index(drop=True)
    )
    return leaderboard[["player", "team", "predicted_season_votes"]]
