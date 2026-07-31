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


def assign_espn_style_votes(model, match_df: pd.DataFrame) -> pd.Series:
    """Assign ESPN-style fractional votes for a SINGLE match (experimental V2).

    Same input/output contract as ``assign_discrete_match_votes``, but instead of
    the official Brownlow 3-2-1 to exactly three players, this mirrors ESPN's AFL
    predictor: a finer-grained 6-tier fractional scale spreading credit across up
    to six players per match, to reduce the "margin for error" of a hard 3-tier
    cutoff. Within the match, rank players by predicted score and award
    3.0 / 2.5 / 2.0 / 1.5 / 1.0 / 0.5 to ranks 1-6 and 0.0 to everyone else. The
    result is a float Series aligned to ``match_df``'s index.

    Tie-breaking is identical to ``assign_discrete_match_votes``: a stable
    descending sort, so equal predicted scores rank in input row order,
    guaranteeing a single reproducible allocation.

    This is a PARALLEL, NON-PRODUCTION scoring scheme for head-to-head comparison
    against the 3-2-1 scheme; it does NOT feed the live leaderboard.
    """
    scores = predict_match_votes(model, match_df)

    # Stable sort keeps input order for equal scores -> deterministic ties,
    # exactly as assign_discrete_match_votes does.
    ranked_index = scores.sort_values(ascending=False, kind="stable").index

    votes = pd.Series(0.0, index=match_df.index, dtype=float)
    for points, idx in zip((3.0, 2.5, 2.0, 1.5, 1.0, 0.5), ranked_index[:6]):
        votes.loc[idx] = points
    return votes


def accumulate_season_votes(
    model, season_df: pd.DataFrame, vote_assigner=assign_discrete_match_votes
) -> pd.DataFrame:
    """Sum per-match votes into a season leaderboard, using a pluggable scheme.

    ``vote_assigner`` is a callable with the same contract as
    ``assign_discrete_match_votes`` / ``assign_espn_style_votes``: it takes
    ``(model, match_df)`` for ONE match and returns a per-match vote ``pd.Series``
    aligned to that match's index. It defaults to the production 3-2-1
    (``assign_discrete_match_votes``) scheme, so existing callers are unchanged;
    passing ``vote_assigner=assign_espn_style_votes`` produces an independent
    ESPN-scheme leaderboard for side-by-side comparison. Columns and sort order
    (``player, team, predicted_season_votes`` descending) are identical either way.
    """
    df = season_df.copy()

    # Assign per-match votes with the chosen scheme, then sum across the season, so
    # the published "predicted_season_votes" mirrors a real Brownlow-style tally
    # (always a non-negative total in a realistic range) instead of summed raw
    # scores.
    per_match_votes = [
        vote_assigner(model, match_df)
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


# Finals-round labels, in the order they occur in a season, sorted AFTER all
# numeric home-and-away rounds.
_FINALS_ORDER = ["QF", "SF", "PF", "GF"]


def round_sort_key(round_value) -> tuple:
    """Sort key for a ``round`` label so rounds display in real season order.

    Convention: a plain-integer label ("1".."23") sorts ascending by its numeric
    value and BEFORE any finals label. A finals label sorts after all numeric
    rounds, in the order QF, SF, PF, GF. Any other (unrecognised) non-numeric
    label sorts last, alphabetically among itself. This avoids the naive
    string-sort trap where "10" would come before "2".
    """
    s = str(round_value)
    if s.isdigit():
        return (0, int(s), "")
    if s in _FINALS_ORDER:
        return (1, _FINALS_ORDER.index(s), "")
    return (2, 0, s)


def per_round_votes(model, season_df: pd.DataFrame, players: list) -> pd.DataFrame:
    """Per-round discrete 3-2-1 votes for a specified set of players.

    Output is scoped to ``players`` (e.g. the top-20 leaderboard). Crucially, the
    3-2-1 allocation is computed on the FULL match (every player's row), exactly
    as ``accumulate_season_votes`` does, and only then filtered down to the
    requested players. This guarantees the per-round breakdown RECONCILES with
    the season total on the leaderboard: a player's votes each round are the same
    votes that summed into their ``predicted_season_votes``. (Filtering rows out
    *before* scoring would re-rank a shrunken field and inflate votes, so a
    player ranked 4th in the real match — 0 votes — could wrongly score 3.)

    Only matches that actually contain a requested player are scored, so this
    stays cheap without affecting correctness (votes are allocated per match).

    Returns a DataFrame with columns ``player, round, votes`` — one row per
    player per round they played (rounds not played simply have no row). If a
    player appears more than once in the same round, their votes for that round
    are summed. Rows are ordered by ``round_sort_key`` (numeric rounds ascending,
    then finals QF/SF/PF/GF), and by player within a round.
    """
    player_set = set(players)
    relevant_match_ids = set(
        season_df.loc[season_df["player"].isin(player_set), "match_id"]
    )
    df = season_df[season_df["match_id"].isin(relevant_match_ids)].copy()

    if df.empty:
        return pd.DataFrame(columns=["player", "round", "votes"])

    per_match_votes = [
        assign_discrete_match_votes(model, match_df)
        for _, match_df in df.groupby("match_id", sort=False)
    ]
    df["votes"] = pd.concat(per_match_votes)

    # Keep only the requested players AFTER the per-match allocation.
    df = df[df["player"].isin(player_set)]

    result = (
        df.groupby(["player", "round"])["votes"]
        .sum()
        .reset_index()
    )
    result["_key"] = result["round"].map(round_sort_key)
    result = (
        result.sort_values(["_key", "player"])
        .drop(columns="_key")
        .reset_index(drop=True)
    )
    return result[["player", "round", "votes"]]
