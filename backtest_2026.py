"""Compare the model's predicted 2026 leaderboard against the real 2026 votes.

The Brownlow Medal count is a secret ballot revealed only after the season
ends, so afltables' round-by-round Brownlow page for a season
(``SEASON_INDEX_URL_TEMPLATE``) only appears once that season concludes. This
script scrapes that now-published 2026 data (real votes included) with the
same ``backfill_seasons`` machinery ``backfill_data.py`` uses for historical
seasons, scores those same matches with the trained production model, and
prints the top-20 hit rate plus a side-by-side ranked comparison.

This is read-only: it does not retrain the model or touch data/training_data.parquet.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from backfill_data import backfill_seasons
from brownlow.backtest import top20_hit_rate_with_scheme
from brownlow.dashboard import render_season_review
from brownlow.model import load_model
from brownlow.weekly import accumulate_season_votes, assign_discrete_match_votes, per_round_votes

logger = logging.getLogger(__name__)

SEASON = 2026


def build_comparison_table(season_df: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side actual vs predicted ranks for the union of both top-20s.

    ``season_df`` is the real scraped data (with ``brownlow_votes``); ``predicted``
    is ``accumulate_season_votes``'s output (``player, team, predicted_season_votes``).
    """
    actual = (
        season_df.groupby(["player", "team"])["brownlow_votes"]
        .sum()
        .reset_index()
        .rename(columns={"brownlow_votes": "actual_votes"})
        .sort_values("actual_votes", ascending=False)
        .reset_index(drop=True)
    )
    actual["actual_rank"] = actual.index + 1
    predicted = predicted.sort_values("predicted_season_votes", ascending=False).reset_index(drop=True)
    predicted["predicted_rank"] = predicted.index + 1

    team_lookup = actual.drop_duplicates("player").set_index("player")["team"]

    actual_top20 = actual.head(20)
    predicted_top20 = predicted.head(20)

    union_players = pd.Index(actual_top20["player"]).union(predicted_top20["player"])
    table = pd.DataFrame({"player": union_players})
    table["team"] = table["player"].map(team_lookup)
    table = table.merge(
        actual[["player", "actual_rank", "actual_votes"]], on="player", how="left"
    )
    table = table.merge(
        predicted[["player", "predicted_rank", "predicted_season_votes"]],
        on="player",
        how="left",
    )

    table["in_actual_top20"] = table["actual_rank"] <= 20
    table["in_predicted_top20"] = table["predicted_rank"] <= 20
    table = table.sort_values(
        ["in_actual_top20", "actual_rank"], ascending=[False, True]
    ).reset_index(drop=True)
    return table[
        [
            "player",
            "team",
            "actual_rank",
            "actual_votes",
            "predicted_rank",
            "predicted_season_votes",
            "in_actual_top20",
            "in_predicted_top20",
        ]
    ]


def actual_round_votes_for(season_df: pd.DataFrame, players: list) -> pd.DataFrame:
    """Real per-round votes for ``players``, shaped like ``brownlow.weekly.per_round_votes``.

    Unlike the predicted side (which needs the model to assign per-match votes),
    the real votes are already in ``season_df["brownlow_votes"]`` -- this is a
    straight groupby, no scoring involved.
    """
    player_set = set(players)
    df = season_df[season_df["player"].isin(player_set)]
    if df.empty:
        return pd.DataFrame(columns=["player", "round", "votes"])
    return (
        df.groupby(["player", "round"])["brownlow_votes"]
        .sum()
        .reset_index()
        .rename(columns={"brownlow_votes": "votes"})
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="model.txt")
    parser.add_argument(
        "--cache",
        type=str,
        default="data/actual_2026.parquet",
        help="Where to save/reuse the scraped real 2026 season data.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-scrape season data even if --cache already exists.",
    )
    parser.add_argument(
        "--html-out",
        type=str,
        default="season_review_2026.html",
        help="Where to write the static predicted-vs-actual review page.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cache_path = Path(args.cache)
    if cache_path.exists() and not args.refresh:
        logger.info("loading cached 2026 season data from %s", cache_path)
        season_df = pd.read_parquet(cache_path)
    else:
        logger.info("scraping real %d season data from afltables/footywire", SEASON)
        season_df = backfill_seasons(SEASON, SEASON)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        season_df.to_parquet(cache_path, index=False)
        logger.info("wrote %d rows to %s", len(season_df), cache_path)

    if season_df.empty:
        logger.error("no 2026 season data available, aborting")
        return

    if season_df["brownlow_votes"].sum() == 0:
        logger.warning(
            "every brownlow_votes value is 0 -- the 2026 Brownlow count may not "
            "be published yet, or the round-by-round page 404'd and fell back "
            "to the season-results page (which carries no votes). Re-run with "
            "--refresh once the count is confirmed public."
        )

    model = load_model(args.model)
    predicted = accumulate_season_votes(model, season_df, vote_assigner=assign_discrete_match_votes)

    hit_rate = top20_hit_rate_with_scheme(model, season_df, assign_discrete_match_votes)
    print(f"2026 top-20 hit rate (Standard 3-2-1 production scoring): {hit_rate:.2f}")
    print()

    table = build_comparison_table(season_df, predicted)
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(table.to_string(index=False))

    players = table["player"].tolist()
    actual_rv = actual_round_votes_for(season_df, players)
    predicted_rv = per_round_votes(model, season_df, players, vote_assigner=assign_discrete_match_votes)

    render_season_review(table, actual_rv, predicted_rv, hit_rate, args.html_out, SEASON)
    logger.info("wrote season review page to %s", args.html_out)


if __name__ == "__main__":
    main()
