import argparse
import logging

import pandas as pd

from brownlow.model import train_ranker, save_model
from brownlow.backtest import top20_hit_rate_with_scheme
from brownlow.weekly import assign_discrete_match_votes

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model-out", type=str, required=True)
    parser.add_argument("--holdout-seasons", type=int, nargs="+", default=[2024, 2025])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    df = pd.read_parquet(args.data)

    train_df = df[~df["season"].isin(args.holdout_seasons)]
    model = train_ranker(train_df)

    for season in args.holdout_seasons:
        season_df = df[df["season"] == season]
        if season_df.empty:
            continue
        # Measure hit rate the way production actually scores: convert each
        # match's raw model scores to discrete 3-2-1 votes (the same
        # assign_discrete_match_votes step accumulate_season_votes uses) before
        # summing per player, so the reported number reflects the deployed
        # leaderboard rather than raw summed scores.
        hit_rate = top20_hit_rate_with_scheme(
            model, season_df, assign_discrete_match_votes
        )
        logger.info(
            "season %s top-20 hit rate (3-2-1 production scoring): %.2f",
            season,
            hit_rate,
        )

    final_model = train_ranker(df)
    save_model(final_model, args.model_out)
    logger.info("saved model to %s", args.model_out)


if __name__ == "__main__":
    main()
