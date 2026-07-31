"""Head-to-head comparison of per-match vote-conversion schemes (read-only).

Compares two ways of turning the SAME trained LightGBM per-match relevance
scores into per-match "votes", then into a season leaderboard:

  V1 (3-2-1)          -> brownlow.weekly.assign_discrete_match_votes
                         (the official Brownlow mechanic used by the live
                          production leaderboard)
  V2 (ESPN fractional) -> brownlow.weekly.assign_espn_style_votes
                         (a finer 6-tier 3/2.5/2/1.5/1/0.5 scale, experimental)

For each holdout season it prints both schemes' top-20 hit rate against the real
historical Brownlow top 20 (summed ``brownlow_votes``). This script does NOT
retrain the model or write any model/data files -- it is purely a comparison and
reporting tool. It does NOT change the production scoring scheme.
"""

import argparse

import pandas as pd

from brownlow.model import load_model
from brownlow.backtest import top20_hit_rate_with_scheme
from brownlow.weekly import assign_discrete_match_votes, assign_espn_style_votes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the training/backtest parquet.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to a trained model.txt.")
    parser.add_argument("--holdout-seasons", type=int, nargs="+", default=[2024, 2025])
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    model = load_model(args.model)

    for season in args.holdout_seasons:
        season_df = df[df["season"] == season]
        if season_df.empty:
            print(f"season {season}: no rows in data, skipping")
            continue
        v1 = top20_hit_rate_with_scheme(model, season_df, assign_discrete_match_votes)
        v2 = top20_hit_rate_with_scheme(model, season_df, assign_espn_style_votes)
        print(
            f"season {season}: V1 (3-2-1) hit rate = {v1:.2f}, "
            f"V2 (ESPN fractional) hit rate = {v2:.2f}"
        )


if __name__ == "__main__":
    main()
