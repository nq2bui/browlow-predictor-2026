import argparse
import logging

import pandas as pd

from brownlow.model import train_ranker, save_model
from brownlow.backtest import top20_hit_rate

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
        hit_rate = top20_hit_rate(model, season_df)
        logger.info("season %s top-20 hit rate: %.2f", season, hit_rate)

    final_model = train_ranker(df)
    save_model(final_model, args.model_out)
    logger.info("saved model to %s", args.model_out)


if __name__ == "__main__":
    main()
