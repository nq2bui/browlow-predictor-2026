from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from brownlow.dataset import STAT_COLUMNS


def train_ranker(df: pd.DataFrame) -> lgb.LGBMRanker:
    df_sorted = df.sort_values("match_id").reset_index(drop=True)
    group_sizes = df_sorted.groupby("match_id", sort=False).size().tolist()

    # min_child_samples=1 because LightGBM's default (20) is too high for our
    # small per-match ranking groups (a match has ~44 players). May need
    # re-tuning once trained on the full real 2012+ dataset (thousands of rows).
    model = lgb.LGBMRanker(objective="lambdarank", min_child_samples=1)
    model.fit(
        df_sorted[STAT_COLUMNS],
        df_sorted["brownlow_votes"],
        group=group_sizes,
    )
    return model


def save_model(model: lgb.LGBMRanker, path: str) -> None:
    model.booster_.save_model(path)


def load_model(path: str) -> lgb.Booster:
    return lgb.Booster(model_file=path)


def predict_match_votes(model, df: pd.DataFrame) -> pd.Series:
    predictions = model.predict(df[STAT_COLUMNS])
    return pd.Series(predictions, index=df.index)
