import pandas as pd
from brownlow.dataset import STAT_COLUMNS
from brownlow.model import train_ranker
from brownlow.backtest import top20_hit_rate


def _season_df_with_known_top_scorer() -> pd.DataFrame:
    rows = []
    for match_num in range(25):
        match_id = f"m{match_num}"
        # "Star" gets 3 votes almost every match with high stats; others vary
        star_row = {col: 20 for col in STAT_COLUMNS}
        star_row.update({"match_id": match_id, "brownlow_votes": 3, "player": "Star"})
        others_row = {col: 5 for col in STAT_COLUMNS}
        others_row.update({"match_id": match_id, "brownlow_votes": 0, "player": f"Other{match_num}"})
        rows.extend([star_row, others_row])
    return pd.DataFrame(rows)


def test_top20_hit_rate_finds_the_obvious_top_scorer():
    df = _season_df_with_known_top_scorer()
    model = train_ranker(df)
    hit_rate = top20_hit_rate(model, df)
    assert 0.0 <= hit_rate <= 1.0
    assert hit_rate > 0  # "Star" should be found since it's the only real vote-getter
