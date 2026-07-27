import pandas as pd
from brownlow.dataset import STAT_COLUMNS
from brownlow.model import train_ranker, save_model, load_model, predict_match_votes


def _fake_training_data() -> pd.DataFrame:
    rows = []
    for match_id in ["m1", "m2", "m3"]:
        for i, votes in enumerate([3, 2, 1, 0]):
            row = {col: (i + 1) * 5 for col in STAT_COLUMNS}
            row["match_id"] = match_id
            row["brownlow_votes"] = votes
            row["player"] = f"{match_id}-player{i}"
            rows.append(row)
    return pd.DataFrame(rows)


def test_train_ranker_returns_fitted_model():
    df = _fake_training_data()
    model = train_ranker(df)
    predictions = predict_match_votes(model, df[df["match_id"] == "m1"])
    assert len(predictions) == 4


def test_save_and_load_model_roundtrip(tmp_path):
    df = _fake_training_data()
    model = train_ranker(df)
    model_path = str(tmp_path / "model.txt")
    save_model(model, model_path)
    loaded = load_model(model_path)
    predictions = predict_match_votes(loaded, df[df["match_id"] == "m1"])
    assert len(predictions) == 4
