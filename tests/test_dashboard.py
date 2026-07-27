import pandas as pd

from brownlow.dashboard import render_leaderboard


def test_render_leaderboard_writes_top_20_table(tmp_path):
    leaderboard = pd.DataFrame({
        "player": [f"Player{i}" for i in range(25)],
        "team": ["Richmond"] * 25,
        "predicted_season_votes": [25 - i for i in range(25)],
    })
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, str(output_path))

    html = output_path.read_text()
    assert "Player0" in html
    assert "Player19" in html
    assert "Player20" not in html  # only top 20 shown
    assert "<table" in html
