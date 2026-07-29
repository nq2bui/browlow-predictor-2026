import json

import pandas as pd

from brownlow.dashboard import render_leaderboard


def _round_votes_for(players, rounds=("1", "2", "10")):
    rows = []
    for p in players:
        for r in rounds:
            rows.append({"player": p, "round": r, "votes": 2})
    return pd.DataFrame(rows, columns=["player", "round", "votes"])


def test_render_leaderboard_writes_top_20_table(tmp_path):
    leaderboard = pd.DataFrame({
        "player": [f"Player{i}" for i in range(25)],
        "team": ["Richmond"] * 25,
        "predicted_season_votes": [25 - i for i in range(25)],
    })
    round_votes = _round_votes_for([f"Player{i}" for i in range(25)])
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, round_votes, str(output_path))

    html = output_path.read_text()
    assert "Player0" in html
    assert "Player19" in html
    assert "Player20" not in html  # only top 20 shown
    assert "<table" in html
    # Richmond's logo path and brand-color accent render for a known team.
    assert "logos/RIC.png" in html
    assert "#FFD200" in html

    # Dropdown lists the top-20 players (Player0..Player19), not the rest.
    assert '<option value="Player0">Player0</option>' in html
    assert '<option value="Player19">Player19</option>' in html
    assert '<option value="Player20">' not in html

    # Embedded JSON blob carries the round-by-round data for a top-20 player,
    # scoped to top-20 only (Player20 absent from the blob too).
    assert "const ROUND_VOTES =" in html
    start = html.index("const ROUND_VOTES =") + len("const ROUND_VOTES =")
    blob = html[start:html.index(";", start)].strip()
    data = json.loads(blob)
    assert "Player0" in data
    assert "Player20" not in data
    assert data["Player0"] == [
        {"round": "1", "votes": 2},
        {"round": "2", "votes": 2},
        {"round": "10", "votes": 2},
    ]


def test_render_leaderboard_omits_logo_for_unknown_team(tmp_path):
    leaderboard = pd.DataFrame({
        "player": ["Nobody"],
        "team": ["Fake Team FC"],
        "predicted_season_votes": [5.0],
    })
    round_votes = _round_votes_for(["Nobody"])
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, round_votes, str(output_path))

    html = output_path.read_text()
    # Unknown team degrades gracefully: no broken logo <img>, neutral swatch instead.
    assert "logos/" not in html
    assert "team-swatch" in html
