import json
import re

import pandas as pd

from brownlow.dashboard import render_leaderboard, render_round_matrix


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

    render_leaderboard(leaderboard, round_votes, [], str(output_path))

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


def test_render_round_matrix_builds_ordered_player_round_grid(tmp_path):
    # Leaderboard order is by season total descending; the matrix must preserve
    # this exact row order (Alpha, then Bravo, then Charlie).
    leaderboard = pd.DataFrame({
        "player": ["A. Alpha", "B. Bravo", "C. Charlie"],
        "team": ["Richmond", "Collingwood", "Geelong"],
        "predicted_season_votes": [8, 5, 3],
    })
    # Rounds deliberately mixed so a naive string sort would place "10" before
    # "2". Bravo is MISSING round "10" entirely (didn't play) but has a real
    # votes=0 row in round "2" (played, no votes). Charlie missing round "2".
    round_votes = pd.DataFrame(
        [
            {"player": "A. Alpha", "round": "1", "votes": 3},
            {"player": "A. Alpha", "round": "2", "votes": 2},
            {"player": "A. Alpha", "round": "10", "votes": 3},
            {"player": "B. Bravo", "round": "1", "votes": 2},
            {"player": "B. Bravo", "round": "2", "votes": 0},
            # B. Bravo has NO row for round "10" -> did not play.
            {"player": "C. Charlie", "round": "1", "votes": 1},
            {"player": "C. Charlie", "round": "10", "votes": 2},
            # C. Charlie has NO row for round "2" -> did not play.
        ],
        columns=["player", "round", "votes"],
    )
    output_path = tmp_path / "rounds.html"

    render_round_matrix(leaderboard, round_votes, str(output_path))

    html = output_path.read_text()

    # Players appear in leaderboard order.
    assert html.index("A. Alpha") < html.index("B. Bravo") < html.index("C. Charlie")

    # Round column headers appear in proper numeric order: 1, 2, 10 (not 1, 10, 2).
    header_cells = re.findall(r'<th[^>]*class="round-col"[^>]*>(.*?)</th>', html)
    assert header_cells == ["1", "2", "10"]

    # A "Total" column header exists.
    assert "Total" in html

    # votes=3 cells carry a distinguishing class (not merely the digit 3).
    assert 'class="cell v3"' in html
    # Muted/subdued classes exist for lower vote counts and 0.
    assert 'class="cell v0"' in html

    # "Did not play" placeholder is a distinct class + em-dash, NOT a "0" cell.
    assert 'class="cell dnp"' in html
    # The dnp cell renders the em-dash placeholder, distinct from a real 0.
    dnp_cells = re.findall(r'<td class="cell dnp">(.*?)</td>', html)
    assert dnp_cells and all(c.strip() == "—" for c in dnp_cells)
    v0_cells = re.findall(r'<td class="cell v0">(.*?)</td>', html)
    assert v0_cells and all(c.strip() == "0" for c in v0_cells)

    # Total column equals each player's season total (and their row sum).
    # Alpha: 3+2+3 = 8, Bravo: 2+0 = 2 (but season total is 5 from leaderboard),
    # Charlie: 1+2 = 3. The Total shown comes from the leaderboard season total.
    assert 'class="total">8<' in html
    assert 'class="total">5<' in html
    assert 'class="total">3<' in html

    # Nav link back to the main leaderboard.
    assert 'href="index.html"' in html


def test_render_round_matrix_total_matches_row_sum(tmp_path):
    # When per_round_votes reconciles (as guaranteed by the real pipeline), the
    # Total column equals the sum of the visible round cells.
    leaderboard = pd.DataFrame({
        "player": ["Solo Player"],
        "team": ["Richmond"],
        "predicted_season_votes": [6],
    })
    round_votes = pd.DataFrame(
        [
            {"player": "Solo Player", "round": "1", "votes": 3},
            {"player": "Solo Player", "round": "2", "votes": 3},
        ],
        columns=["player", "round", "votes"],
    )
    output_path = tmp_path / "rounds.html"

    render_round_matrix(leaderboard, round_votes, str(output_path))
    html = output_path.read_text()
    assert 'class="total">6<' in html


def test_render_leaderboard_omits_logo_for_unknown_team(tmp_path):
    leaderboard = pd.DataFrame({
        "player": ["Nobody"],
        "team": ["Fake Team FC"],
        "predicted_season_votes": [5.0],
    })
    round_votes = _round_votes_for(["Nobody"])
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, round_votes, [], str(output_path))

    html = output_path.read_text()
    # Unknown team degrades gracefully: no broken logo <img>, neutral swatch instead.
    assert "logos/" not in html
    assert "team-swatch" in html


def test_render_leaderboard_shows_odds_columns_and_placeholder(tmp_path):
    leaderboard = pd.DataFrame({
        "player": ["N. Daicos", "M. Bontempelli", "J. Nomatch"],
        "team": ["Collingwood", "Western Bulldogs", "Richmond"],
        "predicted_season_votes": [30.0, 25.0, 20.0],
    })
    round_votes = _round_votes_for(["N. Daicos", "M. Bontempelli", "J. Nomatch"])
    # Odds carried on the already-normalized "F. Surname" join key. J. Nomatch is
    # deliberately absent from the market to exercise the placeholder path.
    odds = [
        {"player": "N. Daicos", "decimal_odds": 1.20},
        {"player": "M. Bontempelli", "decimal_odds": 10.00},
    ]
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, round_votes, odds, str(output_path))

    html = output_path.read_text()
    # New column headers present.
    assert "Odds" in html
    assert "Implied %" in html
    # Matched player: raw decimal odds and implied probability both rendered.
    assert "$1.20" in html
    assert "83%" in html  # 1/1.20*100 = 83.33 -> 83%
    assert "$10.00" in html
    assert "10%" in html
    # Unmatched leaderboard player shows an em-dash placeholder, not a crash/blank.
    assert "—" in html
