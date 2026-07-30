import json
import re
from datetime import datetime

import pandas as pd

from brownlow.dashboard import (
    display_round_label,
    render_leaderboard,
    render_round_matrix,
)


def test_display_round_label_opening_round_season_shifts_numeric_labels():
    # 2026 has an unnumbered AFL Opening Round that afltables labels "Round 1",
    # shifting every afltables numeric round one ahead of the AFL's own label.
    assert display_round_label("1", 2026) == "Opening Round"
    assert display_round_label("2", 2026) == "Round 1"
    assert display_round_label("21", 2026) == "Round 20"
    # Non-numeric finals codes pass through unchanged even for a 2026 season.
    assert display_round_label("QF", 2026) == "QF"
    assert display_round_label("GF", 2026) == "GF"


def test_display_round_label_non_opening_round_season_is_not_shifted():
    # A season NOT in _OPENING_ROUND_SEASONS must NOT get Opening-Round treatment:
    # "1" stays "Round 1" (this guards against applying the fix to history).
    assert display_round_label("1", 2015) == "Round 1"
    assert display_round_label("2", 2015) == "Round 2"
    assert display_round_label("21", 2015) == "Round 21"
    # Non-numeric labels pass through unchanged for non-opening seasons too.
    assert display_round_label("QF", 2015) == "QF"


def _team_votes_blob(html):
    """Extract and parse the embedded ``const TEAM_VOTES = {...};`` JSON blob."""
    marker = "const TEAM_VOTES ="
    start = html.index(marker) + len(marker)
    blob = html[start:html.index(";", start)].strip()
    return json.loads(blob)


def test_render_leaderboard_writes_top_20_table(tmp_path):
    leaderboard = pd.DataFrame({
        "player": [f"Player{i}" for i in range(25)],
        "team": ["Richmond"] * 25,
        "predicted_season_votes": [25 - i for i in range(25)],
    })
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, [], str(output_path), 2026)

    html = output_path.read_text()
    # The leaderboard TABLE shows only the top 20 (Player0..Player19 as rows);
    # Player20 must not be a leaderboard row (">Player20<" is the row cell form).
    assert ">Player0<" in html
    assert ">Player19<" in html
    assert ">Player20<" not in html
    assert "<table" in html
    # Richmond's logo path and brand-color accent render for a known team.
    assert "logos/RIC.png" in html
    assert "#FFD200" in html

    # The team tally, by contrast, covers EVERY scored player for the club, not
    # just the top 20 -- so Player20..Player24 DO appear in the embedded blob.
    data = _team_votes_blob(html)
    richmond_names = [p["player"] for p in data["Richmond"]["players"]]
    assert "Player20" in richmond_names
    assert "Player24" in richmond_names
    assert len(richmond_names) == 25


def test_render_leaderboard_removes_round_by_round_section(tmp_path):
    # The old per-player round-by-round dropdown + ROUND_VOTES blob was replaced
    # by the team-tally feature; none of its markers may survive.
    leaderboard = pd.DataFrame({
        "player": ["A. One", "B. Two"],
        "team": ["Richmond", "Collingwood"],
        "predicted_season_votes": [10.0, 5.0],
    })
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, [], str(output_path), 2026)
    html = output_path.read_text()

    assert "ROUND_VOTES" not in html
    assert "player-select" not in html
    assert "Round-by-round" not in html
    assert "Season total" not in html


def test_render_leaderboard_team_tally(tmp_path):
    # FULL scored field spanning several clubs, with more players per club than
    # the top-20 table would ever show, and per club a mix of nonzero-vote
    # players (must appear, sorted votes-descending) and exactly-0 players (must
    # be excluded from that club's tally).
    leaderboard = pd.DataFrame({
        "player": ["A. One", "A. Two", "A. Three", "A. Zero",
                   "B. Solo", "B. Zed",
                   "C. Only"],
        "team": ["Richmond", "Richmond", "Richmond", "Richmond",
                 "Collingwood", "Collingwood",
                 "Geelong"],
        # Deliberately NOT pre-sorted within Richmond (8 before 12) to prove the
        # tally sorts votes-descending itself.
        "predicted_season_votes": [8.0, 12.0, 4.0, 0.0,
                                    5.0, 0.0,
                                    3.0],
    })
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, [], str(output_path), 2026)
    html = output_path.read_text()

    # A team <select> exists and lists every club present in the data...
    assert 'id="team-select"' in html
    assert '<option value="Richmond"' in html
    assert '<option value="Collingwood"' in html
    assert '<option value="Geelong"' in html
    # ...and, being seeded from the canonical 18-club list, also lists clubs not
    # present in this synthetic data.
    assert '<option value="Hawthorn"' in html
    assert '<option value="West Coast"' in html

    # Default selection is the #1-ranked player's club (A. Two, Richmond, 12).
    assert '<option value="Richmond" selected>' in html

    data = _team_votes_blob(html)

    # Richmond: only the three >0 players, sorted votes-descending; A. Zero (0)
    # is excluded entirely.
    assert data["Richmond"]["players"] == [
        {"player": "A. Two", "votes": 12.0},
        {"player": "A. One", "votes": 8.0},
        {"player": "A. Three", "votes": 4.0},
    ]
    richmond_names = [p["player"] for p in data["Richmond"]["players"]]
    assert "A. Zero" not in richmond_names

    # Collingwood: the 0-vote B. Zed is excluded; only B. Solo remains.
    assert data["Collingwood"]["players"] == [{"player": "B. Solo", "votes": 5.0}]

    # Geelong: its single vote-getter.
    assert data["Geelong"]["players"] == [{"player": "C. Only", "votes": 3.0}]

    # A canonical club with no data embeds an empty player list (renders the
    # "no players" placeholder client-side rather than breaking).
    assert data["Hawthorn"]["players"] == []

    # Each team carries its logo code (reused from brownlow/teams.py).
    assert data["Richmond"]["code"] == "RIC"

    # The vanilla-JS handler + heading logo element are present.
    assert 'id="team-body"' in html
    assert 'id="team-logo"' in html
    assert "function renderTeam" in html


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

    render_round_matrix(leaderboard, round_votes, str(output_path), 2026)

    html = output_path.read_text()

    # Players appear in leaderboard order.
    assert html.index("A. Alpha") < html.index("B. Bravo") < html.index("C. Charlie")

    # Headers stay in real season order (sorted by RAW round 1, 2, 10 — not 1, 10,
    # 2) but are DISPLAY-relabeled for the 2026 Opening-Round off-by-one: afltables
    # "1" -> "Opening Round", "2" -> "Round 1", "10" -> "Round 9".
    header_cells = re.findall(r'<th[^>]*class="round-col"[^>]*>(.*?)</th>', html)
    assert header_cells == ["Opening Round", "Round 1", "Round 9"]

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


def test_render_round_matrix_freezes_player_name_column(tmp_path):
    # The matrix is very wide (up to ~23 round columns + Total), scrolled
    # horizontally inside .matrix-scroll. The player-name column must stay
    # pinned to the left edge (position: sticky; left: 0) so the reader never
    # loses track of whose row they are on once scrolled a few columns in.
    leaderboard = pd.DataFrame({
        "player": ["A. Alpha", "B. Bravo"],
        "team": ["Richmond", "Collingwood"],
        "predicted_season_votes": [8, 5],
    })
    round_votes = pd.DataFrame(
        [
            {"player": "A. Alpha", "round": "1", "votes": 3},
            {"player": "B. Bravo", "round": "1", "votes": 2},
        ],
        columns=["player", "round", "votes"],
    )
    output_path = tmp_path / "rounds.html"

    render_round_matrix(leaderboard, round_votes, str(output_path), 2026)
    html = output_path.read_text()

    # The player-name header cell carries a dedicated class so the sticky rule
    # can target it WITHOUT also freezing the neighbouring rank ("#") header,
    # which shares the generic .player-col class.
    assert 'class="player-col player-name-col"' in html

    # Normalise whitespace so "position: sticky" / "position:sticky" both match.
    css = re.sub(r"\s+", " ", html)

    # Both the header cell (.player-name-col) and every body cell (td.player)
    # of the frozen column are targeted by a sticky, left-pinned rule.
    sticky_rule = re.search(
        r"([^{}]*\.player-name-col[^{}]*|[^{}]*td\.player[^{}]*)\{[^{}]*"
        r"position: ?sticky[^{}]*\}",
        css,
    )
    assert sticky_rule, "expected a position:sticky rule on the player-name column"
    assert "left: 0" in sticky_rule.group(0) or "left:0" in sticky_rule.group(0)

    # Confirm BOTH the header selector and the body-cell selector are sticky.
    assert re.search(r"\.player-name-col[^{}]*\{[^{}]*position: ?sticky", css)
    assert re.search(r"td\.player[^{}]*\{[^{}]*position: ?sticky", css)

    # The frozen cells need a SOLID (non-transparent) background from the
    # existing dark-theme palette so scrolling round columns don't show through.
    assert re.search(
        r"td\.player[^{}]*\{[^{}]*background: ?var\(--(panel|grass)\)", css
    )
    assert re.search(
        r"\.player-name-col[^{}]*\{[^{}]*background: ?var\(--(panel|grass)\)", css
    )


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

    render_round_matrix(leaderboard, round_votes, str(output_path), 2026)
    html = output_path.read_text()
    assert 'class="total">6<' in html


def test_render_leaderboard_omits_logo_for_unknown_team(tmp_path):
    leaderboard = pd.DataFrame({
        "player": ["Nobody"],
        "team": ["Fake Team FC"],
        "predicted_season_votes": [5.0],
    })
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, [], str(output_path), 2026)

    html = output_path.read_text()
    # Unknown team degrades gracefully in its leaderboard ROW: no broken logo
    # <img>, neutral swatch instead. (Scope to the row -- the team-tally JS
    # legitimately contains a literal "logos/" string for known clubs.)
    nobody_row = _row_containing(html, ">Nobody<")
    assert "logos/" not in nobody_row
    assert "team-swatch" in nobody_row


def test_render_leaderboard_shows_odds_columns_and_placeholder(tmp_path):
    leaderboard = pd.DataFrame({
        "player": ["N. Daicos", "M. Bontempelli", "J. Nomatch"],
        "team": ["Collingwood", "Western Bulldogs", "Richmond"],
        "predicted_season_votes": [30.0, 25.0, 20.0],
    })
    # Odds carried on the already-normalized "F. Surname" join key. J. Nomatch is
    # deliberately absent from the market to exercise the placeholder path.
    odds = [
        {"player": "N. Daicos", "decimal_odds": 1.20},
        {"player": "M. Bontempelli", "decimal_odds": 10.00},
    ]
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, odds, str(output_path), 2026)

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


def test_render_leaderboard_has_sort_attributes_and_script(tmp_path):
    leaderboard = pd.DataFrame({
        "player": ["N. Daicos", "M. Bontempelli", "J. Nomatch"],
        "team": ["Collingwood", "Western Bulldogs", "Richmond"],
        "predicted_season_votes": [30.0, 25.0, 20.0],
    })
    odds = [
        {"player": "N. Daicos", "decimal_odds": 1.20},
        {"player": "M. Bontempelli", "decimal_odds": 10.00},
    ]
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, odds, str(output_path), 2026)
    html = output_path.read_text()

    # Sortable headers carry the column index + type metadata the JS reads.
    assert 'id="leaderboard-table"' in html
    assert 'class="sortable"' in html or 'sortable"' in html
    assert 'data-sort-index="4"' in html  # odds column
    assert 'data-sort-type="number"' in html

    # Cells expose the RAW value for sorting, distinct from the display text.
    # Votes: raw number, not the "30.0" display-formatted string only.
    assert 'data-sort-value="30.0000"' in html
    # Odds: raw decimal odds (1.20), not the "$1.20" display string.
    assert 'data-sort-value="1.20"' in html
    assert 'data-sort-value="10.00"' in html
    # Implied %: raw probability (1/1.20*100 = 83.33...), not the "83%" string.
    assert 'data-sort-value="83.3333"' in html
    # Player with no odds: empty sort value so it consistently sinks to bottom.
    assert 'class="odds" data-sort-value=""' in html
    assert 'class="implied" data-sort-value=""' in html

    # The sort script logic is present in the output.
    assert "getAttribute(\"data-sort-value\")" in html
    assert "sortBy" in html
    assert "localeCompare" in html


def test_render_leaderboard_embeds_staleness_check(tmp_path):
    # A staleness safeguard: because this is a static page with no server-side
    # process, the only way to notice the weekly cron has stopped running is to
    # compare the embedded generation time against the viewer's browser clock
    # and warn if the gap exceeds 10 days (the cron runs every ~7 days).
    leaderboard = pd.DataFrame({
        "player": ["N. Daicos", "M. Bontempelli"],
        "team": ["Collingwood", "Western Bulldogs"],
        "predicted_season_votes": [30.0, 25.0],
    })
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, [], str(output_path), 2026)
    html = output_path.read_text()

    # 1. Generation timestamp embedded in a machine-readable, JS-parseable form:
    #    an ISO 8601 UTC string in a data-generated-at attribute.
    m = re.search(r'data-generated-at="([^"]+)"', html)
    assert m, "expected a data-generated-at attribute carrying the ISO timestamp"
    iso = m.group(1)
    # Round-trips through strptime with the exact ISO-8601 Z format (which
    # new Date(...) in the browser also parses reliably).
    parsed = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")
    assert parsed.year >= 2026

    # 2. The warning banner element exists but is HIDDEN by default (only the JS
    #    check reveals it, so a fresh page never flashes the warning).
    assert 'id="stale-banner"' in html
    banner = _element_with_id(html, "stale-banner")
    assert "display: none" in banner  # hidden by default in the initial markup
    assert "hasn't updated in over 10 days" in html  # warning message text

    # 3. The staleness-check script is present with the 10-day threshold.
    assert "stale-banner" in html
    assert "getAttribute(\"data-generated-at\")" in html
    assert "10 * 24 * 60 * 60 * 1000" in html  # 10-day threshold in ms


def test_render_round_matrix_embeds_staleness_check(tmp_path):
    leaderboard = pd.DataFrame({
        "player": ["A. Alpha", "B. Bravo"],
        "team": ["Richmond", "Collingwood"],
        "predicted_season_votes": [8, 5],
    })
    round_votes = pd.DataFrame(
        [
            {"player": "A. Alpha", "round": "1", "votes": 3},
            {"player": "B. Bravo", "round": "1", "votes": 2},
        ],
        columns=["player", "round", "votes"],
    )
    output_path = tmp_path / "rounds.html"

    render_round_matrix(leaderboard, round_votes, str(output_path), 2026)
    html = output_path.read_text()

    m = re.search(r'data-generated-at="([^"]+)"', html)
    assert m, "expected a data-generated-at attribute carrying the ISO timestamp"
    parsed = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ")
    assert parsed.year >= 2026

    assert 'id="stale-banner"' in html
    banner = _element_with_id(html, "stale-banner")
    assert "display: none" in banner  # hidden by default in the initial markup
    assert "hasn't updated in over 10 days" in html  # warning message text

    assert "getAttribute(\"data-generated-at\")" in html
    assert "10 * 24 * 60 * 60 * 1000" in html


def _element_with_id(html, element_id):
    """Return the single opening `<...id="element_id"...>` tag as a string."""
    idx = html.index('id="{}"'.format(element_id))
    start = html.rindex("<", 0, idx)
    end = html.index(">", idx) + 1
    return html[start:end]


def test_render_leaderboard_table_is_horizontally_scrollable(tmp_path):
    # On narrow/mobile viewports the multi-column leaderboard must scroll
    # sideways within its own bounded container instead of overflowing the
    # page body. The table is wrapped in a `.table-scroll` div that gets
    # `overflow-x: auto`, sitting inside the existing `.card` wrapper.
    leaderboard = pd.DataFrame({
        "player": ["N. Daicos", "M. Bontempelli"],
        "team": ["Collingwood", "Western Bulldogs"],
        "predicted_season_votes": [30.0, 25.0],
    })
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, [], str(output_path), 2026)
    html = output_path.read_text()

    # The scroll mechanism itself: a class with overflow-x styling.
    assert ".table-scroll" in html
    assert "overflow-x: auto" in html

    # Mobile scaling requires the viewport meta tag to be present.
    assert '<meta name="viewport" content="width=device-width, initial-scale=1.0">' in html

    # The scroll wrapper sits INSIDE the card and directly around the
    # leaderboard table (not around the whole page), so the card's rounded
    # corners/border are preserved while the table scrolls within it.
    card_start = html.index('<div class="card">')
    scroll_start = html.index('<div class="table-scroll">', card_start)
    table_start = html.index('<table id="leaderboard-table">', scroll_start)
    # Ordering: card opens, then scroll wrapper, then the table.
    assert card_start < scroll_start < table_start


def test_render_leaderboard_divergence_indicator(tmp_path):
    # 8 players in model order (season votes descending). Odds are deliberately
    # REVERSED vs the model, so the model's #1 is the market's worst and vice
    # versa, guaranteeing large model-vs-market rank gaps. P8 has NO odds.
    players = [f"P{i}" for i in range(1, 9)]
    leaderboard = pd.DataFrame({
        "player": players,
        "team": ["Richmond"] * 8,
        "predicted_season_votes": [80, 70, 60, 50, 40, 30, 20, 10],
    })
    # Odds ascending with model rank => market rank is the reverse of model rank
    # for P1..P7. P8 intentionally has no odds.
    odds = [
        {"player": "P1", "decimal_odds": 100.0},  # model #1, market #7
        {"player": "P2", "decimal_odds": 90.0},
        {"player": "P3", "decimal_odds": 80.0},
        {"player": "P4", "decimal_odds": 70.0},   # model #4, market #4 (no gap)
        {"player": "P5", "decimal_odds": 60.0},
        {"player": "P6", "decimal_odds": 50.0},
        {"player": "P7", "decimal_odds": 40.0},   # model #7, market #1
    ]
    output_path = tmp_path / "index.html"

    render_leaderboard(leaderboard, odds, str(output_path), 2026)
    html = output_path.read_text()

    # P1: model #1 vs market #7 (gap 6 >= 5) -> model likes them MORE -> up marker.
    assert 'class="diverge diverge-up" title="Model #1, Market #7"' in html
    # P7: model #7 vs market #1 (gap 6) -> market likes them more -> down marker.
    assert 'class="diverge diverge-down" title="Model #7, Market #1"' in html

    # P4: model #4 vs market #4 (gap 0 < 5) -> NO marker for this player.
    # Isolate P4's row and confirm no diverge span within it.
    p4_row = _row_containing(html, ">P4<")
    assert "diverge" not in p4_row

    # P8: no odds at all -> no market rank -> no divergence marker, and shows the
    # em-dash odds placeholder just like before.
    p8_row = _row_containing(html, ">P8<")
    assert "diverge" not in p8_row
    assert 'data-sort-value=""' in p8_row


def _row_containing(html, needle):
    """Return the single <tr>...</tr> block that contains ``needle``."""
    idx = html.index(needle)
    start = html.rindex("<tr>", 0, idx)
    end = html.index("</tr>", idx) + len("</tr>")
    return html[start:end]
