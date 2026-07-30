import html
import json
from datetime import datetime, timezone

import pandas as pd

from brownlow.odds import implied_probability
from brownlow.teams import get_team_info
from brownlow.weekly import round_sort_key

_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brownlow Predictor 2026</title>
<style>
  :root {{
    --grass: #0d1f0d;
    --panel: #12261a;
    --gold: #f0a500;
    --gold-bright: #ffc940;
    --white: #f5f0e8;
    --muted: #7a9a7a;
    --line: rgba(240, 165, 0, 0.14);
    --row-line: rgba(255, 255, 255, 0.06);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--grass);
    color: var(--white);
    margin: 0;
    min-height: 100vh;
    padding: 40px 16px;
  }}
  .wrap {{ max-width: 680px; margin: 0 auto; }}
  h1 {{
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 4px;
    color: var(--white);
  }}
  h1 .accent {{ color: var(--gold); }}
  .subtitle {{ color: var(--muted); font-size: 14px; margin: 0 0 24px; }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
  }}
  table {{ width: 100%; border-collapse: collapse; }}
  thead th {{
    text-align: left;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 14px 16px;
    border-bottom: 1px solid var(--line);
    background: rgba(0, 0, 0, 0.25);
  }}
  tbody td {{
    padding: 11px 16px;
    border-bottom: 1px solid var(--row-line);
    vertical-align: middle;
    font-size: 14px;
  }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover td {{ background: rgba(240, 165, 0, 0.05); }}
  .rank {{
    color: var(--gold);
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    width: 40px;
    text-align: center;
    border-left: 3px solid transparent;
  }}
  .team-cell {{ }}
  .team-inner {{ display: flex; align-items: center; gap: 9px; }}
  .team-logo {{ width: 24px; height: 24px; object-fit: contain; flex-shrink: 0; }}
  .team-swatch {{
    width: 24px; height: 24px; border-radius: 50%; flex-shrink: 0;
    background: var(--muted);
  }}
  .player {{ font-weight: 600; color: var(--white); }}
  .votes {{
    text-align: right;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: var(--gold-bright);
  }}
  th.votes {{ text-align: right; }}
  .odds {{
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: var(--white);
    font-weight: 600;
  }}
  th.odds {{ text-align: right; }}
  .implied {{
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: var(--muted);
  }}
  th.implied {{ text-align: right; }}
  footer {{
    margin-top: 18px;
    color: var(--muted);
    font-size: 12px;
    text-align: center;
  }}
  .detail {{ margin-top: 28px; }}
  .detail h2 {{
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.3px;
    margin: 0 0 12px;
    color: var(--white);
  }}
  .detail h2 .accent {{ color: var(--gold); }}
  .detail-controls {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
  }}
  .detail-controls label {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
  }}
  #player-select {{
    background: var(--panel);
    color: var(--white);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 14px;
    font-family: inherit;
    flex: 1;
    max-width: 320px;
  }}
  #player-select:focus {{ outline: none; border-color: var(--gold); }}
  td.round-cell {{ font-variant-numeric: tabular-nums; }}
  td.running {{
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: var(--muted);
  }}
  th.running {{ text-align: right; }}
  .nav-link {{ margin: 0 0 24px; font-size: 14px; }}
  .nav-link a {{ color: var(--gold); text-decoration: none; font-weight: 600; }}
  .nav-link a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Brownlow <span class="accent">Predictor</span> 2026</h1>
  <p class="subtitle">Predicted top 20, updated after each round.</p>
  <p class="nav-link"><a href="rounds.html">View round-by-round matrix for all 20 &rarr;</a></p>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th class="rank">#</th>
          <th>Player</th>
          <th>Team</th>
          <th class="votes">Predicted votes</th>
          <th class="odds">Odds</th>
          <th class="implied">Implied %</th>
        </tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </div>

  <div class="detail">
    <h2>Round-by-round <span class="accent">breakdown</span></h2>
    <div class="detail-controls">
      <label for="player-select">Player</label>
      <select id="player-select">
{options}
      </select>
    </div>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Round</th>
            <th class="votes">Votes</th>
            <th class="running">Season total</th>
          </tr>
        </thead>
        <tbody id="detail-body"></tbody>
      </table>
    </div>
  </div>

  <footer>Last updated {timestamp}</footer>
</div>
<script>
const ROUND_VOTES = {round_votes_json};
const select = document.getElementById("player-select");
const body = document.getElementById("detail-body");
function renderDetail(player) {{
  const rows = ROUND_VOTES[player] || [];
  let running = 0;
  let out = "";
  for (const r of rows) {{
    running += r.votes;
    out += "<tr><td class=\\"round-cell\\">" + r.round +
           "</td><td class=\\"votes\\">" + r.votes +
           "</td><td class=\\"running\\">" + running + "</td></tr>";
  }}
  if (!out) {{
    out = "<tr><td class=\\"round-cell\\" colspan=\\"3\\">No rounds played yet.</td></tr>";
  }}
  body.innerHTML = out;
}}
select.addEventListener("change", function() {{ renderDetail(select.value); }});
if (select.options.length > 0) {{ renderDetail(select.options[0].value); }}
</script>
</body>
</html>
"""

_ROW_TEMPLATE = (
    '        <tr>'
    '<td class="rank" style="border-left-color: {color};">{rank}</td>'
    '<td class="player">{player}</td>'
    '<td class="team-cell"><span class="team-inner">{logo}<span>{team}</span></span></td>'
    '<td class="votes">{votes:.1f}</td>'
    '<td class="odds">{odds}</td>'
    '<td class="implied">{implied}</td>'
    '</tr>'
)

# Placeholder for a leaderboard player not present in the Sportsbet market.
_NO_ODDS = "—"


def render_leaderboard(
    leaderboard: pd.DataFrame,
    round_votes: pd.DataFrame,
    odds: list[dict],
    output_path: str,
) -> None:
    # Index odds by the already-normalized "F. Surname" join key so a direct
    # string match against the (identically normalized) leaderboard player works.
    odds_by_player = {o["player"]: o["decimal_odds"] for o in odds}

    top20 = leaderboard.head(20)
    rows = []
    for i, row in enumerate(top20.itertuples()):
        info = get_team_info(str(row.team))
        code = info["code"]
        if code:
            logo = (
                '<img class="team-logo" src="logos/{code}.png" alt="{team} logo">'.format(
                    code=code, team=html.escape(str(row.team))
                )
            )
        else:
            logo = '<span class="team-swatch"></span>'

        decimal_odds = odds_by_player.get(str(row.player))
        if decimal_odds is None:
            odds_display = _NO_ODDS
            implied_display = _NO_ODDS
        else:
            odds_display = "${:.2f}".format(decimal_odds)
            implied_display = "{:.0f}%".format(implied_probability(decimal_odds))

        rows.append(
            _ROW_TEMPLATE.format(
                rank=i + 1,
                player=html.escape(str(row.player)),
                team=html.escape(str(row.team)),
                votes=row.predicted_season_votes,
                color=info["color"],
                logo=logo,
                odds=odds_display,
                implied=implied_display,
            )
        )
    rows_html = "\n".join(rows)

    # Detail data is scoped to the top-20 players, listed in leaderboard order so
    # the dropdown's first entry (and thus the default view) is the #1 player.
    top20_players = [str(row.player) for row in top20.itertuples()]

    # Build the per-player round list from round_votes, preserving its existing
    # round ordering (per_round_votes already sorts numeric-then-finals).
    round_data = {name: [] for name in top20_players}
    for rv in round_votes.itertuples():
        player = str(rv.player)
        if player in round_data:
            round_data[player].append(
                {"round": str(rv.round), "votes": int(rv.votes)}
            )
    # json.dumps safely escapes strings for the JS/JSON context (no manual
    # formatting), guarding against injection via player/round values.
    round_votes_json = json.dumps(round_data)

    options = "\n".join(
        '        <option value="{name}">{name}</option>'.format(
            name=html.escape(name)
        )
        for name in top20_players
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html_doc = _PAGE_TEMPLATE.format(
        rows=rows_html,
        options=options,
        round_votes_json=round_votes_json,
        timestamp=timestamp,
    )
    with open(output_path, "w") as f:
        f.write(html_doc)


_MATRIX_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brownlow Predictor 2026 — Round Matrix</title>
<style>
  :root {{
    --grass: #0d1f0d;
    --panel: #12261a;
    --gold: #f0a500;
    --gold-bright: #ffc940;
    --white: #f5f0e8;
    --muted: #7a9a7a;
    --line: rgba(240, 165, 0, 0.14);
    --row-line: rgba(255, 255, 255, 0.06);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--grass);
    color: var(--white);
    margin: 0;
    min-height: 100vh;
    padding: 40px 16px;
  }}
  .wrap {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 4px;
    color: var(--white);
  }}
  h1 .accent {{ color: var(--gold); }}
  .subtitle {{ color: var(--muted); font-size: 14px; margin: 0 0 8px; }}
  .nav-link {{ margin: 0 0 24px; font-size: 14px; }}
  .nav-link a {{ color: var(--gold); text-decoration: none; font-weight: 600; }}
  .nav-link a:hover {{ text-decoration: underline; }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
  }}
  /* Up to ~23 round columns + player + total is wide; scroll the table inside
     this wrapper rather than letting the whole page overflow horizontally. */
  .matrix-scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  thead th {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 12px 10px;
    border-bottom: 1px solid var(--line);
    background: rgba(0, 0, 0, 0.25);
    white-space: nowrap;
  }}
  thead th.player-col {{ text-align: left; }}
  thead th.round-col, thead th.total-col {{ text-align: center; }}
  tbody td {{
    padding: 10px;
    border-bottom: 1px solid var(--row-line);
    font-size: 13px;
    font-variant-numeric: tabular-nums;
    text-align: center;
    white-space: nowrap;
  }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover td {{ background: rgba(240, 165, 0, 0.05); }}
  td.rank {{ color: var(--gold); font-weight: 800; width: 34px; }}
  td.player {{
    text-align: left;
    font-weight: 600;
    color: var(--white);
    white-space: nowrap;
  }}
  /* Vote-count cell emphasis: 3 votes (best on ground) get the gold accent and
     bold so a reader can scan a row for a player's best rounds; 1-2 subdued, 0
     muted, and a round the player did not play the most muted of all. */
  td.cell.v3 {{ color: var(--gold-bright); font-weight: 800; background: rgba(240, 165, 0, 0.12); }}
  td.cell.v2 {{ color: var(--white); font-weight: 700; }}
  td.cell.v1 {{ color: var(--white); font-weight: 600; }}
  td.cell.v0 {{ color: var(--muted); }}
  td.cell.dnp {{ color: rgba(122, 154, 122, 0.4); }}
  td.total {{
    font-weight: 800;
    color: var(--gold);
    border-left: 1px solid var(--line);
  }}
  thead th.total-col {{ border-left: 1px solid var(--line); }}
  footer {{
    margin-top: 18px;
    color: var(--muted);
    font-size: 12px;
    text-align: center;
  }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-top: 14px;
    color: var(--muted);
    font-size: 12px;
  }}
  .legend .swatch {{ font-weight: 800; }}
  .legend .swatch.v3 {{ color: var(--gold-bright); }}
  .legend .swatch.dnp {{ color: rgba(122, 154, 122, 0.55); }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Round-by-round <span class="accent">matrix</span></h1>
  <p class="subtitle">Every top-20 player's discrete votes, round by round.</p>
  <p class="nav-link"><a href="index.html">&larr; Back to leaderboard</a></p>
  <div class="card">
    <div class="matrix-scroll">
      <table>
        <thead>
          <tr>
{header}
          </tr>
        </thead>
        <tbody>
{rows}
        </tbody>
      </table>
    </div>
  </div>
  <div class="legend">
    <span><span class="swatch v3">3</span> best on ground</span>
    <span><span class="swatch">2 / 1</span> minor votes</span>
    <span><span class="swatch">0</span> played, no votes</span>
    <span><span class="swatch dnp">&mdash;</span> did not play</span>
  </div>
  <footer>Last updated {timestamp}</footer>
</div>
</body>
</html>
"""


def render_round_matrix(
    leaderboard: pd.DataFrame,
    round_votes: pd.DataFrame,
    output_path: str,
) -> None:
    """Render a top-20 x per-round vote matrix to a self-contained HTML file.

    Rows are the top-20 players in leaderboard order (season total descending);
    columns are every round any top-20 player played (in real season order via
    ``round_sort_key``) plus a final Total column showing each player's season
    total from the leaderboard.

    "Did not play" vs "played, scored 0" is distinguished directly from the
    ``round_votes`` structure: ``per_round_votes`` emits a row (with ``votes=0``
    where applicable) for every requested player who appeared in a match, so a
    MISSING (player, round) pair means the player did not play that round — shown
    as a muted em-dash placeholder, distinct from a real ``0`` cell.
    """
    top20 = leaderboard.head(20)
    top20_players = [str(row.player) for row in top20.itertuples()]

    # Full set of rounds played by ANY top-20 player, in real season order.
    all_rounds = sorted(
        {str(rv.round) for rv in round_votes.itertuples()}, key=round_sort_key
    )

    # Lookup of votes by (player, round); presence of a key == the player played
    # that round (votes may legitimately be 0).
    votes_by_key = {
        (str(rv.player), str(rv.round)): int(rv.votes)
        for rv in round_votes.itertuples()
    }
    season_total = {
        str(row.player): row.predicted_season_votes for row in top20.itertuples()
    }

    header_cells = ['            <th class="player-col">#</th>']
    header_cells.append('            <th class="player-col">Player</th>')
    for rnd in all_rounds:
        header_cells.append(
            '            <th class="round-col">{r}</th>'.format(r=html.escape(rnd))
        )
    header_cells.append('            <th class="total-col">Total</th>')
    header_html = "\n".join(header_cells)

    rows = []
    for i, player in enumerate(top20_players):
        cells = [
            '<td class="rank">{rank}</td>'.format(rank=i + 1),
            '<td class="player">{name}</td>'.format(name=html.escape(player)),
        ]
        for rnd in all_rounds:
            if (player, rnd) in votes_by_key:
                v = votes_by_key[(player, rnd)]
                cells.append(
                    '<td class="cell v{v}">{v}</td>'.format(v=v)
                )
            else:
                # No row for this (player, round) => did not play that round.
                cells.append('<td class="cell dnp">—</td>')
        total = season_total.get(player, 0)
        # Total is an integer discrete-vote tally; format cleanly whether it
        # arrives as int or float.
        total_display = "{:g}".format(total)
        cells.append('<td class="total">{t}</td>'.format(t=total_display))
        rows.append("            <tr>" + "".join(cells) + "</tr>")
    rows_html = "\n".join(rows)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html_doc = _MATRIX_TEMPLATE.format(
        header=header_html,
        rows=rows_html,
        timestamp=timestamp,
    )
    with open(output_path, "w") as f:
        f.write(html_doc)
