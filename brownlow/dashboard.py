import html
from datetime import datetime, timezone

import pandas as pd

from brownlow.teams import get_team_info

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
  footer {{
    margin-top: 18px;
    color: var(--muted);
    font-size: 12px;
    text-align: center;
  }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Brownlow <span class="accent">Predictor</span> 2026</h1>
  <p class="subtitle">Predicted top 20, updated after each round.</p>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th class="rank">#</th>
          <th>Player</th>
          <th>Team</th>
          <th class="votes">Predicted votes</th>
        </tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </div>
  <footer>Last updated {timestamp}</footer>
</div>
</body>
</html>
"""

_ROW_TEMPLATE = (
    '        <tr>'
    '<td class="rank" style="border-left-color: {color};">{rank}</td>'
    '<td class="player">{player}</td>'
    '<td class="team-cell"><span class="team-inner">{logo}<span>{team}</span></span></td>'
    '<td class="votes">{votes:.1f}</td>'
    '</tr>'
)


def render_leaderboard(leaderboard: pd.DataFrame, output_path: str) -> None:
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
        rows.append(
            _ROW_TEMPLATE.format(
                rank=i + 1,
                player=html.escape(str(row.player)),
                team=html.escape(str(row.team)),
                votes=row.predicted_season_votes,
                color=info["color"],
                logo=logo,
            )
        )
    rows_html = "\n".join(rows)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html_doc = _PAGE_TEMPLATE.format(rows=rows_html, timestamp=timestamp)
    with open(output_path, "w") as f:
        f.write(html_doc)
