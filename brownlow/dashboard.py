import html
from datetime import datetime, timezone

import pandas as pd

_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Brownlow Predictor 2026</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #ddd; }}
  th {{ color: #555; font-weight: 600; }}
  .rank {{ color: #999; width: 32px; }}
  .votes {{ text-align: right; font-variant-numeric: tabular-nums; }}
  footer {{ margin-top: 24px; color: #999; font-size: 13px; }}
</style>
</head>
<body>
<h1>Brownlow Predictor 2026</h1>
<p>Predicted top 20, updated after each round.</p>
<table>
<thead><tr><th class="rank">#</th><th>Player</th><th>Team</th><th class="votes">Predicted votes</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
<footer>Last updated {timestamp}</footer>
</body>
</html>
"""

_ROW_TEMPLATE = '<tr><td class="rank">{rank}</td><td>{player}</td><td>{team}</td><td class="votes">{votes:.1f}</td></tr>'


def render_leaderboard(leaderboard: pd.DataFrame, output_path: str) -> None:
    top20 = leaderboard.head(20)
    rows_html = "\n".join(
        _ROW_TEMPLATE.format(
            rank=i + 1,
            player=html.escape(str(row.player)),
            team=html.escape(str(row.team)),
            votes=row.predicted_season_votes,
        )
        for i, row in enumerate(top20.itertuples())
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html_doc = _PAGE_TEMPLATE.format(rows=rows_html, timestamp=timestamp)
    with open(output_path, "w") as f:
        f.write(html_doc)
