import html
import json
from datetime import datetime, timezone

import pandas as pd

from brownlow.odds import implied_probability
from brownlow.teams import get_team_info
from brownlow.weekly import round_sort_key

# AFL seasons that have an official, UNNUMBERED "Opening Round" played before
# the AFL's own "Round 1". afltables.com labels that opening round as "Round: 1"
# in its match-header `round` field, which shifts every afltables numeric round
# label ONE AHEAD of the AFL's official label for the whole season (afltables
# "Round 1" == AFL "Opening Round", afltables "Round N" (N>=2) == AFL "Round
# N-1"). This mismatch is DISPLAY-ONLY: `round` is a grouping/display field, not
# a model feature, so predictions/training are unaffected. We correct the shown
# label here for confirmed seasons only. 2026 is confirmed (AFL's 2026 Opening
# Round started 2026-03-05, Round 1 on 2026-03-12); historical seasons are NOT
# relabeled unless/until an Opening Round convention is confirmed for them.
# WARNING to future readers: do not "simplify" this away — the raw afltables
# round number really is off by one for these seasons.
_OPENING_ROUND_SEASONS = {2026}


def display_round_label(round_str: str, season: int) -> str:
    """Map an afltables `round` string to the label shown to users.

    For seasons with an Opening Round (see `_OPENING_ROUND_SEASONS`), afltables'
    "Round 1" is really the AFL's "Opening Round" and every later numeric round
    is one ahead of the AFL's official number, so we shift numeric labels down by
    one for display. Non-numeric labels (finals codes like QF/SF/PF/GF) always
    pass through unchanged. For seasons WITHOUT an Opening Round, a numeric label
    is simply prefixed with "Round " and non-numeric labels pass through.
    """
    round_str = str(round_str)
    if season not in _OPENING_ROUND_SEASONS:
        return f"Round {round_str}" if round_str.isdigit() else round_str
    if round_str == "1":
        return "Opening Round"
    if round_str.isdigit():
        return f"Round {int(round_str) - 1}"
    return round_str  # non-numeric labels (e.g. finals "QF"/"SF"/"PF"/"GF")

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
    /* Distinct "warning" accent for the staleness banner: a muted red-orange,
       deliberately NOT the gold used elsewhere for emphasis, so "stale data"
       reads as a warning rather than ordinary highlighting. */
    --warn: #e8703a;
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
  /* Staleness warning banner. Hidden by default (inline display:none in the
     markup) so a fresh page never flashes it; revealed only by the JS check at
     the bottom of the page when the embedded generation time is >10 days old. */
  .stale-banner {{
    background: rgba(232, 112, 58, 0.12);
    border: 1px solid var(--warn);
    color: var(--warn);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 0 0 20px;
    font-size: 14px;
    font-weight: 600;
    line-height: 1.4;
  }}
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
  /* The leaderboard has many columns (rank, player, team+logo, votes, odds,
     implied %); on a narrow/mobile viewport it would otherwise overflow the
     page. Scroll the table sideways inside this wrapper instead, so the page
     body never scrolls horizontally and the card's rounded corners/border
     (from `.card`'s overflow:hidden) stay intact. */
  .table-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
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
  /* Sortable column headers: clickable, with a small active-direction arrow.
     Vanilla-JS driven (see the script at the bottom), no external library. */
  th.sortable {{ cursor: pointer; user-select: none; }}
  th.sortable:hover {{ color: var(--gold); }}
  .sort-arrow {{ font-size: 9px; color: var(--gold); margin-left: 3px; }}
  /* Model-vs-market divergence marker shown inside the player cell. Up = model
     rates the player notably higher than the market (gold); down = market rates
     them higher than the model (muted). Kept within the existing palette. */
  .diverge {{
    font-size: 11px;
    font-weight: 800;
    margin-right: 6px;
    cursor: help;
    vertical-align: middle;
  }}
  .diverge-up {{ color: var(--gold-bright); }}
  .diverge-down {{ color: var(--muted); }}
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
  /* Narrow viewports: tighten body padding and trim the table's cell padding
     and font a touch so more of the leaderboard fits before horizontal
     scrolling (via `.table-scroll`) kicks in. `th`/`td` keep white-space:nowrap
     implicitly via short values, so columns stay legible while scrolling. */
  @media (max-width: 600px) {{
    body {{ padding: 24px 10px; }}
    h1 {{ font-size: 24px; }}
    thead th {{ padding: 12px 10px; font-size: 10px; }}
    tbody td {{ padding: 10px; font-size: 13px; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Brownlow <span class="accent">Predictor</span> 2026</h1>
  <p class="subtitle">Predicted top 20, updated after each round.</p>
  <p class="nav-link"><a href="rounds.html">View round-by-round matrix for all 20 &rarr;</a></p>
  <div id="stale-banner" class="stale-banner" role="alert" style="display: none;" data-generated-at="{generated_at_iso}">&#9888;&#65039; This leaderboard hasn't updated in over 10 days &mdash; the data may be stale</div>
  <div class="card">
    <div class="table-scroll">
    <table id="leaderboard-table">
      <thead>
        <tr>
          <th class="rank">#</th>
          <th class="sortable" data-sort-index="1" data-sort-type="text">Player<span class="sort-arrow"></span></th>
          <th class="sortable" data-sort-index="2" data-sort-type="text">Team<span class="sort-arrow"></span></th>
          <th class="votes sortable" data-sort-index="3" data-sort-type="number">Predicted votes<span class="sort-arrow"></span></th>
          <th class="odds sortable" data-sort-index="4" data-sort-type="number">Odds<span class="sort-arrow"></span></th>
          <th class="implied sortable" data-sort-index="5" data-sort-type="number">Implied %<span class="sort-arrow"></span></th>
        </tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>
    </div>
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

// Client-side sortable leaderboard columns. Sorts the visible rows by each
// cell's raw data-sort-value (numeric or text) rather than the formatted
// display text, so "$1.20" sorts as 1.20 and "83%" as 83. Rows with no
// underlying value (e.g. players with no Sportsbet odds) always sink to the
// bottom regardless of direction. Default (unsorted) order is model rank.
(function() {{
  const table = document.getElementById("leaderboard-table");
  if (!table) return;
  const tbody = table.querySelector("tbody");
  let sortIndex = null;
  let sortDir = 1;
  function clearArrows() {{
    const arrows = table.querySelectorAll("th.sortable .sort-arrow");
    for (const a of arrows) {{ a.textContent = ""; }}
  }}
  function sortBy(th) {{
    const index = parseInt(th.getAttribute("data-sort-index"), 10);
    const type = th.getAttribute("data-sort-type");
    if (sortIndex === index) {{ sortDir = -sortDir; }}
    else {{ sortIndex = index; sortDir = 1; }}
    const rows = Array.prototype.slice.call(tbody.querySelectorAll("tr"));
    rows.sort(function(a, b) {{
      const av = a.children[index].getAttribute("data-sort-value");
      const bv = b.children[index].getAttribute("data-sort-value");
      const aMiss = av === null || av === "";
      const bMiss = bv === null || bv === "";
      if (aMiss && bMiss) return 0;
      if (aMiss) return 1;
      if (bMiss) return -1;
      let cmp;
      if (type === "number") {{ cmp = parseFloat(av) - parseFloat(bv); }}
      else {{ cmp = av.localeCompare(bv); }}
      return cmp * sortDir;
    }});
    for (const r of rows) {{ tbody.appendChild(r); }}
    clearArrows();
    const arrow = th.querySelector(".sort-arrow");
    if (arrow) {{ arrow.textContent = sortDir === 1 ? "\\u25B2" : "\\u25BC"; }}
  }}
  const headers = table.querySelectorAll("th.sortable");
  for (const th of headers) {{
    th.addEventListener("click", function() {{ sortBy(th); }});
  }}
}})();

// Staleness safeguard. This is a static page regenerated by a weekly (~7-day)
// cron; there is no live server to notice if that cron silently stops. So we
// compare the embedded generation timestamp against the viewer's own clock and
// reveal the (initially hidden) warning banner if the page is more than 10 days
// old — a full missed cycle plus slack. The banner starts hidden and is only
// ever shown here, so a fresh page never flashes a false warning.
(function() {{
  const banner = document.getElementById("stale-banner");
  if (!banner) return;
  const generatedAt = new Date(banner.getAttribute("data-generated-at"));
  if (isNaN(generatedAt.getTime())) return;  // unparseable => don't warn
  const STALE_THRESHOLD_MS = 10 * 24 * 60 * 60 * 1000;  // 10 days
  const ageMs = Date.now() - generatedAt.getTime();
  if (ageMs > STALE_THRESHOLD_MS) {{
    banner.style.display = "";
  }}
}})();
</script>
</body>
</html>
"""

_ROW_TEMPLATE = (
    '        <tr>'
    '<td class="rank" style="border-left-color: {color};">{rank}</td>'
    '<td class="player" data-sort-value="{player_sort}">{diverge}{player}</td>'
    '<td class="team-cell" data-sort-value="{team_sort}"><span class="team-inner">{logo}<span>{team}</span></span></td>'
    '<td class="votes" data-sort-value="{votes_sort}">{votes:.1f}</td>'
    '<td class="odds" data-sort-value="{odds_sort}">{odds}</td>'
    '<td class="implied" data-sort-value="{implied_sort}">{implied}</td>'
    '</tr>'
)

# Placeholder for a leaderboard player not present in the Sportsbet market.
_NO_ODDS = "—"

# Minimum gap between a player's model rank and market rank before we flag the
# disagreement. Among only ~20 players, 5 positions is a substantial move that's
# genuinely interesting to surface without cluttering near-agreements.
_DIVERGENCE_THRESHOLD = 5


def render_leaderboard(
    leaderboard: pd.DataFrame,
    round_votes: pd.DataFrame,
    odds: list[dict],
    output_path: str,
    season: int,
) -> None:
    # Index odds by the already-normalized "F. Surname" join key so a direct
    # string match against the (identically normalized) leaderboard player works.
    odds_by_player = {o["player"]: o["decimal_odds"] for o in odds}

    top20 = leaderboard.head(20)

    # Market rank: rank ONLY the top-20 players who have real odds by decimal
    # odds ascending (lower odds = more likely = better rank). Ties break by
    # model position so the ordering is deterministic. Model rank is the row's
    # 1-based position. A meaningful gap between the two ranks is flagged with a
    # divergence marker. Players with no odds get no market rank (nothing to
    # compare against).
    odds_positions = [
        (i, odds_by_player[str(row.player)])
        for i, row in enumerate(top20.itertuples())
        if str(row.player) in odds_by_player
    ]
    odds_positions.sort(key=lambda pair: (pair[1], pair[0]))
    market_rank_by_index = {
        model_index: rank
        for rank, (model_index, _odds) in enumerate(odds_positions, start=1)
    }

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
            odds_sort = ""
            implied_sort = ""
        else:
            implied = implied_probability(decimal_odds)
            odds_display = "${:.2f}".format(decimal_odds)
            implied_display = "{:.0f}%".format(implied)
            odds_sort = "{:.2f}".format(decimal_odds)
            implied_sort = "{:.4f}".format(implied)

        # Divergence marker (only when the player has a market rank to compare).
        diverge = ""
        model_rank = i + 1
        market_rank = market_rank_by_index.get(i)
        if market_rank is not None and abs(model_rank - market_rank) >= _DIVERGENCE_THRESHOLD:
            title = "Model #{}, Market #{}".format(model_rank, market_rank)
            if model_rank < market_rank:
                # Model rates this player notably higher than the market does.
                cls, symbol = "diverge-up", "▲"
            else:
                # Market rates this player notably higher than the model does.
                cls, symbol = "diverge-down", "▼"
            diverge = '<span class="diverge {cls}" title="{title}">{symbol}</span>'.format(
                cls=cls, title=html.escape(title), symbol=symbol
            )

        rows.append(
            _ROW_TEMPLATE.format(
                rank=model_rank,
                player=html.escape(str(row.player)),
                player_sort=html.escape(str(row.player).lower()),
                team=html.escape(str(row.team)),
                team_sort=html.escape(str(row.team).lower()),
                votes=row.predicted_season_votes,
                votes_sort=html.escape("{:.4f}".format(float(row.predicted_season_votes))),
                color=info["color"],
                logo=logo,
                odds=odds_display,
                odds_sort=odds_sort,
                implied=implied_display,
                implied_sort=implied_sort,
                diverge=diverge,
            )
        )
    rows_html = "\n".join(rows)

    # Detail data is scoped to the top-20 players, listed in leaderboard order so
    # the dropdown's first entry (and thus the default view) is the #1 player.
    top20_players = [str(row.player) for row in top20.itertuples()]

    # Build the per-player round list from round_votes, preserving its existing
    # round ordering (per_round_votes already sorts numeric-then-finals). Only the
    # DISPLAY label is relabeled (via display_round_label) to correct afltables'
    # Opening-Round off-by-one for the current season; the underlying ordering is
    # untouched (rows arrive pre-sorted by the real round value).
    round_data = {name: [] for name in top20_players}
    for rv in round_votes.itertuples():
        player = str(rv.player)
        if player in round_data:
            round_data[player].append(
                {
                    "round": display_round_label(str(rv.round), season),
                    "votes": int(rv.votes),
                }
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

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d %H:%M UTC")
    # Machine-readable ISO 8601 UTC form (with a trailing "Z") for the
    # client-side staleness check; new Date(...) parses this reliably.
    generated_at_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    html_doc = _PAGE_TEMPLATE.format(
        rows=rows_html,
        options=options,
        round_votes_json=round_votes_json,
        timestamp=timestamp,
        generated_at_iso=generated_at_iso,
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
    /* Distinct "warning" accent for the staleness banner: a muted red-orange,
       deliberately NOT the gold used elsewhere for emphasis. */
    --warn: #e8703a;
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
  /* Staleness warning banner. Hidden by default (inline display:none) so a
     fresh page never flashes it; revealed only by the JS check below when the
     embedded generation time is >10 days old. */
  .stale-banner {{
    background: rgba(232, 112, 58, 0.12);
    border: 1px solid var(--warn);
    color: var(--warn);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 0 0 20px;
    font-size: 14px;
    font-weight: 600;
    line-height: 1.4;
  }}
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
  /* Freeze the player-name column so it stays pinned to the left edge of
     .matrix-scroll while the (up to ~23) round columns scroll horizontally
     beside it — otherwise, once scrolled a few columns in, the reader loses
     track of whose row they are on. Only the name column is frozen (the narrow
     rank "#" column scrolls underneath it). Each frozen cell needs a SOLID
     background because the round cells scroll UNDER it and both the default
     thead fill and the row-hover tint are translucent — a transparent frozen
     cell would let the scrolling columns show through. The right border marks
     where the frozen column ends and the scrollable region begins. z-index
     stays low (1/2) — the only positioned elements on the page — so the frozen
     cells sit above the scrolling content without shadowing any other UI. */
  thead th.player-name-col,
  tbody td.player {{
    position: sticky;
    left: 0;
    border-right: 1px solid var(--line);
  }}
  tbody td.player {{ background: var(--panel); z-index: 1; }}
  /* The header cell matches the darker thead fill (grass) rather than the
     lighter panel used for body cells, so the frozen corner reads as part of
     the header strip. */
  thead th.player-name-col {{ background: var(--grass); z-index: 2; }}
  /* Keep the frozen name cell opaque on row hover: the shared hover tint is
     translucent gold, which would otherwise let scrolled round cells show
     through the pinned name cell. */
  tbody tr:hover td.player {{ background: var(--panel); }}
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
  <div id="stale-banner" class="stale-banner" role="alert" style="display: none;" data-generated-at="{generated_at_iso}">&#9888;&#65039; This page hasn't updated in over 10 days &mdash; the data may be stale</div>
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
<script>
// Staleness safeguard: this static page is regenerated by a weekly (~7-day)
// cron with no live server to notice if it silently stops. Compare the embedded
// generation timestamp against the viewer's clock and reveal the (initially
// hidden) warning banner if the page is more than 10 days old. The banner starts
// hidden and is only ever shown here, so a fresh page never flashes a warning.
(function() {{
  const banner = document.getElementById("stale-banner");
  if (!banner) return;
  const generatedAt = new Date(banner.getAttribute("data-generated-at"));
  if (isNaN(generatedAt.getTime())) return;  // unparseable => don't warn
  const STALE_THRESHOLD_MS = 10 * 24 * 60 * 60 * 1000;  // 10 days
  const ageMs = Date.now() - generatedAt.getTime();
  if (ageMs > STALE_THRESHOLD_MS) {{
    banner.style.display = "";
  }}
}})();
</script>
</body>
</html>
"""


def render_round_matrix(
    leaderboard: pd.DataFrame,
    round_votes: pd.DataFrame,
    output_path: str,
    season: int,
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
    # Sort by the RAW round value (round_sort_key) so ordering stays correct;
    # the Opening-Round relabeling below is applied only to the displayed header.
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
    header_cells.append(
        '            <th class="player-col player-name-col">Player</th>'
    )
    for rnd in all_rounds:
        header_cells.append(
            '            <th class="round-col">{r}</th>'.format(
                r=html.escape(display_round_label(rnd, season))
            )
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

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d %H:%M UTC")
    # Machine-readable ISO 8601 UTC form for the client-side staleness check.
    generated_at_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    html_doc = _MATRIX_TEMPLATE.format(
        header=header_html,
        rows=rows_html,
        timestamp=timestamp,
        generated_at_iso=generated_at_iso,
    )
    with open(output_path, "w") as f:
        f.write(html_doc)
