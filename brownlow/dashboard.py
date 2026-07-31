import html
import json
from datetime import datetime, timezone

import pandas as pd

from brownlow.odds import implied_probability
from brownlow.teams import TEAM_INFO, get_team_info
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
  /* Scoring-scheme toggle: a small segmented control ("Standard" | "ESPN") above
     the leaderboard card. Standard (the production 3-2-1 scheme) is the default;
     ESPN (the experimental 6-tier fractional scheme) is opt-in for side-by-side
     comparison. Vanilla-JS driven — swaps the precomputed per-scheme row/tally
     data embedded below. Styled in the existing dark/gold palette. */
  .scheme-toggle {{
    display: inline-flex;
    margin: 0 0 16px;
    border: 1px solid var(--line);
    border-radius: 10px;
    overflow: hidden;
    background: var(--panel);
  }}
  .scheme-btn {{
    background: transparent;
    color: var(--muted);
    border: none;
    padding: 9px 22px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: inherit;
    cursor: pointer;
  }}
  .scheme-btn + .scheme-btn {{ border-left: 1px solid var(--line); }}
  .scheme-btn:hover {{ color: var(--white); }}
  .scheme-btn.active {{ background: var(--gold); color: var(--grass); }}
  .scheme-caption {{ color: var(--muted); font-size: 12px; margin: 0 0 16px; }}
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
  #team-select {{
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
  #team-select:focus {{ outline: none; border-color: var(--gold); }}
  /* Selected-team logo shown beside the "Team tally" heading; swapped by the
     vanilla-JS handler on each selection. Reuses the logos/{{code}}.png assets
     via brownlow/teams.py's team codes. */
  .detail h2 .team-logo-lg {{
    width: 26px;
    height: 26px;
    object-fit: contain;
    vertical-align: -6px;
    margin-right: 8px;
  }}
  td.team-player {{ font-weight: 600; color: var(--white); }}
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
  <div class="scheme-toggle" role="group" aria-label="Vote scoring scheme">
    <button type="button" class="scheme-btn active" data-scheme="Standard">Standard</button>
    <button type="button" class="scheme-btn" data-scheme="ESPN">ESPN</button>
  </div>
  <p class="scheme-caption">Standard = official 3-2-1 votes. ESPN = experimental 6-tier fractional votes (comparison only).</p>
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
    <h2><img id="team-logo" class="team-logo-lg" src="" alt="" style="display: none;">Team <span class="accent">tally</span></h2>
    <div class="detail-controls">
      <label for="team-select">Team</label>
      <select id="team-select">
{team_options}
      </select>
    </div>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Player</th>
            <th class="votes">Predicted votes</th>
          </tr>
        </thead>
        <tbody id="team-body"></tbody>
      </table>
    </div>
  </div>

  <footer>Last updated {timestamp}</footer>
</div>
<script>
// ===== Precomputed per-scheme data, embedded server-side =====
// The dashboard shows TWO scoring schemes with a toggle: "Standard" (the
// production official 3-2-1 votes) and "ESPN" (an experimental 6-tier fractional
// scheme), for side-by-side comparison. BOTH schemes' full row/tally data are
// precomputed in Python and embedded here; the toggle just swaps which is shown,
// with NO recomputation in JS. Standard is the default on load.
//   LEADERBOARD_ROWS: scheme -> the complete top-20 <tbody> HTML for that scheme
//     (rank, player, team, votes and the model-vs-market divergence marker are
//     all per-scheme; every dynamic value was html.escaped in Python).
//   TEAM_VOTES_BY_SCHEME: scheme -> team -> {{code, players}}, players being the
//     list of {{player, votes}} (>0 votes, votes-descending, names html.escaped).
//   DEFAULT_TEAM_BY_SCHEME: scheme -> the club of that scheme's #1-ranked player.
const LEADERBOARD_ROWS = {leaderboard_rows_json};
const TEAM_VOTES_BY_SCHEME = {team_votes_json};
const DEFAULT_TEAM_BY_SCHEME = {default_team_json};

const teamSelect = document.getElementById("team-select");
const teamBody = document.getElementById("team-body");
const teamLogo = document.getElementById("team-logo");
const leaderboardTable = document.getElementById("leaderboard-table");
const leaderboardBody = leaderboardTable ? leaderboardTable.querySelector("tbody") : null;

// Which scheme is currently displayed. Starts on Standard, matching the server-
// rendered initial markup and the button that carries the `active` class.
let currentScheme = "Standard";

// Team tally for the CURRENTLY-selected scheme + team. Player names were
// html.escaped in Python before embedding, so concatenating them into innerHTML
// is safe; team is only ever assigned to element properties.
function renderTeam(team) {{
  const teamData = TEAM_VOTES_BY_SCHEME[currentScheme] || {{}};
  const info = teamData[team] || {{code: "", players: []}};
  if (info.code) {{
    teamLogo.src = "logos/" + info.code + ".png";
    teamLogo.alt = team + " logo";
    teamLogo.style.display = "";
  }} else {{
    teamLogo.removeAttribute("src");
    teamLogo.style.display = "none";
  }}
  let out = "";
  for (const r of info.players) {{
    out += "<tr><td class=\\"team-player\\">" + r.player +
           "</td><td class=\\"votes\\">" + r.votes.toFixed(1) + "</td></tr>";
  }}
  if (!out) {{
    out = "<tr><td class=\\"team-player\\" colspan=\\"2\\">No players with predicted votes yet.</td></tr>";
  }}
  teamBody.innerHTML = out;
}}
teamSelect.addEventListener("change", function() {{ renderTeam(teamSelect.value); }});

// Client-side sortable leaderboard columns. Sorts the visible rows by each
// cell's raw data-sort-value (numeric or text) rather than the formatted
// display text, so "$1.20" sorts as 1.20 and "83%" as 83. Rows with no
// underlying value (e.g. players with no Sportsbet odds) always sink to the
// bottom regardless of direction. Default (unsorted) order is model rank.
// Sort state lives in this shared scope so a scheme swap (which rebuilds the
// rows) can reset it back to model-rank order.
let sortIndex = null;
let sortDir = 1;
function clearArrows() {{
  const arrows = leaderboardTable ? leaderboardTable.querySelectorAll("th.sortable .sort-arrow") : [];
  for (const a of arrows) {{ a.textContent = ""; }}
}}
function sortBy(th) {{
  if (!leaderboardBody) return;
  const index = parseInt(th.getAttribute("data-sort-index"), 10);
  const type = th.getAttribute("data-sort-type");
  if (sortIndex === index) {{ sortDir = -sortDir; }}
  else {{ sortIndex = index; sortDir = 1; }}
  const rows = Array.prototype.slice.call(leaderboardBody.querySelectorAll("tr"));
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
  for (const r of rows) {{ leaderboardBody.appendChild(r); }}
  clearArrows();
  const arrow = th.querySelector(".sort-arrow");
  if (arrow) {{ arrow.textContent = sortDir === 1 ? "\\u25B2" : "\\u25BC"; }}
}}
if (leaderboardTable) {{
  const headers = leaderboardTable.querySelectorAll("th.sortable");
  for (const th of headers) {{
    th.addEventListener("click", function() {{ sortBy(th); }});
  }}
}}

// Apply a scheme: swap the top-20 rows, the divergence markers (baked into those
// rows), the team-tally data and the toggle's active button. Odds/Implied %
// columns are scheme-independent and already baked identically into each
// scheme's precomputed rows. Rebuilding the rows resets any active column sort
// back to model-rank order, so the sort state + arrows are cleared here.
function applyScheme(scheme) {{
  if (!LEADERBOARD_ROWS[scheme]) return;
  currentScheme = scheme;
  if (leaderboardBody) {{ leaderboardBody.innerHTML = LEADERBOARD_ROWS[scheme]; }}
  sortIndex = null;
  sortDir = 1;
  clearArrows();
  const btns = document.querySelectorAll(".scheme-btn");
  for (const b of btns) {{
    if (b.getAttribute("data-scheme") === scheme) {{ b.classList.add("active"); }}
    else {{ b.classList.remove("active"); }}
  }}
  // Keep the selected club if this scheme knows it; else fall back to this
  // scheme's default team (the club of its #1-ranked player).
  const teamData = TEAM_VOTES_BY_SCHEME[scheme] || {{}};
  if (!(teamSelect.value in teamData) && DEFAULT_TEAM_BY_SCHEME[scheme]) {{
    teamSelect.value = DEFAULT_TEAM_BY_SCHEME[scheme];
  }}
  renderTeam(teamSelect.value);
}}
const schemeButtons = document.querySelectorAll(".scheme-btn");
for (const b of schemeButtons) {{
  b.addEventListener("click", function() {{ applyScheme(b.getAttribute("data-scheme")); }});
}}

// Initial paint: Standard scheme (its default team is already selected in the
// server-rendered <select>), so just render the team tally for that selection.
if (teamSelect.options.length > 0) {{ renderTeam(teamSelect.value); }}

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


# Scheme labels and their fixed display order. "Standard" is the production
# official 3-2-1 scheme (the default view); "ESPN" is the experimental 6-tier
# fractional scheme, shown only for side-by-side comparison via the toggle.
_STANDARD_SCHEME = "Standard"
_ESPN_SCHEME = "ESPN"
_SCHEMES = (_STANDARD_SCHEME, _ESPN_SCHEME)


def _build_scheme_view(leaderboard: pd.DataFrame, odds_by_player: dict):
    """Precompute one scheme's top-20 rows HTML + per-club tally + default team.

    ``leaderboard`` is that scheme's FULL scored field (every player), already
    sorted ``predicted_season_votes`` descending. Returns
    ``(rows_html, players_by_team, default_team)`` where ``rows_html`` is the
    complete top-20 ``<tbody>`` markup for this scheme (rank/player/team/votes and
    the model-vs-market divergence marker are all computed against THIS scheme's
    ranking), ``players_by_team`` maps club -> votes-descending list of
    ``{player, votes}`` for its >0-vote players, and ``default_team`` is the club
    of this scheme's #1-ranked player. Odds/Implied % come from the shared,
    scheme-independent ``odds_by_player`` and are identical across schemes for a
    given player (only which player sits in each row changes).
    """
    top20 = leaderboard.head(20)

    # Market rank: rank ONLY the top-20 players who have real odds by decimal
    # odds ascending (lower odds = more likely = better rank). Ties break by
    # model position so the ordering is deterministic. Model rank is the row's
    # 1-based position under THIS scheme. A meaningful gap between the two ranks
    # is flagged with a divergence marker; because the model ranking differs per
    # scheme, the marker is recomputed per scheme here.
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

    # Team-tally data. Group the FULL scored field (not just the top 20) by club,
    # keeping only players with MORE THAN 0 predicted votes, sorted
    # votes-descending. leaderboard is already sorted votes-descending overall,
    # so each club's players arrive in that order; we sort again defensively.
    qualifying = leaderboard[leaderboard["predicted_season_votes"] > 0]
    players_by_team: dict[str, list[dict]] = {}
    for row in qualifying.itertuples():
        players_by_team.setdefault(str(row.team), []).append(
            {
                # html.escape here so the names are safe to concatenate straight
                # into innerHTML by the client-side renderTeam handler.
                "player": html.escape(str(row.player)),
                "votes": float(row.predicted_season_votes),
            }
        )
    for lst in players_by_team.values():
        lst.sort(key=lambda d: d["votes"], reverse=True)

    default_team = (
        str(leaderboard.iloc[0]["team"]) if not leaderboard.empty else None
    )
    return rows_html, players_by_team, default_team


def render_leaderboard(
    leaderboards: dict,
    odds: list[dict],
    output_path: str,
    season: int,
) -> None:
    """Render the leaderboard page with a Standard/ESPN scoring-scheme toggle.

    ``leaderboards`` is a dict mapping scheme name -> that scheme's FULL scored
    leaderboard DataFrame, i.e. ``{"Standard": df, "ESPN": df}`` (each with the
    usual ``player, team, predicted_season_votes`` columns, sorted descending).
    BOTH schemes' top-20 rows, divergence markers and per-club team tallies are
    precomputed here and embedded as JSON; the page's toggle swaps between them
    client-side with no recomputation. The initial server-rendered view is
    Standard (the production 3-2-1 scheme); ESPN is opt-in for comparison. Odds
    and Implied % are scheme-independent and shared across both.
    """
    # Index odds by the already-normalized "F. Surname" join key so a direct
    # string match against the (identically normalized) leaderboard player works.
    odds_by_player = {o["player"]: o["decimal_odds"] for o in odds}

    # Precompute each scheme's rows + tally. Standard must always be present;
    # fall back to the Standard leaderboard for ESPN if a caller somehow omits it
    # so the page still renders (the toggle then simply shows identical data).
    standard_lb = leaderboards[_STANDARD_SCHEME]
    scheme_leaderboards = {
        _STANDARD_SCHEME: standard_lb,
        _ESPN_SCHEME: leaderboards.get(_ESPN_SCHEME, standard_lb),
    }

    rows_by_scheme: dict[str, str] = {}
    players_by_team_by_scheme: dict[str, dict] = {}
    default_team_by_scheme: dict[str, str] = {}
    for name in _SCHEMES:
        rows_html, players_by_team, default_team = _build_scheme_view(
            scheme_leaderboards[name], odds_by_player
        )
        rows_by_scheme[name] = rows_html
        players_by_team_by_scheme[name] = players_by_team
        default_team_by_scheme[name] = default_team

    # Canonical 18 AFL clubs (in TEAM_INFO order) form the dropdown, plus any
    # other team actually present in EITHER scheme's data (defensive: unknown/
    # renamed clubs still show up rather than silently vanishing). The dropdown is
    # a single static union; each scheme's tally map is built over this same union
    # so the JS never dereferences an unknown team.
    ordered_teams = list(TEAM_INFO.keys())
    for name in _SCHEMES:
        for team_name in players_by_team_by_scheme[name]:
            if team_name not in ordered_teams:
                ordered_teams.append(team_name)

    team_votes_by_scheme = {
        name: {
            team_name: {
                "code": get_team_info(team_name)["code"],
                "players": players_by_team_by_scheme[name].get(team_name, []),
            }
            for team_name in ordered_teams
        }
        for name in _SCHEMES
    }
    # json.dumps safely escapes for the JS/JSON context; player names are already
    # html.escaped above so they render correctly inside innerHTML.
    team_votes_json = json.dumps(team_votes_by_scheme)
    leaderboard_rows_json = json.dumps(rows_by_scheme)
    default_team_json = json.dumps(default_team_by_scheme)

    # Initial server-rendered view + dropdown selection are Standard's, so the
    # page shows the production scheme with no JS and existing no-JS expectations
    # (and tests) hold; the toggle then swaps in ESPN's precomputed data on click.
    rows_html = rows_by_scheme[_STANDARD_SCHEME]
    default_team = default_team_by_scheme[_STANDARD_SCHEME]
    team_options = "\n".join(
        '        <option value="{name}"{sel}>{name}</option>'.format(
            name=html.escape(team_name),
            sel=" selected" if team_name == default_team else "",
        )
        for team_name in ordered_teams
    )

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d %H:%M UTC")
    # Machine-readable ISO 8601 UTC form (with a trailing "Z") for the
    # client-side staleness check; new Date(...) parses this reliably.
    generated_at_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    html_doc = _PAGE_TEMPLATE.format(
        rows=rows_html,
        team_options=team_options,
        leaderboard_rows_json=leaderboard_rows_json,
        team_votes_json=team_votes_json,
        default_team_json=default_team_json,
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
  /* ESPN-only intermediate tiers (2.5 / 1.5 / 0.5). The Standard 3-2-1 scheme
     never emits these, so the four rules above are untouched and Standard looks
     identical. Only the exact 3.0 keeps the strongest gold (v3); 2.5 (v2h) gets
     a lighter-gold treatment, 1.5 (v1h) sits with the white minor-vote tier, and
     0.5 (v0h) gets the lightest/muted tier — dimmer than a played-but-scoreless
     0 but brighter than a did-not-play em-dash. */
  td.cell.v2h {{ color: var(--gold-bright); font-weight: 800; background: rgba(240, 165, 0, 0.06); }}
  td.cell.v1h {{ color: var(--white); font-weight: 700; }}
  td.cell.v0h {{ color: rgba(122, 154, 122, 0.75); }}
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
  /* Scoring-scheme toggle: same segmented control ("Standard" | "ESPN") used on
     the main leaderboard, in the same dark/gold palette. Standard (production
     3-2-1) is the default; ESPN (experimental 6-tier fractional) is opt-in.
     Vanilla-JS driven — swaps the precomputed per-scheme header + row data
     embedded below; nothing is recomputed client-side. */
  .scheme-toggle {{
    display: inline-flex;
    margin: 0 0 16px;
    border: 1px solid var(--line);
    border-radius: 10px;
    overflow: hidden;
    background: var(--panel);
  }}
  .scheme-btn {{
    background: transparent;
    color: var(--muted);
    border: none;
    padding: 9px 22px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: inherit;
    cursor: pointer;
  }}
  .scheme-btn + .scheme-btn {{ border-left: 1px solid var(--line); }}
  .scheme-btn:hover {{ color: var(--white); }}
  .scheme-btn.active {{ background: var(--gold); color: var(--grass); }}
  .scheme-caption {{ color: var(--muted); font-size: 12px; margin: 0 0 16px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Round-by-round <span class="accent">matrix</span></h1>
  <p class="subtitle">Every top-20 player's votes, round by round.</p>
  <p class="nav-link"><a href="index.html">&larr; Back to leaderboard</a></p>
  <div id="stale-banner" class="stale-banner" role="alert" style="display: none;" data-generated-at="{generated_at_iso}">&#9888;&#65039; This page hasn't updated in over 10 days &mdash; the data may be stale</div>
  <div class="scheme-toggle" role="group" aria-label="Vote scoring scheme">
    <button type="button" class="scheme-btn active" data-scheme="Standard">Standard</button>
    <button type="button" class="scheme-btn" data-scheme="ESPN">ESPN</button>
  </div>
  <p class="scheme-caption">Standard = official 3-2-1 votes. ESPN = experimental 6-tier fractional votes (comparison only).</p>
  <div class="card">
    <div class="matrix-scroll">
      <table>
        <thead>
          <tr id="matrix-head-row">
{header}
          </tr>
        </thead>
        <tbody id="matrix-body">
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
// ===== Precomputed per-scheme matrix data, embedded server-side =====
// Two scoring schemes are shown via the toggle: "Standard" (production 3-2-1)
// and "ESPN" (experimental 6-tier fractional). Because each scheme ranks the
// field independently, the top-20 SET — and therefore both the round columns
// and the rows — can differ between them, so BOTH the header row and the tbody
// are precomputed per scheme in Python and embedded here; the toggle just swaps
// which is shown, with NO recomputation in JS. Every dynamic value was
// html.escaped in Python, so assigning these strings to innerHTML is safe.
// Standard is the default on load (matching the server-rendered markup).
const MATRIX_HEADER = {matrix_header_json};
const MATRIX_ROWS = {matrix_rows_json};

const headRow = document.getElementById("matrix-head-row");
const matrixBody = document.getElementById("matrix-body");

// Apply a scheme: swap the header cells (round columns can differ per scheme),
// the player rows (the top-20 set can differ per scheme), and the toggle's
// active button. Everything is precomputed, so this is a pure innerHTML swap.
function applyScheme(scheme) {{
  if (!MATRIX_ROWS[scheme]) return;
  if (headRow && MATRIX_HEADER[scheme]) {{ headRow.innerHTML = MATRIX_HEADER[scheme]; }}
  if (matrixBody) {{ matrixBody.innerHTML = MATRIX_ROWS[scheme]; }}
  const btns = document.querySelectorAll(".scheme-btn");
  for (const b of btns) {{
    if (b.getAttribute("data-scheme") === scheme) {{ b.classList.add("active"); }}
    else {{ b.classList.remove("active"); }}
  }}
}}
const schemeButtons = document.querySelectorAll(".scheme-btn");
for (const b of schemeButtons) {{
  b.addEventListener("click", function() {{ applyScheme(b.getAttribute("data-scheme")); }});
}}

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


def _cell_class(votes) -> str:
    """Map a per-round vote value to its matrix cell CSS tier class.

    Standard (3-2-1) only ever produces the whole-number tiers v3/v2/v1/v0, so
    it looks exactly as before. ESPN's fractional halves get intermediate tiers:
    only the exact 3.0 keeps the strongest gold (v3); 2.5 -> v2h (lighter gold),
    1.5 -> v1h (white minor-vote tier), 0.5 -> v0h (lightest/muted). ``>=``
    thresholds keep this robust to float noise (e.g. 2.4999999).
    """
    v = float(votes)
    if v >= 3.0:
        return "v3"
    if v >= 2.5:
        return "v2h"
    if v >= 2.0:
        return "v2"
    if v >= 1.5:
        return "v1h"
    if v >= 1.0:
        return "v1"
    if v > 0.0:
        return "v0h"
    return "v0"


def _build_matrix_view(
    leaderboard: pd.DataFrame, round_votes: pd.DataFrame, season: int
):
    """Precompute one scheme's matrix header + body HTML.

    Rows are that scheme's OWN top-20 players in leaderboard order (season total
    descending); columns are every round any of THOSE top-20 players played (in
    real season order via ``round_sort_key``) plus a final Total column showing
    each player's season total from that scheme's leaderboard. Because the two
    schemes rank the field independently, both the round set and the player set
    can differ between them — hence header + body are built per scheme.

    "Did not play" vs "played, scored 0" is distinguished directly from the
    ``round_votes`` structure: ``per_round_votes`` emits a row (with ``votes=0``
    where applicable) for every requested player who appeared in a match, so a
    MISSING (player, round) pair means the player did not play that round — shown
    as a muted em-dash placeholder, distinct from a real ``0`` cell. This holds
    identically under either scheme. Returns ``(header_html, rows_html)``.
    """
    top20 = leaderboard.head(20)
    top20_players = [str(row.player) for row in top20.itertuples()]

    all_rounds = sorted(
        {str(rv.round) for rv in round_votes.itertuples()}, key=round_sort_key
    )

    # Lookup of votes by (player, round); presence of a key == the player played
    # that round (votes may legitimately be 0). Kept as float so ESPN's
    # fractional halves survive — Standard's integers format cleanly either way.
    votes_by_key = {
        (str(rv.player), str(rv.round)): float(rv.votes)
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
                # {:g} shows Standard's integers as "3"/"0" and ESPN's halves as
                # "2.5"/"0.5"; the tier class carries the visual emphasis.
                cells.append(
                    '<td class="cell {cls}">{v}</td>'.format(
                        cls=_cell_class(v), v="{:g}".format(v)
                    )
                )
            else:
                # No row for this (player, round) => did not play that round.
                cells.append('<td class="cell dnp">—</td>')
        total = season_total.get(player, 0)
        # Total may be an int (Standard) or float (ESPN); format cleanly for both.
        total_display = "{:g}".format(total)
        cells.append('<td class="total">{t}</td>'.format(t=total_display))
        rows.append("            <tr>" + "".join(cells) + "</tr>")
    rows_html = "\n".join(rows)
    return header_html, rows_html


def render_round_matrix(
    leaderboards: dict,
    round_votes: dict,
    output_path: str,
    season: int,
) -> None:
    """Render the round-by-round matrix with a Standard/ESPN scoring toggle.

    ``leaderboards`` maps scheme name -> that scheme's FULL scored leaderboard
    DataFrame (``{"Standard": df, "ESPN": df}``), and ``round_votes`` maps scheme
    name -> that scheme's ``per_round_votes`` DataFrame computed with the MATCHING
    ``vote_assigner`` over that scheme's own top-20 players. Because each scheme
    ranks the field independently, the two top-20 SETS (and thus the rows and the
    round columns) can differ, so both schemes' header + body are precomputed here
    and embedded as JSON; the page's toggle swaps between them client-side with no
    recomputation. The initial server-rendered view is Standard (production 3-2-1);
    ESPN is opt-in for comparison.
    """
    # Standard must always be present; fall back to it for ESPN if a caller omits
    # ESPN so the page still renders (the toggle then shows identical data).
    standard_lb = leaderboards[_STANDARD_SCHEME]
    standard_rv = round_votes[_STANDARD_SCHEME]
    scheme_leaderboards = {
        _STANDARD_SCHEME: standard_lb,
        _ESPN_SCHEME: leaderboards.get(_ESPN_SCHEME, standard_lb),
    }
    scheme_round_votes = {
        _STANDARD_SCHEME: standard_rv,
        _ESPN_SCHEME: round_votes.get(_ESPN_SCHEME, standard_rv),
    }

    header_by_scheme: dict[str, str] = {}
    rows_by_scheme: dict[str, str] = {}
    for name in _SCHEMES:
        header_html, rows_html = _build_matrix_view(
            scheme_leaderboards[name], scheme_round_votes[name], season
        )
        header_by_scheme[name] = header_html
        rows_by_scheme[name] = rows_html

    # json.dumps emits each const on a single line (newlines inside the HTML are
    # escaped to \n) and safely escapes for the JS/JSON context.
    matrix_header_json = json.dumps(header_by_scheme)
    matrix_rows_json = json.dumps(rows_by_scheme)

    # Initial server-rendered view is Standard's, so the page shows the production
    # scheme with no JS; the toggle then swaps in ESPN's precomputed data on click.
    header_html = header_by_scheme[_STANDARD_SCHEME]
    rows_html = rows_by_scheme[_STANDARD_SCHEME]

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d %H:%M UTC")
    # Machine-readable ISO 8601 UTC form for the client-side staleness check.
    generated_at_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    html_doc = _MATRIX_TEMPLATE.format(
        header=header_html,
        rows=rows_html,
        matrix_header_json=matrix_header_json,
        matrix_rows_json=matrix_rows_json,
        timestamp=timestamp,
        generated_at_iso=generated_at_iso,
    )
    with open(output_path, "w") as f:
        f.write(html_doc)
