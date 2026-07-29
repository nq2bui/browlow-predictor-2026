# Brownlow Predictor 2026 — Project Context

## Setup

macOS users need `brew install libomp` before `pip install -r requirements.txt`, or LightGBM's native library fails to load on import.

After `pip install -r requirements.txt`, run `python -m playwright install chromium` to download the headless browser Playwright drives to render the Sportsbet Brownlow odds page (the pip install only installs the Playwright Python package, not the browser binary). `weekly_update.py` degrades gracefully to no odds if the browser or fetch fails.

## Overview

Predicts the top 20 finishers in the 2026 AFL Brownlow Medal count. A
LightGBM ranking model trained on 2012-2025 historical data (123,166 rows,
2,753 matches), scored weekly against the current season and published as
a static leaderboard — sibling project to `afl-tipster`, same single-repo
+ GitHub Actions cron pattern. See Known Limitations for backtest accuracy
and the pre/post-2026 voting-rule caveat.

## Files

- `brownlow/` — scraping, joining, modeling, and dashboard logic
- `backfill_data.py` — one-time historical data scrape → `data/training_data.parquet`
- `train_model.py` — trains and backtests the model → `model.txt`
- `weekly_update.py` — GitHub Actions cron entry point; scores the latest
  round and regenerates `index.html`

## Data Sources

| Data | Source | Notes |
|------|--------|-------|
| Brownlow votes + 12 stats | afltables.com match pages | `afl/stats/games/{year}/{id}.html`, enumerated via `afl/brownlow/brownlow{year}rbr.html` |
| Score involvements + intercepts | footywire.com advanced stats | `ft_match_statistics?mid=...&advv=Y`, match IDs enumerated via `ft_match_list?year=...`. Verified against real pages: these 2 columns don't exist on footywire's advanced-stats page before **2015** (2012-2014 pages have 11 columns, no SI/ITC; 2015+ have 18). `parse_advanced_stats_page` defaults both to 0 when absent rather than crashing. |

See `docs/superpowers/specs/2026-07-27-brownlow-predictor-design.md` for
full data source research and rationale.

## Design docs

- Spec: `docs/superpowers/specs/2026-07-27-brownlow-predictor-design.md`
- Plan: `docs/superpowers/plans/2026-07-27-brownlow-predictor.md`

## Known Limitations

- **Sportsbet odds scraping and Terms of Service.** The leaderboard displays
  live Sportsbet Brownlow Medal odds (decimal odds + implied probability) next
  to each top-20 player, fetched by rendering the operator's page with a
  headless browser (`brownlow/odds.py`). Sportsbet is a regulated Australian
  gambling operator, and automated scraping of their site likely conflicts with
  their Terms of Service, even though the page's `robots.txt` does not explicitly
  disallow this path. This was a deliberate, informed decision by the project
  owner, not an oversight. The odds are a display-only, current-moment feature
  (no historical odds are stored, and odds are not a model/training feature); a
  fetch failure degrades gracefully to no odds and never blocks the weekly update.

- **Pre-2026 training data reflects pre-2026 voting behavior.** Every
  historical Brownlow vote in `data/training_data.parquet` (2021-2025) was
  cast by umpires who did **not** have Champion Data stat access — that
  access only begins in the 2026 season (see the design spec's Background
  section). If umpires weight stats differently now that they can see them
  directly (e.g. valuing contested possessions or intercepts more heavily),
  historical vote patterns won't fully capture that shift, and there is no
  way to get "pre-2026 votes under 2026 rules" — that data doesn't exist.
  The model is trained on historical data as a reasonable prior/baseline,
  not a guarantee it reflects actual 2026 voting behavior. This is a
  deliberate, accepted limitation for v1 (decided 2026-07-28) rather than a
  bug — a fast-follow option would be periodically retraining/recalibrating
  on real 2026 votes as the season accumulates them, since early 2026 data,
  though small, is the only evidence of the actual new regime.

- **Team-name mismatches between afltables and footywire — RESOLVED.**
  `brownlow/teams.py`'s `canonicalize_team_name` fixes the only 2 real
  aliases that exist across all 18 AFL teams (verified by diffing the full
  real team-name sets from both sites for the 2025 season): afltables
  "Brisbane Lions" ↔ footywire "Brisbane", and afltables "Greater Western
  Sydney" ↔ footywire "GWS". Applied at all 3 join points (`backfill_data.py`,
  `weekly_update.py` match-level lookup, `brownlow/dataset.py` per-player
  join). footywire's arrow-suffixed substitution markers (↗/↙, e.g. "N.
  Vlastuin↗") are also now stripped in `parse_advanced_stats_page` before
  name normalization. Verified against real data: a Brisbane Lions v Geelong
  2025 match went from a 100% footywire miss to 44/46 players correctly
  joined. The previously-measured ~65% join rate (on the pre-fix
  2021-2025 backfill) should improve meaningfully once the full backfill is
  re-run with this fix — re-measure after the next real backfill.

- **New 15th feature: `team_margin`.** Captures Brownlow voting's
  well-documented bias toward players on winning teams — `team_margin` is
  this player's team's final score minus the opponent's (positive = win,
  negative = loss, magnitude = margin), extracted from afltables' match
  summary table via `parse_match_header`'s new `home_score`/`away_score`
  fields. Unlike `score_involvements`/`intercepts`, this comes from
  afltables directly and is available for every match regardless of the
  footywire join outcome. `STAT_COLUMNS` (`brownlow/dataset.py`) is now 15
  entries, not 14.

- **Backtest hit rates on the improved 15-feature, better-joined 2012-2025
  data: 50% (2024 holdout), 25% (2025 holdout)** — both very slightly LOWER
  than the pre-fix 14-feature run (60%/30%). This is not read as evidence the
  join-gap fix or `team_margin` made things worse: `team_margin` is the
  model's 4th-highest feature by gain (`model.feature_importance("gain")`),
  well above several stats that were already trusted (contested_possessions,
  hitouts, score_involvements, clearances), so it's genuinely being used,
  not noise. The top-20 hit-rate backtest compares against only ~20 real
  players per holdout season — a couple of rank swaps swings the percentage
  by 5-10 points, so a single before/after comparison at this sample size
  isn't strong evidence either way. Shipped as-is (decided 2026-07-30)
  since the underlying data quality is objectively better (join rate) and
  the new feature has real, measured signal — but treat both this number
  and the 60%/30% one it replaced as noisy, not as a reliable trend.
  Note `score_involvements`/`intercepts` are still 0 for all of 2012-2014
  regardless of the join fix (these stats don't exist on footywire before
  2015 at all — see Data Sources), so roughly a fifth of the training
  window has only 13 of the 15 features populated.
