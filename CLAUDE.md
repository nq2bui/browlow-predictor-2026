# Brownlow Predictor 2026 — Project Context

## Setup

macOS users need `brew install libomp` before `pip install -r requirements.txt`, or LightGBM's native library fails to load on import.

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

- **Backtest hit rates were modest (60%/30% on 2024/2025 holdouts) on the
  14-feature, pre-join-fix 2012-2025 backfill.** Re-measure after retraining
  on the improved 15-feature, better-joined data — numbers above are stale
  as of the join-gap fix and `team_margin` addition. Note `score_involvements`/
  `intercepts` are still 0 for all of 2012-2014 regardless of the join fix
  (these stats don't exist on footywire before 2015 at all — see Data
  Sources), so roughly a fifth of the training window has only 13 of the
  15 features populated.
