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

- **Team-name mismatches between afltables and footywire are real and
  measured, not just theoretical.** `brownlow/names.py` normalizes player
  names only; the afltables→footywire join keys on raw team-name strings
  (match-level lookup in `backfill_data.py`/`weekly_update.py`, and the
  per-player join in `brownlow/dataset.py`). Verified against the real
  2021-2025 backfill (46,782 rows, 1,017 matches): about **65% of player-rows
  get real `score_involvements`/`intercepts` data**; the remaining ~35% is a
  mix of match-level team-name mismatches (logged as `logger.warning`, e.g.
  differing home/away conventions or aliases) and per-player misses (mostly
  unused substitutes and footywire's arrow-suffixed sub markers, e.g. "N.
  Vlastuin↗", which don't match afltables' plain name — `normalize_player_name`
  doesn't strip these). This degrades 2 of the 14 features to 0 for the
  affected rows; LightGBM handles the missingness but real join coverage is
  worth improving. **Fast-follow:** strip footywire's ↗/↙ substitution
  markers in `normalize_player_name`, and build a verified team-name alias
  map for the mismatched cases.

- **Backtest hit rates remain modest even on the full 2012-2025 backfill**:
  60% (2024 holdout) and 30% (2025 holdout) of actual top-20 vote-getters
  were also in the model's predicted top-20, trained on 2012-2023
  (~104k rows, up from 55%/25% on a smaller 2021-2023-only training run).
  More history gave a real but modest improvement — accuracy is still
  meaningfully below what a production model would want. Note `score_involvements`/
  `intercepts` are 0 for all of 2012-2014 (these stats don't exist on
  footywire before 2015 — see Data Sources), so roughly a fifth of the
  training window has only 12 of the 14 features populated.
