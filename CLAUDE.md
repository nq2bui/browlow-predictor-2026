# Brownlow Predictor 2026 — Project Context

## Setup

macOS users need `brew install libomp` before `pip install -r requirements.txt`, or LightGBM's native library fails to load on import.

## Overview

Predicts the top 20 finishers in the 2026 AFL Brownlow Medal count. A
LightGBM ranking model trained on 2012+ historical data, scored weekly
against the current season and published as a static leaderboard —
sibling project to `afl-tipster`, same single-repo + GitHub Actions
cron pattern.

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
| Score involvements + intercepts | footywire.com advanced stats | `ft_match_statistics?mid=...&advv=Y`, match IDs enumerated via `ft_match_list?year=...` |

See `docs/superpowers/specs/2026-07-27-brownlow-predictor-design.md` for
full data source research and rationale.

## Design docs

- Spec: `docs/superpowers/specs/2026-07-27-brownlow-predictor-design.md`
- Plan: `docs/superpowers/plans/2026-07-27-brownlow-predictor.md`

## Known Limitations

- **Team names are NOT canonicalized between afltables and footywire.**
  `brownlow/names.py` normalizes player names only. The afltables→footywire
  join is keyed on raw team-name strings at two points: the match-level lookup
  (`footywire_html_by_teams.get((home_team, away_team))` in `backfill_data.py`
  and `weekly_update.py`) and the per-player join in `brownlow/dataset.py`.
  This relies on both sites spelling every team identically. That assumption
  holds for the sample fixtures (Richmond, Carlton, Sydney, Geelong,
  Collingwood) but has **not** been verified against real historical data
  across all 18 AFL teams, where known aliases exist (e.g. "Greater Western
  Sydney" vs "GWS", "Western Bulldogs" vs "Footscray", "Brisbane Lions" vs
  "Brisbane"). A mismatch silently degrades the 2 footywire-derived features
  (`score_involvements`, `intercepts`) to 0 for every player in the affected
  match. As of the final-review fixes, the match-level miss is at least logged
  (`logger.warning` in both entry points) so it is visible in run logs rather
  than fully silent — but nothing corrects it. **Fast-follow:** build and
  verify a real team-name canonicalization map against live data before
  trusting footywire features on the full historical backfill.
