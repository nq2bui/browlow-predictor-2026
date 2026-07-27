# Brownlow Predictor 2026 — Design Spec

## Purpose

Predict the top 20 players likely to finish highest in the 2026 AFL Brownlow
Medal count, and keep that prediction updated weekly as the season is played.

## Background

Starting in the 2026 AFL season, the four field umpires casting Brownlow
votes each match get supplementary access to 17 official Champion Data
statistical categories on secure, AFL-issued devices immediately post-match
(personal phones banned). The stats are a memory aid for their joint 3-2-1
vote, not a replacement for their subjective judgment. The 17 categories:
kicks, handballs, disposals, marks, contested marks, tackles, goals, behinds,
goal assists, score involvements, clearances, contested possessions, hitouts,
kick-ins, intercept marks, intercept possessions, spoils.

This is a new project, sibling to the existing `afl-tipster` project
(`/Users/nambui/PycharmProjects/afl-tipster`), and reuses its architectural
pattern (single static HTML dashboard + GitHub Actions cron) for consistency.

## Data source findings

Research (via public web sources — afltables.com, footywire.com,
wheeloratings.com, squiggle.com.au, AFL CFS API) found:

| Stat group | Categories | Free source | Historical depth |
|---|---|---|---|
| Basic | kicks, handballs, disposals, marks, tackles, goals, behinds, hitouts | afltables.com | Back to 1984 |
| Mid-tier advanced | contested marks, goal assists, score involvements, clearances, contested possessions | footywire.com | ~2012 onward |
| Newest advanced | kick-ins, intercept marks, intercept possessions, spoils | wheeloratings.com (needs deeper verification of structure/reliability before building the poller) | ~2019-2021 onward, not free/reliable before that |
| Brownlow votes (label) | actual 3-2-1 votes per player per match | afltables.com | Back to 1984 |

Other findings:
- Squiggle API has no per-player stats (team/game-level only, explicitly
  excludes Champion Data advanced stats) — keep using it only for
  fixtures/round detection, as `afl-tipster` already does.
- The AFL CFS API (already used in `afl-tipster` for rosters/injuries) does
  not expose player box-score stats — not usable as a stats source here.
- No formal scraping restriction found on afltables.com (no robots.txt);
  footywire.com's robots.txt blocks specific bots but not general stat
  pages. No explicit ToS blessing either way — poll politely and
  rate-limit, consistent with how `afl-tipster` treats its external APIs.

**Training window**: 2012–present, since that's the first year with the
5 mid-tier advanced stats available. The 4 newest stats (kick-ins, intercept
marks, intercept possessions, spoils) will be missing/NaN for rows before
their respective source's coverage starts (~2019-2021) — see Modeling
approach below for how this is handled.

## Modeling approach

**Single gradient-boosted model (LightGBM), trained on the full 2012+
history, using all 17 features where available and leaving the 4 newest
Champion-Data stats as NaN for seasons before their source's coverage
begins.** LightGBM/XGBoost-style tree models handle missing features
natively (no imputation needed), so this needs only one training pipeline
while still using every season of data available for each feature.

This was chosen over two alternatives:
- *Basic-stats-only, 1984+ history*: simpler and has more historical data,
  but ignores the very stats whose new visibility to umpires is the reason
  this project exists — rejected.
- *Two-model ensemble* (long-history basic-stats model blended with a
  recent-history full-stats model): potentially higher accuracy, but doubles
  pipeline complexity for a hobby-scale project — rejected for now, may be
  revisited later if the single-model approach underperforms in backtesting.

Brownlow votes are awarded per match (3-2-1 among that match's players), so
this is framed as a per-game ranking problem — the model predicts each
player-game's vote-worthiness relative to others in the same match. Season
totals for the leaderboard are the sum of a player's predicted per-game
votes across all matches played so far in 2026.

## Architecture

Same pattern as `afl-tipster`: a GitHub repo with a single static
`index.html` dashboard, hosted via GitHub Pages, updated by a Python script
that runs on a schedule via GitHub Actions and commits the refreshed page
directly to `main`. No separate frontend/backend hosting.

Two pipelines share one trained model artifact:
1. **Backfill/training pipeline** — run once, re-run occasionally to
   retrain as more 2026 data accumulates or data sources change.
2. **Weekly update pipeline** — GitHub Actions cron, runs after each round
   completes: scores the round, updates the leaderboard, pushes.

## Components

- **`backfill_data.py`** — one-time (re-runnable) script. Scrapes
  afltables.com for Brownlow votes + 8 basic stats (2012–2025) and
  footywire.com for the 5 mid-tier advanced stats (2012+). Pulls
  kick-ins/intercept marks/intercept possessions/spoils from
  wheeloratings.com (2019+). Joins everything into one
  row-per-player-per-game training table (17 stat columns + actual votes
  0/1/2/3 as label). Saved as a versioned dataset file (e.g. Parquet).

- **`train_model.py`** — trains the LightGBM ranking model on the backfill
  table, backtests against held-out seasons, saves the trained model
  artifact (e.g. `model.txt`).

- **`weekly_update.py`** — GitHub Actions cron entry point. After each
  round: fetches that round's completed-match stats from the live
  source(s), scores every player-game through the saved model, adds
  predicted votes to a running season-to-date table, re-renders
  `index.html`'s top-20 leaderboard, commits and pushes — mirroring
  `update_simulator.py`'s commit/push pattern in `afl-tipster`.

- **`index.html`** — single-file dashboard: top-20 leaderboard table
  (player, team, predicted season votes, trend), following `afl-tipster`'s
  single-file convention.

- Shared **player-ID mapping** step to reconcile name/team differences
  across data sources (a known pain point already documented in
  `afl-tipster`'s CLAUDE.md — trades, retirements, name variants);
  unmatched names are flagged to a log rather than silently dropped.

## Error handling

- If a live-source scrape fails or returns incomplete data for a round
  (site down, schema change, missing stat column), `weekly_update.py` skips
  the update and leaves the previous week's leaderboard live rather than
  pushing a broken/partial one, logging clearly in the Action run.
- If the live source covering the 4 newest stats proves unreliable, the
  model degrades gracefully: those features become NaN for the round and
  LightGBM predicts off the remaining 13 — no hard failure.
- Player name/team mismatches are resolved via the shared player-ID mapping
  step described above.

## Testing / validation

- **Backtest**: train on seasons 2012–2023, hold out 2024 and 2025 (known
  actual Brownlow results), and measure how many of the actual top-20
  vote-getters the model would have ranked in its predicted top 20. This is
  the primary sanity check before trusting the model for live 2026
  predictions.
- Unit tests for the data-join step (correct player-game alignment across
  sources) and the scoring step (predicted votes non-negative, top-20
  output well-formed).
- No UI testing framework — visual check after each deploy, same as
  `afl-tipster`.

## Open items for implementation phase

- Verify wheeloratings.com's actual structure/reliability/historical depth
  before building the live poller and backfill scraper for the 4 newest
  stats — flagged during research as needing deeper confirmation.
- Confirm cron timing (after which point post-round are all stats final
  and available from the live source) before finalizing the GitHub Actions
  schedule.

## Notes

- Model implementation and coding work for this project should use the
  Opus model, per user preference.
