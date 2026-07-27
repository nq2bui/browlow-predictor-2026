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
wheeloratings.com, squiggle.com.au, AFL CFS API) found, and **verified
against real fetched pages**:

| Stat group | Categories | Verified source | Historical depth |
|---|---|---|---|
| Basic (8) | kicks, handballs, disposals, marks, tackles, goals, behinds, hitouts | afltables.com match pages (`afl/stats/games/{year}/{code}{date}.html`) — columns `KI, HB, DI, MK, GL, BH, HO, TK` | Back to 1984 |
| Mid-tier (4 of 5) | clearances (`CL`), contested possessions (`CP`), contested marks (`CM`), goal assists (`GA`) | **Same afltables.com match page** — these columns are already present alongside the basic 8 and the votes column, one fetch per match | Same page/window as basic stats |
| Mid-tier (1 of 5) | score involvements | footywire.com match_statistics page (`ft_match_statistics?mid=...`), column `SC` | ~2012 onward (confirmed via a 2021 match; earliest coverage not exhaustively tested) |
| Newest (4) | kick-ins, intercept marks, intercept possessions, spoils | **Unconfirmed.** wheeloratings.com renders its stats table via a client-side JS/API call that static fetching cannot see; its glossary confirms contested-possession/contested-mark/intercept-mark/intercept-possession terminology but "spoils" was not found in glossary text at all. Getting a real, working scraper for these 4 requires live browser network inspection (out of scope for v1 — see Modeling approach). | Unknown |
| Brownlow votes (label) | actual 3-2-1 votes per player per match | **Same afltables.com match page**, column `BR` — one fetch gives votes + 12 stats together | Back to 1984 (though `BR` is presumably only populated in the 3-2-1-era rows; earlier eras used different vote systems and are out of scope) |

Also verified: `afltables.com/afl/brownlow/brownlow{year}rbr.html` is a
round-by-round index page that links to every match's stats page for that
season — this is the practical entry point for bulk-harvesting match URLs
per season.

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

**v1 scope decision**: ship with the **12 of 17 stats that are reliably
free and structurally verified** (all 8 basic + CL/CP/CM/GA from afltables,
SC from footywire), rather than blocking the whole project on wheeloratings'
unconfirmed API. The remaining 4 stats (kick-ins, intercept marks, intercept
possessions, spoils) are a documented follow-up once a live browser session
can inspect wheeloratings' actual network traffic (or another source is
found) — this is additive, not a redesign, because the model already
handles missing features as NaN (see Modeling approach).

**Training window**: 2012–present. afltables' 12 stats + votes are
structurally available for this whole window (actual data completeness at
the older end to be confirmed empirically during backfill); footywire's SC
column is confirmed back to at least 2021 and needs a quick empirical check
for earlier seasons during backfill, not blocking the plan.

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
