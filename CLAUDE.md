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
| Score involvements + intercepts + 8 more advanced stats | footywire.com advanced stats | `ft_match_statistics?mid=...&advv=Y`, match IDs enumerated via `ft_match_list?year=...`. `parse_advanced_stats_page` extracts 10 columns total from this one already-fetched page: `score_involvements` (SI), `intercepts` (ITC), plus `uncontested_possessions` (UP), `effective_disposals` (ED), `disposal_efficiency` (DE%), `marks_inside_50` (MI5), `one_percenters` (1%), `centre_clearances` (CCL), `metres_gained` (MG), `tackles_inside_50` (T5). Verified against real pages: these columns don't exist on footywire's advanced-stats page before **2015** (2012-2014 pages have 11 columns; 2015+ have 18). `parse_advanced_stats_page` defaults every one of them to 0 when absent rather than crashing. |
| Player position (4 one-hot features) | footywire.com team-roster pages | `tp-{slug}?year={year}`, one page per club per season. Slugs in `brownlow/teams.py`'s `FOOTYWIRE_TEAM_SLUGS` (all 18 clubs, keyed by canonical afltables spelling; note North Melbourne's slug is `kangaroos`). `parse_team_roster` (in `brownlow/footywire.py`) reads the last `<td class="data">` cell of each roster row as the raw position. Supports historical years back to at least 2012. Parsed with **lxml** (not html.parser) because the roster table has a stray extra `</tr>` per row that html.parser mis-nests. `build_position_lookup` (in `brownlow/dataset.py`) fetches these once per run and `add_position_features` joins them onto the assembled DataFrame by `(season, team, player)`. |

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

- **New 16th–19th features: player position (4 one-hot columns).**
  `position_forward`, `position_midfield`, `position_defender`,
  `position_ruck` capture Brownlow voting's well-documented positional bias
  (mids/forwards dominate the medal; defenders/rucks are historically
  underrepresented). Sourced from footywire team-roster pages (see Data
  Sources). A player maps to **exactly one** bucket, or to all-zero when the
  position is missing/unrecognized — a deliberate "unknown" encoding for the
  tree model rather than a 5th column. Combo positions in the real data
  (`MidfieldForward`, `ForwardRuck`, verified live against Richmond 2015)
  resolve to the **primary (first-listed) role** so the encoding stays
  strictly one-hot; `normalize_position` matches case-insensitively and
  handles common abbreviations (`FWD` etc.). The 4 columns are NOT emitted by
  `assemble_match_records` (which stays a pure per-match afltables+footywire
  function); they are joined onto the whole assembled DataFrame afterward by
  `add_position_features`, using a `(season, team, player)` lookup built once
  per run via `build_position_lookup`. Both `backfill_data.py` (full season
  range) and `weekly_update.py` (current season only, 18 fetches) wire this
  in, and degrade gracefully to all-zero position columns for every row if the
  lookup build fails entirely (footywire down). `STAT_COLUMNS` is now **19**
  entries, not 15; `brownlow/model.py` reads `STAT_COLUMNS` generically, so it
  picks the 4 up with no change (the shipped `model.txt` must be retrained
  before it can score against the new 19-column feature set).

- **Position is now per-season historical, not just "current".** Positions
  are fetched from each club's roster page **for the specific season**
  (`tp-{slug}?year={year}`), so a 2015 training row gets that player's 2015
  role, a 2020 row gets their 2020 role, and so on — not a single present-day
  snapshot. This largely resolves the originally-anticipated
  "current-position-may-not-reflect-historical-role" concern for players who
  changed positions across their career (e.g. a forward who became a
  defender): each season's rows use that season's real roster. A residual
  caveat remains only *within* a season — footywire lists one nominal
  position per player for the whole year, so a mid-season role change isn't
  captured, and combo/hybrid players are collapsed to their primary role.

- **Backtest hit rates on the 15-feature, better-joined 2012-2025 data: 50%
  (2024 holdout), 25% (2025 holdout)** — both very slightly LOWER than the
  original pre-fix 14-feature run (60%/30%). This is not read as evidence the
  join-gap fix or `team_margin` made things worse: `team_margin` is the
  model's 4th-highest feature by gain (`model.feature_importance("gain")`),
  well above several stats that were already trusted (contested_possessions,
  hitouts, score_involvements, clearances), so it's genuinely being used,
  not noise. The top-20 hit-rate backtest compares against only ~20 real
  players per holdout season — a couple of rank swaps swings the percentage
  by 5-10 points, so a single before/after comparison at this sample size
  isn't strong evidence either way. Shipped as-is (decided 2026-07-30)
  since the underlying data quality is objectively better (join rate) and
  `team_margin` has real, measured signal — but treat both this number
  and the 60%/30% one it replaced as noisy, not as a reliable trend.
  Note `score_involvements`/`intercepts` are still 0 for all of 2012-2014
  regardless of the join fix (these stats don't exist on footywire before
  2015 at all — see Data Sources), so roughly a fifth of the training
  window has only 13 of the 15 stat features populated.

- **New 20th–27th features: 8 more footywire advanced stats.**
  `parse_advanced_stats_page` (`brownlow/footywire.py`) already fetched
  footywire's advanced-stats page (`ft_match_statistics?...&advv=Y`) per match
  but only extracted 2 of its ~17 columns (`score_involvements`,
  `intercepts`). It now extracts 8 more from the **same already-fetched page**
  (zero new network requests): `uncontested_possessions` (UP),
  `effective_disposals` (ED), `disposal_efficiency` (DE%), `marks_inside_50`
  (MI5), `one_percenters` (1%), `centre_clearances` (CCL), `metres_gained`
  (MG), and `tackles_inside_50` (T5). All 8 use the identical index-lookup +
  `is not None` guard + `.replace(".","").isdigit()` numeric check + default-0
  pattern SI/ITC use, so they degrade gracefully on footywire's pre-2015 pages
  (which lack these columns) and default to 0 when no footywire row joins onto
  an afltables player. **`disposal_efficiency` is a percentage float** (e.g.
  `73.1`, range ~0-100) parsed with `float()`; the other 7 are integer counts.
  All 8 are added to `_FOOTYWIRE_ONLY_COLUMNS` alongside SI/ITC so they get the
  same "default 0 on a footywire miss" treatment in `assemble_match_records`.
  `STAT_COLUMNS` (`brownlow/dataset.py`) is now **27** entries (was 19); they
  sit after `intercepts` and before `team_margin`. `brownlow/model.py` reads
  `STAT_COLUMNS` generically, so it picks the 8 up with no change (the shipped
  `model.txt` must be retrained before it can score against the 27-column set).

- **Position features (16th-19th) add measured signal but very little of
  it.** Retraining on the full 19-feature, 123,166-row dataset (98% of rows —
  120,843 — got a real roster-matched position, only 2 transient network
  timeouts across the whole 14-season backfill) produced the EXACT SAME
  backtest hit rates as the 15-feature run: 50%/25%. `model.feature_importance("gain")`
  confirms why: `position_forward`/`position_midfield`/`position_ruck`/
  `position_defender` rank 124.1/86.1/62.1/45.1 respectively — genuinely
  used (nonzero), but 2-3 orders of magnitude below every other feature
  (the next-lowest, `behinds`, is 212.1; the top feature, `disposals`, is
  34,979.4). Read this as position being largely redundant with the stat
  profile already in the other 15 features — a ruckman's hitout count and a
  defender's disposal/mark pattern already implicitly encode "position-like"
  signal for a tree model, so the explicit one-hot columns add only a small
  residual on top. Shipped as-is (decided 2026-07-30): the feature is
  well-motivated and genuinely non-zero, just not a strong lever on its own.

- **Retrained on the 27-feature dataset (2026-07-31): the 8 new footywire
  stats are genuinely useful, more so than position.** `metres_gained`
  (1088.6 gain) ranks above `marks` and several already-trusted features;
  `effective_disposals` (554.9), `disposal_efficiency` (497.2),
  `centre_clearances` (497.2), `one_percenters` (425.4),
  `uncontested_possessions` (398.4), and `marks_inside_50` (288.9) all
  clear every position feature (55-97). Only `tackles_inside_50` (96.3)
  lands in the same low range as position. `train_model.py`'s own
  hit-rate metric moved to 45%/30% (2024/2025 holdouts, was 50%/25%) —
  another small, likely-noisy swing at this sample size, consistent with
  every retrain so far.

- **`train_model.py`'s reported hit-rate understates real production
  performance — discovered 2026-07-31 via the V2 experiment below.**
  `top20_hit_rate` (what `train_model.py` prints) sums the model's RAW
  continuous per-match scores directly, but the actual production
  leaderboard (`accumulate_season_votes`) converts those scores to
  discrete 3-2-1 votes per match FIRST (via `assign_discrete_match_votes`)
  before summing. Measuring hit-rate the way production actually scores
  (see `compare_vote_schemes.py` below) gives **85% (2024), 80% (2025)**
  on the 27-feature model — dramatically higher than the 45%/30% the
  training script reports. The discrete 3-2-1 conversion step itself is
  doing real, non-trivial work as a denoising/regularizing step on top of
  the raw model scores. **Fast-follow:** `train_model.py`'s own logged
  metric should probably be updated to use the production-realistic
  scoring path instead of raw scores, so the number it reports isn't
  misleadingly pessimistic. **DONE (2026-07-31):** `train_model.py` now
  logs the production-realistic hit rate — it calls
  `top20_hit_rate_with_scheme(model, season_df, assign_discrete_match_votes)`
  (the same discrete 3-2-1 conversion `accumulate_season_votes` uses) and
  labels the line "3-2-1 production scoring". The old raw-score
  `top20_hit_rate` is kept in `brownlow/backtest.py` (its docstring corrected)
  as a baseline and for the per-player-sum aggregation regression test, but is
  no longer on the training-report path. Verified on the identical fresh
  holdout model: raw 45%/30% → 3-2-1 75%/80% (the shipped `model.txt`
  production path is 85%/80%; the 2024 gap is fresh-retrain noise at this
  ~20-sample size, per the retrain caveats above).

- **V2 experiment: ESPN-style fractional voting (3, 2.5, 2, 1.5, 1, 0.5 to
  up to 6 players/match) does NOT beat the standard 3-2-1 scheme.** Added
  `assign_espn_style_votes` (`brownlow/weekly.py`) and
  `top20_hit_rate_with_scheme` (`brownlow/backtest.py`) as a parallel,
  non-production comparison path, plus `compare_vote_schemes.py` to run
  both schemes head-to-head against the same holdout seasons using the
  same trained model (no retraining needed — this is purely a
  scoring-conversion experiment, not a model change). Real result on the
  27-feature model: 2024 — V1 85% vs V2 75% (V1 wins); 2025 — V1 80% vs
  V2 80% (tied). **Production stays on the 3-2-1 scheme** — the finer
  fractional granularity ESPN uses didn't translate into a better top-20
  match against real historical outcomes on this data. `assign_espn_style_votes`
  and the comparison harness are kept in the codebase for future
  re-testing (e.g. once more 2026-specific data exists) but are not wired
  into `weekly_update.py` or the dashboard.
