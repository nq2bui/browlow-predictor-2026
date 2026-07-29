# Brownlow Predictor 2026

Predicts the top 20 players to finish highest in the 2026 AFL Brownlow
Medal count, updated weekly via a LightGBM ranking model trained on
2012–2025 match data.

See `CLAUDE.md` for project structure and data sources, and
`docs/superpowers/specs/2026-07-27-brownlow-predictor-design.md` for the
full design rationale.

## Running locally

```bash
pip install -r requirements.txt
python -m playwright install chromium  # browser for the Sportsbet odds fetch
python backfill_data.py --start-season 2012 --end-season 2025 --output data/training_data.parquet
python train_model.py --data data/training_data.parquet --model-out model.txt
python weekly_update.py
```

`python -m playwright install chromium` downloads the headless browser binary
Playwright drives to render the JS-heavy Sportsbet Brownlow odds page (the
`pip install` above only installs the Playwright Python package, not the
browser). `weekly_update.py` degrades gracefully to no odds if it's missing.

`backfill_data.py` creates the `--output` parent directory (e.g. `data/`) if it
doesn't already exist, so the command above works on a fresh clone.

### Committing the trained model (required for the weekly cron)

The GitHub Actions weekly cron runs **only** `weekly_update.py`, which loads an
already-trained `model.txt` via `load_model("model.txt")`. The Action does
**not** run `backfill_data.py` or `train_model.py` — it scores against a
committed model artifact, it does not train one.

This means you must train the model locally and commit `model.txt` to the
repository before the cron can succeed:

```bash
python train_model.py --data data/training_data.parquet --model-out model.txt
git add model.txt && git commit -m "Add trained model artifact"
git push
```

Without a committed `model.txt`, the weekly Action will fail with a
file-not-found error on `load_model("model.txt")` once real 2026 match data
exists.
