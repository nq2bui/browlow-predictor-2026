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
python backfill_data.py --start-season 2012 --end-season 2025 --output data/training_data.parquet
python train_model.py --data data/training_data.parquet --model-out model.txt
python weekly_update.py
```
