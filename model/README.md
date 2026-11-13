# Netflix #1 model (Stage 2)

Run from the repo root: `python3 model/run.py` -> writes `model_output/report.html` and `model_output/latest.json`.

- `run.py` - the model (Stage 1 prior + daily-rank update + Monte Carlo + inefficiency flags)
- `stage1_coefs.json` - Stage 1 regression coefficients (fit on 2021-2026 weekly charts)
- `us_weekly_history.csv` - Netflix US weekly Top 10 movies history
- `title_flags.csv` - known facts per title (Netflix original, catalog, theatrical, non-English)
- `week_config.json` - facts about NEW movies each week (release date, original). Edit weekly.
- `seed_daily.csv` - daily Top 10 lists logged by hand before the laptop logger started
