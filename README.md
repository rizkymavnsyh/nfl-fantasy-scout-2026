# NFL Fantasy Scout 2026

**Next-game PPR forecasting and fantasy lineup exploration**  
Mochamad Rizky Maulana Aviansyah

A personal machine learning portfolio built around NFL fantasy football. Forecast next-game full-PPR points for QB, RB, WR and TE, compare start/sit options, and explore a seven-slot lineup from a selected roster.

The current active model is **V6**, a 50/50 ensemble of a LightGBM regressor and a two-stage hurdle model. V7 red-zone experiments are included as research history but did not pass the promotion rule.

## Explore the dashboard

**Live dashboard:** [https://rizkymavnsyh.github.io/nfl-fantasy-scout-2026/](https://rizkymavnsyh.github.io/nfl-fantasy-scout-2026/)

Download the repository with **Code → Download ZIP** and extract it. Open [`NFL_Fantasy_Scout_2026.html`](NFL_Fantasy_Scout_2026.html) in a browser for the self-contained offline dashboard. GitHub's file viewer displays HTML source rather than running the dashboard.

Alternatively, from the extracted project directory:

```sh
python -m http.server 8766 --directory dist
```

Then open `http://localhost:8766`. Windows users can also run `Buka_Dashboard.cmd`.

Features include a player scout board, start/sit comparison, seven-slot lineup optimization, schedule-aware next-game forecasts, WIB kickoff times, and model/evaluation reports. The offensive lineup contains one QB, two RB, two WR, one TE and one FLEX.

## Verified V6 results

The target is **a player's PPR points in their next scheduled regular-season game**. This is regression, so MAE, RMSE and R² are reported instead of classification accuracy.

| Evaluation | Player-games | MAE, PPR points | RMSE, PPR points | R² |
|---|---:|---:|---:|---:|
| Train, 2019–2024 | 36,290 | 4.1434 | 5.7781 | 0.4900 |
| Mean of validation seasons 2023 and 2024 | 12,228 combined | 4.2688 | 6.0323 | — |
| Retrospective test replay, 2025–2026 | 8,288 | 4.2820 | 6.0253 | 0.4259 |
| Five-game rolling-average baseline, same test rows | 8,288 | 4.7835 | 6.7749 | 0.2741 |

V6 reduced replay MAE by **10.48%** relative to the baseline. On this same replay, **69.94%** of predictions were within five PPR points of the outcome. That tolerance result is not exact-match accuracy or a guarantee for future games.

The selected model and weights were chosen on expanding-season validation: train through 2022 and validate on 2023, then train through 2023 and validate on 2024. The 2025–2026 period had already been inspected during earlier model development, so its score is a **retrospective replay**, not an untouched independent holdout.

Source of truth: [`reports/v6_metrics.json`](reports/v6_metrics.json), [`reports/v6_comparison.json`](reports/v6_comparison.json), and [`reports/v6_validation.json`](reports/v6_validation.json).

## Data and feature timing

Data comes from [nflverse player statistics](https://github.com/nflverse/nflverse-data/releases/tag/stats_player), [nflverse schedules](https://github.com/nflverse/nfldata), offensive snap counts and injury reports. Public source snapshots, provenance and checksums are included.

The 117 model features cover historical usage and form, rolling fantasy points, opponent context, offensive snap share, prior-week injury information, and game-market context such as spread and implied team total.

Historical market features use archived closing lines. New forecasts use lines available at refresh time, with unavailable lines median-imputed. Consequently, historical scores do not establish performance at an earlier weekly prediction cutoff. Archived snap publication timestamps are also incomplete. Current roster status is not a guarantee of game-day participation.

The published forecast snapshot was generated on **28 October 2026 at 16:47 WIB**, using available statistics through Week 7. It is a manually refreshed snapshot, not a continuously updated live service. See [`reports/forecast_snapshot.json`](reports/forecast_snapshot.json). Frozen forecasts and separate score files are retained for later prospective checks.

## Repository map

| Location | Purpose |
|---|---|
| `dist/` | Static dashboard, data payload and downloadable reports |
| `src/` | Data preparation, features, models, refresh, tests and export tools |
| `data/` | Cached public source data and next-game inputs |
| `models/` | Serialized evaluation and forecast models |
| `reports/` | Metrics, validation plans, predictions, provenance and frozen forecasts |
| `THIRD_PARTY/` | Source data attribution and licensing information |
| `NFL_Fantasy_Scout_2026_Mochamad_Rizky_Maulana_Aviansyah.ipynb` | Earlier executed V5 walkthrough, retained as development history |

The supplied notebook predates the V6 upgrade. For the current V6 model and scores, use the dashboard, V6 scripts and JSON reports listed above. Earlier version reports are deliberately preserved and should not be presented as the current model.

## Python environment and reproduction

Python 3.11 was used locally. Install dependencies from the extracted project root:

```sh
python -m pip install -r requirements.txt
```

Cached data and the active serialized models are included. Source URLs and SHA-256 digests are recorded under `reports/`. The scripts resolve paths relative to the repository rather than relying on the original E: location.

Version-upgrade scripts are stateful. To reconstruct the modeling sequence in a **separate copy**, run these in order:

```sh
python src/pipeline.py
python src/upgrade.py
python src/rebalance.py
python src/improve_v4.py
python src/upgrade_v5.py
python src/upgrade_v6.py
```

Each upgrade reads its predecessor's artifacts. Do not rerun `upgrade_v6.py` directly against the already-active V6 bundle. Upstream source data can change, so the included snapshot is the reference for the reported figures. The static dashboard works without retraining or a Python installation.

For a new forecast refresh on Windows, `Perbarui_Prediksi.cmd` updates source data, rebuilds next-game forecasts, checks dates and exports the dashboard. It may take several minutes and requires internet access. Existing historical forecasts remain in `reports/frozen/`.

## Scope

This is a personal portfolio and decision-support prototype. It does not predict game winners, betting returns, full-season totals or playoff results. Rookies without sufficient historical usage, kickers, team defense and IDP are outside the modeled population. Injuries, role changes and unusually high-scoring games remain important sources of error. Real fantasy-league ownership and constraints may differ from the simulated roster setup.

Data attribution: see [`THIRD_PARTY/nflverse-data-CC-BY-4.0.txt`](THIRD_PARTY/nflverse-data-CC-BY-4.0.txt). Third-party data retain their respective terms.
