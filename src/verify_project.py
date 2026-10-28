"""Meaningful integrity and same-week leakage checks for the forecasting pipeline."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import ROOT, STATS, feature_columns, load_data, make_features

stats, games = load_data()
# A small real-data slice makes the leakage check cheap and reproducible.
small_stats = stats[(stats.season == 2026) & stats.week.le(3)].copy()
small_games = games[(games.season == 2026) & games.week.le(4)].copy()
small_games.loc[small_games.week == 4, ['home_score', 'away_score']] = np.nan
a, future_a, _ = make_features(small_stats, small_games)
changed = small_stats.copy()
changed.loc[changed.week == 3, 'fantasy_points_ppr'] += 77
b, future_b, _ = make_features(changed, small_games)
cols = feature_columns(a)
keys = ['player_id', 'season', 'week']
left = a[a.week == 3].sort_values(keys).reset_index(drop=True)
right = b[b.week == 3].sort_values(keys).reset_index(drop=True)
assert len(left) > 100
assert_frame_equal(left[keys + cols], right[keys + cols], check_exact=True)
assert not np.allclose(left.target, right.target), 'Perturbation must change outcomes.'
assert not np.allclose(future_a.fantasy_points_ppr_mean3, future_b.fantasy_points_ppr_mean3), 'Later forecasts should respond to new history.'
from feature_engineering import enhance
enhanced_a, next_a, extras = enhance(a, future_a, small_stats)
enhanced_b, next_b, _ = enhance(b, future_b, changed)
ea = enhanced_a[enhanced_a.week==3].sort_values(keys).reset_index(drop=True)
eb = enhanced_b[enhanced_b.week==3].sort_values(keys).reset_index(drop=True)
assert_frame_equal(ea[keys+extras],eb[keys+extras],check_exact=True)
assert not np.allclose(next_a.v2_ppr_ewm3,next_b.v2_ppr_ewm3)
pred = pd.read_csv(ROOT / 'reports/predictions_test.csv',float_precision='round_trip')
features = pd.read_csv(ROOT / 'reports/feature_dataset.csv')
forecasts = pd.read_csv(ROOT / 'reports/forecasts_week8.csv')
report = json.loads((ROOT / 'reports/metrics.json').read_text(encoding='utf-8'))
assert not features.duplicated(keys).any()
assert set(pred.season.unique()) == {2025, 2026}
assert forecasts.week.min() == report['prediction_week']
assert forecasts.week.max() <= report['prediction_week'] + 1
assert np.isfinite(forecasts.prediction).all()
assert np.isfinite(pred.prediction).all()
assert abs(np.mean(abs(pred.target-pred.prediction))-report['test']['mae']) < 1e-10
assert '2026_07_DAL_PHI' not in set(features.game_id)
assert not features[(features.season == 2026)&(features.week == 7)].team.isin(['DAL','PHI']).any()
assert (forecasts.prediction >= 0).all()
assert len(forecasts) == len(json.loads((ROOT / 'dist/data.json').read_text(encoding='utf-8'))['players'])
if report.get('version') in ('v2','v3','v5','v6'):
    import joblib
    bundle=joblib.load(ROOT/'models/evaluation_model.joblib')
    sample=pred.sample(50,random_state=42)
    assert np.allclose(bundle['model'].predict(sample[bundle['columns']]),sample.prediction,atol=1e-10)
    old=pd.read_csv(ROOT/'reports/v1/predictions_test.csv')
    assert len(old)==len(pred) and np.allclose(old.target,pred.target)
    assert report['protocol']['test_status'].startswith('Previously inspected')
result = {'checks': {'same_week_outcomes_do_not_change_same_week_features': True,
 'v2_enhanced_features_have_no_same_week_target_leakage': True,
 'later_forecasts_respond_to_new_history': True, 'unique_player_season_week': True,
 'test_seasons_frozen': True, 'missing_source_game_excluded_not_zero': True,
 'reported_mae_matches_predictions': True, 'finite_next_game_forecasts': True},
 'test_n': len(pred), 'forecast_n': len(forecasts)}
(ROOT / 'reports/verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
