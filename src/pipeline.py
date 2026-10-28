"""Reproducible pre-game NFL PPR forecasts. All features precede target week."""
from __future__ import annotations
import argparse
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
POSITIONS = ['QB', 'RB', 'WR', 'TE']
STATS = ['fantasy_points_ppr', 'attempts', 'passing_yards', 'passing_tds',
         'passing_interceptions', 'carries', 'rushing_yards', 'rushing_tds',
         'targets', 'receptions', 'receiving_yards', 'receiving_tds', 'target_share']
SOURCES = {
    'stats': 'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv',
    'schedule': 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv',
}
TEAM_ALIASES = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def download(force=False):
    """Cache source bytes and provenance. Disable only environment proxy routing."""
    session = requests.Session()
    session.trust_env = False
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest = []
    targets = [(f'stats_player_week_{y}.csv', SOURCES['stats'].format(season=y)) for y in range(2019, 2027)]
    targets += [('games.csv', SOURCES['schedule'])]
    for name, url in targets:
        path = raw / name
        if force or not path.is_file():
            r = session.get(url, timeout=90)
            r.raise_for_status()
            path.write_bytes(r.content)
        manifest.append({'file': name, 'url': url, 'bytes': path.stat().st_size,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print('Dataset:', name, path.stat().st_size, flush=True)
    dump(ROOT / 'reports/provenance.json', {'retrieved_at': datetime.now(timezone.utc).isoformat(),
         'sources': manifest, 'note': 'Cached downloads retained unless --refresh is specified.'})

def load_data():
    raw = ROOT / 'data/raw'
    frames = [pd.read_csv(raw / f'stats_player_week_{y}.csv', low_memory=False) for y in range(2019, 2027)]
    stats = pd.concat(frames, ignore_index=True)
    stats = stats[(stats.season_type == 'REG') & stats.position.isin(POSITIONS)].copy()
    keys = ['player_id', 'season', 'week']
    assert not stats.duplicated(keys).any(), 'Duplicate player-season-week records must be reviewed.'
    for c in STATS:
        stats[c] = pd.to_numeric(stats[c], errors='coerce').fillna(0)
    stats['team'] = stats.team.replace(TEAM_ALIASES)
    stats['opponent_team'] = stats.opponent_team.replace(TEAM_ALIASES)
    games = pd.read_csv(raw / 'games.csv', low_memory=False)
    games = games[(games.game_type == 'REG') & games.season.between(2019, 2026)].copy()
    games['away_team'] = games.away_team.replace(TEAM_ALIASES)
    games['home_team'] = games.home_team.replace(TEAM_ALIASES)
    assert not games.game_id.duplicated().any()
    return stats, games

def complete_weeks(stats, games, season):
    valid = []
    for week, g in games[games.season == season].groupby('week'):
        supplied = set(stats[(stats.season == season) & (stats.week == week)].game_id)
        if g.home_score.notna().all() and set(g.game_id).issubset(supplied):
            valid.append(int(week))
    return valid

def make_features(stats, games, allow_empty_forecast=False):
    """Simulate weekly forecasting before reading any outcomes from that week.

    Universe: players with an offensive opportunity in the same season, last seen
    <=3 calendar weeks ago, assigned to the last observed team. DNP outcomes in
    this candidate universe are zero. No hindsight-based active-player filter.
    Week 1 and players before first observed use are intentionally outside scope.
    """
    samples, forecasts = [], []
    defense = defaultdict(list)
    latest_complete = max(complete_weeks(stats, games, 2026))
    # Schedule results can arrive before the corresponding player-stat file.
    # Never turn a not-yet-ingested game into zero-point player outcomes.
    closed_2026 = games[(games.season == 2026) & games.home_score.notna()]
    last_closed_week = int(closed_2026.week.max())
    next_week = last_closed_week + 1
    for season in range(2019, 2027):
        histories = defaultdict(list)
        identities = {}
        seen = {}
        sg = games[games.season == season]
        season_stats = stats[stats.season == season]
        valid = set(complete_weeks(stats, games, season))
        observed_weeks = set(int(w) for w in season_stats.week.unique())
        weeks = sorted(observed_weeks | ({next_week} if season == 2026 else set()))
        for week in weeks:
            gw = sg[sg.week == week]
            if not (season == 2026 and week == next_week):
                supplied_games = set(season_stats[season_stats.week == week].game_id)
                gw = gw[gw.game_id.isin(supplied_games) & gw.home_score.notna()]
            schedule = {}
            for game in gw.to_dict('records'):
                schedule[game['home_team']] = (game['away_team'], 1, game['home_rest'], game)
                schedule[game['away_team']] = (game['home_team'], 0, game['away_rest'], game)
            rows = season_stats[season_stats.week == week]
            actual = {r['player_id']: r for r in rows.to_dict('records')}
            candidates = []
            # Histories and defense contain outcomes only from earlier weeks.
            for pid, identity in identities.items():
                team = identity['team']
                if team not in schedule or week - seen[pid] > 3:
                    continue
                opponent, home, rest, game = schedule[team]
                hist = histories[pid]
                x = {'player_id': pid, 'name': identity['player_display_name'],
                     'position': identity['position'], 'team': team, 'opponent': opponent,
                     'season': season, 'week': week, 'game_id': game['game_id'],
                     'date': game['gameday'], 'home': home, 'rest': float(rest) if pd.notna(rest) else 7.,
                     'weeks_since_seen': week - seen[pid], 'history_games': len(hist)}
                for col in STATS:
                    values = [h.get(col, 0.) for h in hist]
                    x[f'{col}_lag1'] = values[-1]
                    x[f'{col}_mean3'] = float(np.mean(values[-3:]))
                    x[f'{col}_mean5'] = float(np.mean(values[-5:]))
                    x[f'{col}_season_mean'] = float(np.mean(values))
                x['ppr_std5'] = float(np.std([h['fantasy_points_ppr'] for h in hist[-5:]]))
                allowed = defense[(opponent, identity['position'])]
                x['opponent_allowed_mean5'] = float(np.mean(allowed[-5:])) if allowed else np.nan
                x['baseline'] = x['fantasy_points_ppr_mean5']
                if season == 2026 and week == next_week:
                    forecasts.append(x)
                else:
                    # A new team this week cannot retroactively change the forecast team.
                    target = actual.get(pid, {})
                    x['target'] = float(target.get('fantasy_points_ppr', 0.))
                    x['observed_stats'] = bool(target)
                    samples.append(x)
                candidates.append(pid)
            if season == 2026 and week == next_week:
                continue
            # Append zero for non-appearing candidate players after forecasting.
            for pid in candidates:
                histories[pid].append(actual.get(pid, {c: 0. for c in STATS}))
            for pid, record in actual.items():
                offensive_use = record['attempts'] + record['carries'] + record['targets'] > 0
                if pid not in histories and not offensive_use:
                    continue
                if pid not in candidates:
                    histories[pid].append(record)
                if offensive_use:
                    identities[pid] = record
                    seen[pid] = week
            # Defense derived from completed earlier games. Missing position = 0.
            for team in schedule:
                for pos in POSITIONS:
                    val = rows[(rows.opponent_team == team) & (rows.position == pos)].fantasy_points_ppr.sum()
                    defense[(team, pos)].append(float(val))
        print('Features:', season, 'completed weeks', len(valid), flush=True)
    data = pd.DataFrame(samples)
    future = pd.DataFrame(forecasts)
    assert not data.duplicated(['player_id', 'season', 'week']).any()
    if not allow_empty_forecast:
        assert len(future) > 0
        assert future.week.nunique() == 1
    return data, future, latest_complete

def feature_columns(data):
    cols = [c for c in data if c.endswith(('_lag1', '_mean3', '_mean5', '_season_mean'))]
    return cols + ['home', 'rest', 'weeks_since_seen', 'history_games', 'week', 'ppr_std5', 'position']

def estimator(name, columns):
    numeric = [c for c in columns if c != 'position']
    prep = ColumnTransformer([
        ('numbers', Pipeline([('fill', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('position', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['position']),
    ])
    choices = {
        'ridge': Ridge(alpha=60.),
        'hist_conservative': HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=12,
                  min_samples_leaf=50, l2_regularization=15., learning_rate=.055, early_stopping=False, random_state=42),
        'hist_flexible': HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=20,
                  min_samples_leaf=35, l2_regularization=25., learning_rate=.045, early_stopping=False, random_state=42),
    }
    return Pipeline([('prepare', prep), ('model', choices[name])])

def predict(model, name, frame, columns):
    if name == 'rolling_mean5':
        return frame.baseline.to_numpy()
    return np.maximum(0, model.predict(frame[columns]))

def metrics(y, p):
    return {'n': int(len(y)), 'mae': float(mean_absolute_error(y, p)),
            'rmse': float(np.sqrt(mean_squared_error(y, p))), 'r2': float(r2_score(y, p))}

def train_and_evaluate(data, future, latest_complete):
    cols = feature_columns(data)
    candidates = ['rolling_mean5', 'ridge', 'hist_conservative', 'hist_flexible']
    scores, cv_predictions = [], {}
    for name in candidates:
        predictions = []
        fold_results = []
        for year in [2023, 2024]:
            tr, va = data[data.season < year], data[data.season == year]
            model = None if name == 'rolling_mean5' else estimator(name, cols).fit(tr[cols], tr.target)
            p = predict(model, name, va, cols)
            fold_results.append({'validation_season': year, 'train_n': len(tr), **metrics(va.target, p)})
            temp = va[['position', 'target', 'season']].copy()
            temp['prediction'] = p
            predictions.append(temp)
        cv_predictions[name] = pd.concat(predictions)
        scores.append({'model': name, 'cv_mae': float(np.mean([f['mae'] for f in fold_results])), 'folds': fold_results})
        print('CV:', name, scores[-1]['cv_mae'], flush=True)
    chosen = min(scores, key=lambda s: s['cv_mae'])['model']
    train = data[data.season <= 2024]
    test = data[data.season >= 2025].copy()
    model = None if chosen == 'rolling_mean5' else estimator(chosen, cols).fit(train[cols], train.target)
    test['prediction'] = predict(model, chosen, test, cols)
    base = test.baseline.to_numpy()
    # Fixed test data is evaluated after model selection, with no test-driven tuning.
    calibration = cv_predictions[chosen]
    bands = {pos: float(np.quantile(np.abs(g.target - g.prediction), .8))
             for pos, g in calibration.groupby('position')}
    test['band'] = test.position.map(bands)
    test['low'] = np.maximum(0, test.prediction - test.band)
    test['high'] = test.prediction + test.band
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'selected_model': chosen,
        'target': 'Next scheduled game PPR fantasy points',
        'positions': POSITIONS, 'features': cols, 'feature_count': len(cols),
        'protocol': {'training_seasons': '2019–2024', 'validation_seasons': [2023, 2024],
                     'test_seasons': [2025, 2026], 'latest_complete_2026_week': latest_complete,
                     'selection': 'Lowest mean MAE across expanding season folds, before test inspection',
                     'candidate_universe': 'Offensive opportunity earlier in same season, last seen <=3 weeks, team scheduled',
                     'excluded': 'Week 1, unobserved rookies, K, DST, IDP, bye weeks',
                     'injury_info': 'No live injury reports or official active roster filter'},
        'train': metrics(train.target, predict(model, chosen, train, cols)),
        'test': metrics(test.target, test.prediction), 'baseline': metrics(test.target, base),
        'cv_candidates': scores,
        'by_season': {str(int(y)): {'model': metrics(g.target, g.prediction), 'baseline': metrics(g.target, g.baseline)}
                      for y, g in test.groupby('season')},
        'by_position': {pos: {'model': metrics(g.target, g.prediction), 'baseline': metrics(g.target, g.baseline)}
                        for pos, g in test.groupby('position')},
        'interval': {'nominal': .8, 'coverage_test': float(((test.target >= test.low) & (test.target <= test.high)).mean()),
                     'description': 'Symmetric empirical absolute-error band from 2023/24 development folds. No guarantee for individuals.',
                     'half_width_by_position': bands},
        'missing_stat_outcomes': int((~test.observed_stats).sum()),
        'prediction_week': int(future.week.iloc[0]), 'prediction_season': 2026,
    }
    report['protocol']['latest_observed_2026_week'] = int(data[data.season == 2026].week.max())
    report['protocol']['partial_week_note'] = 'Week 7 player stats omit DAL–PHI in this source snapshot. That game is excluded from test labels and histories, never filled with false zeros.'
    report['mae_improvement_pct'] = float(100 * (report['baseline']['mae'] - report['test']['mae']) / report['baseline']['mae'])
    (ROOT / 'models').mkdir(exist_ok=True)
    joblib.dump({'model': model, 'columns': cols, 'name': chosen}, ROOT / 'models/evaluation_model.joblib')
    # Refit final forecast model on all completed games only after frozen testing.
    production = None if chosen == 'rolling_mean5' else estimator(chosen, cols).fit(data[cols], data.target)
    joblib.dump({'model': production, 'columns': cols, 'name': chosen}, ROOT / 'models/forecast_model.joblib')
    future = future.copy()
    future['prediction'] = predict(production, chosen, future, cols)
    future['low'] = np.maximum(0, future.prediction - future.position.map(bands))
    future['high'] = future.prediction + future.position.map(bands)
    future['risk'] = np.where(future.weeks_since_seen > 1, 'Check availability', 'No live injury check')
    future.to_csv(ROOT / 'reports/forecasts_week8.csv', index=False)
    test.to_csv(ROOT / 'reports/predictions_test.csv', index=False)
    dump(ROOT / 'reports/metrics.json', report)
    export_dashboard(data, future, test, report)
    print('TEST:', json.dumps(report['test']), 'BASELINE:', json.dumps(report['baseline']), flush=True)
    return report

def export_dashboard(data, future, test, report):
    players = []
    for r in future.sort_values('prediction', ascending=False).to_dict('records'):
        history = data[(data.player_id == r['player_id']) & (data.season == 2026)].sort_values('week')
        recent = history[['week', 'target']].tail(7).rename(columns={'target': 'points'}).to_dict('records')
        # First observed appearance isn't a modeling row, but the dashboard keeps
        # all modeled historical outcomes available transparently.
        players.append({'id': r['player_id'], 'name': r['name'], 'position': r['position'], 'team': r['team'],
            'opponent': r['opponent'], 'home': bool(r['home']), 'date': r['date'], 'week': int(r['week']),
            'prediction': round(r['prediction'], 2), 'low': round(r['low'], 2), 'high': round(r['high'], 2),
            'average': round(r['baseline'], 2), 'last': round(r['fantasy_points_ppr_lag1'], 2),
            'targets': round(r['targets_mean3'], 1), 'carries': round(r['carries_mean3'], 1),
            'passing_yards': round(r['passing_yards_mean3'], 1), 'receiving_yards': round(r['receiving_yards_mean3'], 1),
            'rushing_yards': round(r['rushing_yards_mean3'], 1),
            'opponent_allowed': round(r['opponent_allowed_mean5'], 2) if pd.notna(r['opponent_allowed_mean5']) else None,
            'gap': int(r['weeks_since_seen']), 'history_games': int(r['history_games']), 'history': recent})
    audit = test.sort_values(['season', 'week', 'prediction'], ascending=[False, False, False]).head(180)
    sample = audit[['name', 'position', 'season', 'week', 'prediction', 'target']].round(2).to_dict('records')
    payload = {'metrics': report, 'players': players, 'test_examples': sample,
        'author': 'Mochamad Rizky Maulana Aviansyah',
        'sources': [{'name': 'nflverse player stats', 'url': 'https://github.com/nflverse/nflverse-data/releases/tag/stats_player'},
                    {'name': 'nflverse schedules', 'url': SOURCES['schedule']}],
        'snapshot_note': 'Snapshot hasil pengambilan data, bukan pembaruan langsung. Periksa status cedera dan roster sebelum memilih pemain.'}
    dump(ROOT / 'dist/data.json', payload)
    dump(ROOT / 'dist/metrics.json', report)
    future[['player_id', 'name', 'position', 'team', 'opponent', 'season', 'week', 'prediction', 'low', 'high', 'baseline', 'risk']].round(3).to_csv(ROOT / 'dist/forecasts.csv', index=False)
    test[['name', 'position', 'season', 'week', 'target', 'prediction', 'baseline']].round(3).to_csv(ROOT / 'dist/test_predictions.csv', index=False)

def run(refresh=False):
    download(force=refresh)
    stats, games = load_data()
    data, future, complete = make_features(stats, games)
    data.to_csv(ROOT / 'reports/feature_dataset.csv', index=False)
    print('Samples:', data.shape, 'Forecasts:', future.shape, flush=True)
    return train_and_evaluate(data, future, complete)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='Replace cached source datasets with latest source bytes')
    args = parser.parse_args()
    run(args.refresh)
