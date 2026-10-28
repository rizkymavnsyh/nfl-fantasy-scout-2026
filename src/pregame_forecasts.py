"""Attach historical role evidence and the explicit fantasy-pool flag to forecasts."""
import hashlib,json
import pandas as pd
import numpy as np
import joblib
from pipeline import ROOT,dump
from pregame_features import enrich,load_sources,fantasy_pool

def update():
    snaps,injuries=load_sources()
    forecasts=pd.read_csv(ROOT/'reports/forecasts_next_game.csv',float_precision='round_trip')
    path=ROOT/'data/live/candidate_history.csv'
    history=pd.read_csv(path if path.exists() else ROOT/'reports/feature_dataset.csv')
    forecasts,cols=enrich(forecasts,snaps,injuries,candidate_history=history)
    forecasts['fantasy_pool']=fantasy_pool(forecasts)
    forecasts['prior_snap_pct']=forecasts.v5_snap_pct_lag1
    forecasts['prior_snap_weeks_ago']=forecasts.v5_snap_weeks_ago
    forecasts['prior_injury_note']=forecasts.apply(lambda r:
       'Not on last week\'s injury report' if r.v5_injury_previous_week_missing else
       'Out/Doubtful last week' if r.v5_injury_prev_out else
       'Questionable last week' if r.v5_injury_prev_questionable else
       'Listed on last week\'s injury report',axis=1)
    report=json.loads((ROOT/'reports/metrics.json').read_text(encoding='utf-8'))
    if report['version']=='v5':
        model=joblib.load(ROOT/'models/forecast_model.joblib')
        forecasts['prediction']=model['model'].predict(forecasts[model['columns']])
        band=forecasts.position.map(report['interval']['half_width_by_position'])
        forecasts['low']=np.maximum(0,forecasts.prediction-band);forecasts['high']=forecasts.prediction+band
    for name in ['forecasts_next_game.csv','forecasts_week8.csv']:forecasts.to_csv(ROOT/'reports'/name,index=False)
    payload=json.loads((ROOT/'dist/data.json').read_text(encoding='utf-8'))
    lookup=forecasts.set_index('player_id')
    for p in payload['players']:
        row=lookup.loc[p['id']]
        for key in ['fantasy_pool','prior_snap_pct','prior_snap_weeks_ago','prior_injury_note']:
            value=row[key];p[key]=None if pd.isna(value) else value.item() if hasattr(value,'item') else value
        for key in ['prediction','low','high']:p[key]=round(float(row[key]),2)
    summary={'n':len(forecasts),'fantasy_pool_n':int(forecasts.fantasy_pool.sum()),
      'role_history_n':int(forecasts.prior_snap_pct.notna().sum()),
      'snap_source_max_week_2026':int(snaps[snaps.season.eq(2026)].week.max()),
      'injury_source_max_week_2026':int(injuries[injuries.season.eq(2026)].week.max()),
      'source_manifest_sha256':hashlib.sha256((ROOT/'reports/pregame_sources.json').read_bytes()).hexdigest(),
      'rule':'Uses previous-week/game role only, never future participation. A fantasy-pool flag is not a game-day active confirmation.'}
    payload['pregame_context']=summary
    if (ROOT/'reports/v5_comparison.json').exists():payload['pregame_comparison']=json.loads((ROOT/'reports/v5_comparison.json').read_text(encoding='utf-8'))
    dump(ROOT/'reports/pregame_forecast_context.json',summary);dump(ROOT/'dist/data.json',payload)
    print(json.dumps(summary))
    return summary

if __name__=='__main__':update()
