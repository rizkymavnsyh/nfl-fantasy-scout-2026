"""Update next-game forecasts using completed-game data and timezone-aware dates.

Benchmark CSVs/raw snapshots remain frozen. Live source files are kept separately.
No forecasts of finished games, no recursive fabrication of future player stats.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import joblib
import numpy as np
import pandas as pd
import requests
from pipeline import ROOT,POSITIONS,STATS,TEAM_ALIASES,load_data,make_features,complete_weeks,dump,export_dashboard
from feature_engineering import enhance
from forecast_models import FantasyEnsemble

def kickoff_utc(games):
    # Source gametime is US Eastern local time. zoneinfo accounts for DST.
    text=games.gameday.astype(str)+' '+games.gametime.fillna('').astype(str)
    local=pd.to_datetime(text,format='%Y-%m-%d %H:%M',errors='coerce')
    return local.dt.tz_localize('America/New_York',ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC')

def upcoming_games(games,as_of,season=2026):
    frame=games[(games.season==season)&(games.game_type=='REG')].copy()
    frame['kickoff_utc']=kickoff_utc(frame)
    # Both score and time are checked. Missing old scores alone do not make an
    # already-started game eligible for an upcoming prediction.
    frame=frame[(frame.kickoff_utc>as_of)&frame.home_score.isna()&frame.away_score.isna()]
    return frame.sort_values(['kickoff_utc','game_id']).reset_index(drop=True)

def source_snapshot(refresh):
    live=ROOT/'data/live';live.mkdir(parents=True,exist_ok=True)
    session=requests.Session();session.trust_env=False
    inputs=[('stats_player_week_2026.csv','https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_2026.csv'),
            ('games.csv','https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv')]
    manifest=[]
    for name,url in inputs:
        path=live/name
        if refresh or not path.exists():
            response=session.get(url,timeout=90);response.raise_for_status();path.write_bytes(response.content)
            modified=response.headers.get('Last-Modified')
        else:modified=None
        manifest.append({'file':name,'url':url,'last_modified':modified,'bytes':path.stat().st_size,
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    stats_old,games_old=load_data()
    current=pd.read_csv(live/'stats_player_week_2026.csv',low_memory=False)
    current=current[(current.season_type=='REG')&current.position.isin(POSITIONS)].copy()
    for col in STATS:current[col]=pd.to_numeric(current[col],errors='coerce').fillna(0)
    current['team']=current.team.replace(TEAM_ALIASES);current['opponent_team']=current.opponent_team.replace(TEAM_ALIASES)
    stats=pd.concat([stats_old[stats_old.season<2026],current],ignore_index=True)
    games=pd.read_csv(live/'games.csv',low_memory=False)
    games=games[(games.game_type=='REG')&games.season.between(2019,2026)].copy()
    games['home_team']=games.home_team.replace(TEAM_ALIASES);games['away_team']=games.away_team.replace(TEAM_ALIASES)
    return stats,games,manifest

def build_next_rows(stats,games,history,as_of):
    upcoming=upcoming_games(games,as_of)
    if upcoming.empty:raise ValueError('Tidak ada jadwal musim reguler 2026 yang belum dimulai. Perbarui jadwal atau pilih musim yang didukung.')
    targets={}
    for game in upcoming.to_dict('records'):
        targets.setdefault(game['home_team'],game);targets.setdefault(game['away_team'],game)
    current=stats[stats.season==2026].sort_values(['week','player_id'])
    offense=current[(current.attempts+current.carries+current.targets)>0]
    first=offense.groupby('player_id').week.min()
    actual=current[current.apply(lambda r:r.week>=first.get(r.player_id,999),axis=1)].copy()
    zero=history[(history.season==2026)&(~history.observed_stats)].copy()
    for col in STATS:zero[col]=0.
    timeline=pd.concat([actual[['player_id','season','week']+STATS],zero[['player_id','season','week']+STATS]],ignore_index=True)
    timeline=timeline.drop_duplicates(['player_id','season','week']).sort_values(['player_id','week'])
    histories={pid:g for pid,g in timeline.groupby('player_id')}
    latest=offense.drop_duplicates('player_id',keep='last')
    defense=defaultdict(list)
    # Full team-position totals, including zeros where a completed game had no
    # recorded scoring for that position. Everything precedes as_of.
    for (season,week),gw in games[games.home_score.notna()&games.away_score.notna()].groupby(['season','week'],sort=True):
        rows=stats[(stats.season==season)&(stats.week==week)]
        supplied=set(rows.game_id)
        for game in gw[gw.game_id.isin(supplied)].to_dict('records'):
            for team in [game['home_team'],game['away_team']]:
                for pos in POSITIONS:
                    points=rows[(rows.game_id==game['game_id'])&(rows.opponent_team==team)&(rows.position==pos)].fantasy_points_ppr.sum()
                    defense[(team,pos)].append(float(points))
    forecast=[]
    for r in latest.to_dict('records'):
        pid=r['player_id'];team=r['team'];game=targets.get(team)
        if game is None or int(game['week'])-int(r['week'])>3:continue
        hist=histories[pid]
        home=team==game['home_team'];opp=game['away_team'] if home else game['home_team']
        x={'player_id':pid,'name':r['player_display_name'],'position':r['position'],'team':team,'opponent':opp,
           'season':2026,'week':int(game['week']),'game_id':game['game_id'],'date':game['gameday'],
           'home':int(home),'rest':float(game['home_rest'] if home else game['away_rest']),
           'weeks_since_seen':int(game['week'])-int(r['week']),'history_games':len(hist),
           'kickoff_utc':game['kickoff_utc'].isoformat(),
           'kickoff_wib':game['kickoff_utc'].tz_convert('Asia/Jakarta').strftime('%d %b %Y %H:%M WIB')}
        for col in STATS:
            vals=hist[col].to_numpy(dtype=float)
            x[col+'_lag1']=float(vals[-1]);x[col+'_mean3']=float(np.mean(vals[-3:]))
            x[col+'_mean5']=float(np.mean(vals[-5:]));x[col+'_season_mean']=float(np.mean(vals))
        x['ppr_std5']=float(np.std(hist.fantasy_points_ppr.to_numpy()[-5:]))
        allowed=defense[(opp,r['position'])]
        x['opponent_allowed_mean5']=float(np.mean(allowed[-5:])) if allowed else np.nan
        x['baseline']=x['fantasy_points_ppr_mean5']
        forecast.append(x)
    frame=pd.DataFrame(forecast)
    if frame.empty:raise ValueError('Riwayat pemain terlalu lama untuk jadwal berikutnya. Perbarui statistik terlebih dahulu.')
    assert not frame.player_id.duplicated().any()
    return frame,upcoming

def refresh(as_of=None,download=True):
    cutoff=pd.Timestamp(as_of or datetime.now(timezone.utc))
    if cutoff.tzinfo is None:raise ValueError('as-of harus mencantumkan zona waktu, misalnya 2026-10-27T12:00:00+07:00.')
    cutoff=cutoff.tz_convert('UTC')
    stats,games,manifest=source_snapshot(download)
    games['kickoff_cutoff']=kickoff_utc(games)
    completed=games[(games.kickoff_cutoff<cutoff)&games.home_score.notna()&games.away_score.notna()]
    stats=stats[stats.game_id.isin(set(completed.game_id))].copy()
    # Future score data must not enter histories even if a cached file is newer.
    bounded=games.copy()
    bounded.loc[~bounded.game_id.isin(set(completed.game_id)),['home_score','away_score']]=np.nan
    if upcoming_games(bounded,cutoff).empty:
        raise ValueError('Semua laga musim reguler 2026 sudah dimulai atau selesai. Playoff dan musim 2027 belum didukung.')
    history,unused,last_full=make_features(stats,bounded.drop(columns=['kickoff_cutoff']),allow_empty_forecast=True)
    history[['player_id','season','week','team','game_id']].to_csv(ROOT/'data/live/candidate_history.csv',index=False)
    next_rows,upcoming=build_next_rows(stats,bounded.drop(columns=['kickoff_cutoff']),history,cutoff)
    enhanced,next_rows,_=enhance(history,next_rows,stats)
    bundle=joblib.load(ROOT/'models/evaluation_model.joblib')
    evaluation=bundle['model'];cols=bundle['columns']
    if any(c.startswith('v4_') for c in cols):
        from context_features import enrich
        enhanced,_=enrich(enhanced,stats)
        next_rows,_=enrich(next_rows,stats)
    if any(c.startswith('v5_') for c in cols):
        from pregame_features import enrich as enrich_pregame,load_sources as load_pregame
        snap_history,injury_history=load_pregame()
        print('Pregame: enriching historical role features',flush=True)
        enhanced,_=enrich_pregame(enhanced,snap_history,injury_history)
        next_rows,_=enrich_pregame(next_rows,snap_history,injury_history,candidate_history=enhanced)
        print('Pregame: role features ready',flush=True)
    if any(c.startswith('v6_') for c in cols):
        from market_features import enrich as enrich_market
        market_games=bounded.drop(columns=['kickoff_cutoff'])
        enhanced,_=enrich_market(enhanced,market_games)
        next_rows,_=enrich_market(next_rows,market_games)
    print('Fitting forecast model',flush=True)
    production=FantasyEnsemble(evaluation.specifications,evaluation.weights).fit(enhanced[cols],enhanced.target)
    joblib.dump({'model':production,'columns':cols,'name':bundle['name'],'as_of':cutoff.isoformat()},ROOT/'models/forecast_model.joblib')
    report=json.loads((ROOT/'reports/metrics.json').read_text(encoding='utf-8'))
    bands=report['interval']['half_width_by_position']
    next_rows['prediction']=production.predict(next_rows[cols])
    next_rows['low']=np.maximum(0,next_rows.prediction-next_rows.position.map(bands));next_rows['high']=next_rows.prediction+next_rows.position.map(bands)
    next_rows['risk']=np.where(next_rows.weeks_since_seen>1,'Check availability / bye or gap','No live injury check')
    next_rows.to_csv(ROOT/'reports/forecasts_next_game.csv',index=False)
    # Compatibility alias for the original notebook; contains multiple weeks
    # where bye teams' next game is in a later week. Prefer forecasts_next_game.
    next_rows.to_csv(ROOT/'reports/forecasts_week8.csv',index=False)
    available_current=stats[stats.season==2026]
    missing=completed[(completed.season==2026)&(~completed.game_id.isin(set(available_current.game_id)))]
    missing_ids=missing.game_id.tolist()
    snapshot={'as_of_utc':cutoff.isoformat(),'as_of_wib':cutoff.tz_convert('Asia/Jakarta').strftime('%d %b %Y %H:%M WIB'),
      'season':2026,'season_label':'Musim reguler 2026 · berlanjut ke Januari 2027',
      'latest_complete_week':int(last_full),'latest_observed_week':int(available_current.week.max()),
      'next_week':int(upcoming.week.min()),'next_kickoff_utc':upcoming.kickoff_utc.iloc[0].isoformat(),
      'next_kickoff_wib':upcoming.kickoff_utc.iloc[0].tz_convert('Asia/Jakarta').strftime('%d %b %Y %H:%M WIB'),
      'missing_completed_games':missing_ids,'sources':manifest,'candidate_count':len(next_rows),
      'next_game_per_team':True,'history_training_n':len(enhanced),
      'coverage':'Next scheduled regular-season game for each currently eligible player; no season-total or playoff forecasts.',
      'update_mode':'Manual refresh command, not an automatic live feed.'}
    report['forecast_snapshot']=snapshot;report['prediction_week']=snapshot['next_week']
    dump(ROOT/'reports/forecast_snapshot.json',snapshot);dump(ROOT/'reports/metrics.json',report)
    test=pd.read_csv(ROOT/'reports/predictions_test.csv')
    export_dashboard(history,next_rows,test,report)
    payload=json.loads((ROOT/'dist/data.json').read_text(encoding='utf-8'))
    lookup=next_rows.set_index('player_id')
    for p in payload['players']:
        row=lookup.loc[p['id']]
        p['game_id']=row.game_id;p['kickoff_utc']=row.kickoff_utc;p['kickoff_wib']=row.kickoff_wib
        for key,col in [('implied_total','v6_implied_team_total'),('spread','v6_team_spread'),('total_line','v6_total_line')]:
            p[key]=round(float(row[col]),1) if col in row and pd.notna(row[col]) else None
    used=set(next_rows.game_id)
    fixtures=[]
    for game in upcoming[upcoming.game_id.isin(used)].to_dict('records'):
        fixtures.append({'id':game['game_id'],'week':int(game['week']),'away':game['away_team'],'home':game['home_team'],
          'kickoff_utc':game['kickoff_utc'].isoformat(),
          'kickoff_wib':game['kickoff_utc'].tz_convert('Asia/Jakarta').strftime('%d %b %Y %H:%M WIB'),
          'players':int((next_rows.game_id==game['game_id']).sum())})
    payload['fixtures']=fixtures;payload['forecast_snapshot']=snapshot
    dump(ROOT/'dist/data.json',payload)
    next_rows[['player_id','name','position','team','opponent','season','week','game_id','kickoff_utc','kickoff_wib',
               'prediction','low','high','baseline','risk']].round(3).to_csv(ROOT/'dist/forecasts.csv',index=False)
    print('FORECAST SNAPSHOT',json.dumps(snapshot),flush=True)
    return snapshot

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--as-of');parser.add_argument('--cached',action='store_true')
    args=parser.parse_args();refresh(args.as_of,not args.cached)
