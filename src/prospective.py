"""Append-only forecast snapshots, post-game scoring, and simulated lineup replay."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import ROOT, dump

PROTOCOL = {'seed':20261027,'rosters_per_week':30,'slots':{'QB':1,'RB':2,'WR':2,'TE':1,'FLEX':1},
            'roster_sizes':{'QB':2,'RB':4,'WR':5,'TE':2},
            'candidate_caps':{'QB':24,'RB':60,'WR':72,'TE':30},
            'pool_rule':'Top pre-game rolling-five average per position, then seeded sampling without replacement.',
            'claim':'Simulated rosters, not real league ownership. Weekly clusters are not independent roster trials.',
            'missing_target_rule':'Zero only after final scores and player-stat rows for both teams are present.'}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def lineup(frame, score):
    """Exact seven-slot selection. Stable ID tie-break is shared across policies."""
    by = {p:g.sort_values([score,'player_id'],ascending=[False,True]) for p,g in frame.groupby('position')}
    if any(len(by.get(p,[]))<n for p,n in [('QB',1),('RB',2),('WR',2),('TE',1)]): return None
    best = None
    for flex in frame[frame.position.isin(['RB','WR','TE'])].sort_values('player_id').itertuples():
        chosen = [by['QB'].iloc[0].player_id]
        for pos,n in [('RB',2),('WR',2),('TE',1)]:
            chosen += by[pos][by[pos].player_id.ne(flex.player_id)].head(n).player_id.tolist()
        if len(chosen)!=6: continue
        chosen.append(flex.player_id)
        total = float(frame.set_index('player_id').loc[chosen,score].sum())
        if best is None or total>best[0]: best=(total,chosen)
    return None if best is None else best[1]

def make_rosters(frame):
    output=[]
    for (season,week), group in frame.groupby(['season','week'],sort=True):
        pools = {pos:group[group.position.eq(pos)].sort_values(['baseline','player_id'],ascending=[False,True]).head(cap)
                 for pos,cap in PROTOCOL['candidate_caps'].items()}
        if any(len(pools[p])<n for p,n in PROTOCOL['roster_sizes'].items()): continue
        rng=np.random.default_rng(PROTOCOL['seed']+int(season)*100+int(week))
        for trial in range(PROTOCOL['rosters_per_week']):
            ids=[]
            for pos,n in PROTOCOL['roster_sizes'].items():
                ids.extend(rng.choice(pools[pos].player_id.to_numpy(),n,replace=False).tolist())
            roster=group[group.player_id.isin(ids)]
            output.append({'season':int(season),'week':int(week),'trial':trial,'roster':ids,
                           'model':lineup(roster,'prediction'),'baseline':lineup(roster,'baseline')})
    return output

def score_rosters(rosters, targets):
    records=[]
    for r in rosters:
        key=(r['season'],r['week'])
        needed=set(r['model']+r['baseline'])
        values={pid:targets.get((*key,pid)) for pid in needed}
        if any(v is None or not np.isfinite(v) for v in values.values()): continue
        model=sum(values[p] for p in r['model']);base=sum(values[p] for p in r['baseline'])
        records.append({'season':key[0],'week':key[1],'trial':r['trial'],'model_points':model,
                        'baseline_points':base,'difference':model-base})
    frame=pd.DataFrame(records)
    if frame.empty:return frame,{'n_rosters':0,'n_weeks':0,'status':'Menunggu hasil pertandingan'}
    weekly=frame.groupby(['season','week'])[['model_points','baseline_points','difference']].mean().reset_index()
    return frame,{'n_rosters':len(frame),'n_weeks':len(weekly),
      'model_mean_points':float(frame.model_points.mean()),'baseline_mean_points':float(frame.baseline_points.mean()),
      'mean_difference':float(frame.difference.mean()),'win_rate':float((frame.difference>1e-8).mean()),
      'tie_rate':float((abs(frame.difference)<=1e-8).mean()),'weekly':weekly.to_dict('records'),
      'status':'Simulasi retrospektif' if frame.season.min()<2026 else 'Evaluasi snapshot terkunci'}

def replay():
    # Protocol is recorded before examining outcomes. No tuning follows this replay.
    dump(ROOT/'reports/lineup_protocol.json',PROTOCOL)
    frame=pd.read_csv(ROOT/'reports/predictions_test.csv')
    rosters=make_rosters(frame)
    targets={(int(r.season),int(r.week),r.player_id):float(r.target) for r in frame.itertuples()}
    scored,summary=score_rosters(rosters,targets)
    summary['status']='Simulasi retrospektif pada data uji yang sudah pernah diperiksa'
    summary['protocol']=PROTOCOL
    scored.to_csv(ROOT/'reports/lineup_replay.csv',index=False)
    dump(ROOT/'reports/lineup_replay.json',summary)
    # IDs and decisions allow reconstructing both recommendations from the same roster.
    dump(ROOT/'reports/lineup_replay_rosters.json',rosters)
    return summary

def freeze():
    frame=pd.read_csv(ROOT/'reports/forecasts_next_game.csv',float_precision='round_trip')
    current=json.loads((ROOT/'reports/metrics.json').read_text(encoding='utf-8'))
    if current['version']=='v5':
        import joblib
        model=joblib.load(ROOT/'models/forecast_model.joblib')
        assert np.allclose(model['model'].predict(frame[model['columns']]),frame.prediction,atol=1e-10,rtol=0), 'Forecast feature round-trip mismatch'
    now=pd.Timestamp(datetime.now(timezone.utc))
    # Do not permit artificially backdated freezes, even if forecast features are older.
    frame=frame[pd.to_datetime(frame.kickoff_utc,utc=True)>now].copy()
    if frame.empty:raise ValueError('Tidak ada prediksi sebelum kickoff untuk dikunci')
    root=ROOT/'reports/frozen';root.mkdir(exist_ok=True)
    stamp=now.strftime('%Y%m%dT%H%M%S%fZ')
    dest=root/stamp;dest.mkdir(exist_ok=False)
    frame.to_csv(dest/'forecasts.csv',index=False)
    allowed=frame[frame.lineup_eligible.eq(True)] if 'lineup_eligible' in frame else frame.iloc[0:0]
    rosters=make_rosters(allowed);dump(dest/'lineups.json',rosters)
    forecast_meta=json.loads((ROOT/'reports/forecast_snapshot.json').read_text(encoding='utf-8'))
    model_meta=json.loads((ROOT/'reports/metrics.json').read_text(encoding='utf-8'))
    manifest={'snapshot_id':stamp,'frozen_at_utc':now.isoformat(),'feature_cutoff_utc':forecast_meta['as_of_utc'],
       'n':len(frame),'protocol':PROTOCOL,'forecast_sha256':sha(dest/'forecasts.csv'),
       'lineups_sha256':sha(dest/'lineups.json'),'model_sha256':sha(ROOT/'models/forecast_model.joblib'),
       'source_forecast':forecast_meta,'availability':json.loads((ROOT/'reports/availability.json').read_text(encoding='utf-8')),
       'code_sha256':{name:sha(ROOT/'src'/name) for name in ['prospective.py','availability.py','forecast_models.py','refresh_forecasts.py']},
       'note':'Append-only snapshot. Hash detects local edits, not independent third-party timestamp certification.'}
    manifest['model_name']=model_meta['selected_model'];manifest['model_version']=model_meta['version']
    if (ROOT/'reports/pregame_sources.json').exists():
        manifest['pregame_sources']=json.loads((ROOT/'reports/pregame_sources.json').read_text(encoding='utf-8'))
        manifest['code_sha256']['pregame_features.py']=sha(ROOT/'src/pregame_features.py')
    dump(dest/'manifest.json',manifest)
    print('FROZEN',stamp,len(frame),'predictions',len(rosters),'simulated lineups',flush=True)
    return dest

def score_snapshot(dest, stats, games, now=None):
    manifest=json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
    if sha(dest/'forecasts.csv')!=manifest['forecast_sha256'] or sha(dest/'lineups.json')!=manifest['lineups_sha256']:
        raise ValueError('Hash snapshot berubah: '+dest.name)
    frame=pd.read_csv(dest/'forecasts.csv',float_precision='round_trip')
    frozen=pd.Timestamp(manifest['frozen_at_utc'])
    if not (pd.to_datetime(frame.kickoff_utc,utc=True)>frozen).all():raise ValueError('Snapshot memuat laga yang sudah dimulai saat dikunci')
    now=pd.Timestamp(now or datetime.now(timezone.utc))
    from refresh_forecasts import kickoff_utc
    games=games.copy();games['utc']=kickoff_utc(games)
    complete=games[games.home_score.notna()&games.away_score.notna()&(games.utc<now)]
    complete=complete.set_index('game_id')
    scores=[];targets={}
    # Only games with both teams represented in the supplied stat file can be scored.
    for r in frame.itertuples():
        actual=None;status='Menunggu hasil/statistik lengkap'
        if r.game_id in complete.index:
            game=complete.loc[r.game_id];observed=stats[stats.game_id.eq(r.game_id)]
            if {game.home_team,game.away_team}.issubset(set(observed.team)):
                player_rows=observed[observed.player_id.eq(r.player_id)]
                if len(player_rows)>1:raise ValueError('Duplikasi statistik pemain/laga')
                actual=float(player_rows.fantasy_points_ppr.iloc[0]) if len(player_rows) else 0.
                status='Statistik tercatat' if len(player_rows) else 'Tidak tercatat, nol setelah kelengkapan laga diperiksa'
                targets[(int(r.season),int(r.week),r.player_id)]=actual
        scores.append({'player_id':r.player_id,'name':r.name,'game_id':r.game_id,'prediction':r.prediction,
                       'baseline':r.baseline,'actual':actual,'status':status})
    scored=pd.DataFrame(scores);ready=scored[scored.actual.notna()]
    result={'snapshot_id':dest.name,'frozen_at_utc':manifest['frozen_at_utc'],'n_frozen':len(frame),
            'n_scored':len(ready),'n_pending':len(frame)-len(ready),'evaluated_at_utc':now.isoformat(),
            'status':'Menunggu laga masa depan' if ready.empty else 'Hasil pascalaga, statistik dapat mengalami koreksi',
            'missing_target_rule':PROTOCOL['missing_target_rule']}
    result['model_version']=manifest.get('model_version','v3');result['model_name']=manifest.get('model_name','V3 frozen model')
    if len(ready):
        result.update(mae=float(abs(ready.prediction-ready.actual).mean()),
                      rmse=float(np.sqrt(((ready.prediction-ready.actual)**2).mean())),
                      baseline_mae=float(abs(ready.baseline-ready.actual).mean()))
    rosters=json.loads((dest/'lineups.json').read_text(encoding='utf-8'))
    lineups,lineup_summary=score_rosters(rosters,targets);result['lineup']=lineup_summary
    return scored,lineups,result

def score_all():
    stats=pd.read_csv(ROOT/'data/live/stats_player_week_2026.csv',low_memory=False)
    games=pd.read_csv(ROOT/'data/live/games.csv',low_memory=False)
    results=[]
    out=ROOT/'reports/prospective_scores';out.mkdir(exist_ok=True)
    for dest in sorted((ROOT/'reports/frozen').glob('*')):
        if not dest.is_dir():continue
        scored,lineups,result=score_snapshot(dest,stats,games)
        # Evaluation outputs are separate: frozen inputs are never written again.
        scored.to_csv(out/(dest.name+'_players.csv'),index=False)
        lineups.to_csv(out/(dest.name+'_lineups.csv'),index=False)
        dump(out/(dest.name+'.json'),result);results.append(result)
    dump(ROOT/'reports/prospective_evaluation.json',results)
    return results

def export_panels():
    data=json.loads((ROOT/'dist/data.json').read_text(encoding='utf-8'))
    data['lineup_evaluation']=json.loads((ROOT/'reports/lineup_replay.json').read_text(encoding='utf-8'))
    data['prospective_evaluation']=json.loads((ROOT/'reports/prospective_evaluation.json').read_text(encoding='utf-8'))
    if (ROOT/'reports/optimization_summary.json').exists():
        data['optimization_summary']=json.loads((ROOT/'reports/optimization_summary.json').read_text(encoding='utf-8'))
    if (ROOT/'reports/v5_comparison.json').exists():
        data['pregame_comparison']=json.loads((ROOT/'reports/v5_comparison.json').read_text(encoding='utf-8'))
    dump(ROOT/'dist/data.json',data)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    if args.replay or not (ROOT/'reports/lineup_replay.json').exists(): print('REPLAY',json.dumps(replay()),flush=True)
    if args.freeze:freeze()
    print('PROSPECTIVE',json.dumps(score_all()),flush=True);export_panels()
