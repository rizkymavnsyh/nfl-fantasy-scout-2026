"""Historical role and prior-week injury evidence, never target-game snaps/status."""
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib,io,json
import numpy as np
import pandas as pd
import requests
from pipeline import ROOT,TEAM_ALIASES,dump

DATA=ROOT/'data/pre_game'

def download_sources(refresh_current=False):
    DATA.mkdir(parents=True,exist_ok=True)
    previous_path=ROOT/'reports/pregame_sources.json'
    previous=json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else []
    old={r['file']:r for r in previous}
    tasks=[(tag,f'{prefix}_{y}.csv',y) for y in range(2019,2027)
           for tag,prefix in [('rosters','roster'),('snap_counts','snap_counts'),('injuries','injuries')]]
    def get(task):
        tag,name,year=task;path=DATA/name
        url=f'https://github.com/nflverse/nflverse-data/releases/download/{tag}/{name}'
        if path.exists() and not (refresh_current and year==2026):
            record=old.get(name,{'file':name,'url':url,'retrieved_at_utc':None})
            record['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();return record
        session=requests.Session();session.trust_env=False
        try:
            response=session.get(url,timeout=75);response.raise_for_status()
            pd.read_csv(io.BytesIO(response.content),nrows=1)
            path.write_bytes(response.content)
            return {'file':name,'url':url,'bytes':len(response.content),'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
                    'last_modified':response.headers.get('Last-Modified'),'sha256':hashlib.sha256(response.content).hexdigest(),'error':None}
        except (requests.RequestException,ValueError) as exc:
            return {'file':name,'url':url,'error':str(exc),'retrieved_at_utc':datetime.now(timezone.utc).isoformat()}
    with ThreadPoolExecutor(max_workers=4) as pool:manifest=list(pool.map(get,tasks))
    dump(previous_path,manifest)
    print('Pregame sources:',sum(not r.get('error') for r in manifest),'of',len(manifest),'available',flush=True)
    return manifest

def load_sources():
    def read(prefix):
        files=sorted(DATA.glob(prefix+'_*.csv'))
        return pd.concat([pd.read_csv(f,low_memory=False) for f in files],ignore_index=True) if files else pd.DataFrame()
    rosters=read('roster');snaps=read('snap_counts');injuries=read('injuries')
    bridge=rosters[['season','pfr_id','gsis_id']].dropna().drop_duplicates()
    conflicts=bridge.groupby(['season','pfr_id']).gsis_id.nunique()
    safe=conflicts[conflicts.eq(1)].reset_index()[['season','pfr_id']]
    bridge=bridge.merge(safe,on=['season','pfr_id']).drop_duplicates(['season','pfr_id'])
    snaps=snaps[snaps.game_type.eq('REG')].merge(bridge,left_on=['season','pfr_player_id'],right_on=['season','pfr_id'],how='left',validate='many_to_one')
    coverage={'snap_rows':len(snaps),'snap_rows_mapped':int(snaps.gsis_id.notna().sum()),'ambiguous_player_ids_removed':int(conflicts.gt(1).sum())}
    snaps=snaps.dropna(subset=['gsis_id']).rename(columns={'gsis_id':'player_id'})
    snaps['team']=snaps.team.replace(TEAM_ALIASES)
    if not injuries.empty:
        injuries=injuries[injuries.game_type.eq('REG')].dropna(subset=['gsis_id']).rename(columns={'gsis_id':'player_id'})
        injuries['team']=injuries.team.replace(TEAM_ALIASES)
        injuries['modified']=pd.to_datetime(injuries.get('date_modified'),utc=True,errors='coerce')
    dump(ROOT/'reports/pregame_id_coverage.json',coverage)
    return snaps,injuries

def enrich(frame,snaps,injuries,candidate_history=None):
    frame=frame.drop(columns=[c for c in frame if c.startswith('v5_')]).reset_index(drop=True).copy()
    # Build a timeline for every model candidate; an absent previous-game snap
    # is explicit zero only when source rows for that game/team were supplied.
    keys=['player_id','season','week','team']
    source=snaps.groupby(keys,as_index=False)[['offense_snaps','offense_pct']].sum()
    supplied=set(zip(snaps.game_id,snaps.team))
    first_known=snaps.groupby(['player_id','season','team']).week.min().to_dict()
    existing=set(map(tuple,source[keys].to_numpy()))
    zeros=[]
    zero_rows=frame if candidate_history is None else pd.concat([frame,candidate_history],ignore_index=True)
    for row in zero_rows.itertuples():
        key=(row.player_id,row.season,row.week,row.team)
        known_before=first_known.get((row.player_id,row.season,row.team),999)<row.week
        if key not in existing and known_before and (row.game_id,row.team) in supplied:
            zeros.append(dict(zip(keys,key),offense_snaps=0.,offense_pct=0.))
    if zeros:source=pd.concat([source,pd.DataFrame(zeros)],ignore_index=True).drop_duplicates(keys)
    by_player={key:g.sort_values('week') for key,g in source.groupby(['player_id','season','team'],sort=False)}
    inj={key:g.sort_values('week') for key,g in injuries.groupby(['player_id','season','team'],sort=False)} if len(injuries) else {}
    result=[]
    for row in frame.itertuples():
        key=(row.player_id,row.season,row.team)
        history=by_player.get(key)
        record={
          'v5_snap_pct_lag1':np.nan,'v5_snap_pct_mean3':np.nan,'v5_snap_count_mean3':np.nan,
          'v5_snap_pct_trend':np.nan,'v5_snap_weeks_ago':np.nan,'v5_snap_history_available':0.,
          'v5_injury_prev_report':0.,'v5_injury_prev_out':0.,'v5_injury_prev_questionable':0.,
          'v5_injury_prev_dnp':0.,'v5_injury_prev_limited':0.,'v5_injury_previous_week_missing':1.}
        if history is not None:
            # Strict prior-week cutoff is deliberately stronger than kickoff-only matching.
            h=history[history.week<row.week].tail(3)
            if len(h):
                values=h.offense_pct.to_numpy(dtype=float)
                record.update(v5_snap_pct_lag1=float(values[-1]),v5_snap_pct_mean3=float(values.mean()),
                  v5_snap_count_mean3=float(h.offense_snaps.mean()),v5_snap_pct_trend=float(values[-1]-values.mean()),
                  v5_snap_weeks_ago=float(row.week-h.week.iloc[-1]),v5_snap_history_available=1.)
        prior=inj.get(key)
        if prior is not None:
            h=prior[prior.week.eq(row.week-1)].copy()
            if len(h):
                # date_modified is optional in 2025/26. For older dated reports,
                # exclude records whose update occurred after the target date.
                cutoff=pd.Timestamp(row.date,tz='UTC')
                h=h[h.modified.isna()|(h.modified<cutoff)]
            if len(h):
                status=h.report_status.fillna('').astype(str).str.lower()
                practice=h.practice_status.fillna('').astype(str).str.lower()
                record.update(v5_injury_prev_report=1.,v5_injury_prev_out=float(status.isin(['out','doubtful']).any()),
                  v5_injury_prev_questionable=float(status.eq('questionable').any()),
                  v5_injury_prev_dnp=float(practice.str.contains('did not').any()),
                  v5_injury_prev_limited=float(practice.str.contains('limited').any()),v5_injury_previous_week_missing=0.)
        result.append(record)
    extra=pd.DataFrame(result,index=frame.index)
    frame=pd.concat([frame,extra],axis=1)
    # Rates of chances relative to actual prior playing time, without future outcomes.
    frame['v5_chances_per_snap3']=(frame.targets_mean3+frame.carries_mean3)/frame.v5_snap_count_mean3.replace(0,np.nan)
    frame['v5_qb_attempts_per_snap3']=frame.attempts_mean3/frame.v5_snap_count_mean3.replace(0,np.nan)
    frame['v5_points_per_snap3']=frame.fantasy_points_ppr_mean3/frame.v5_snap_count_mean3.replace(0,np.nan)
    cols=[c for c in frame if c.startswith('v5_')]
    return frame,cols

def fantasy_pool(frame):
    """A priori relevance rule, using history only. Never actual target participation."""
    qb=frame.position.eq('QB') & ((frame.attempts_mean3>=15)|(frame.v5_snap_pct_lag1>=.5))
    skill=frame.position.isin(['RB','WR','TE']) & (((frame.carries_mean3+frame.targets_mean3)>=5)|(frame.v5_snap_pct_mean3>=.25))
    # A stale record alone must not grant relevance after several absent games.
    recent=frame.v5_snap_weeks_ago.le(2)|frame.v5_snap_weeks_ago.isna()
    return (qb|skill)&recent

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-current',action='store_true')
    download_sources(parser.parse_args().refresh_current);load_sources()
