"""V7: red-zone usage and expected touchdowns from nflverse play-by-play, strictly before the target week.

Touchdown rates per field-position bucket come from 2019-2022 only, before every validation and test season.
"""
from datetime import datetime,timezone
import hashlib
import numpy as np
import pandas as pd
import requests
from pipeline import ROOT,dump

DATA=ROOT/'data/pbp'
RATE_SEASONS=(2019,2022)
BUCKETS=[0,2,5,10,20,40,100]
COLUMNS=['season','week','season_type','game_id','play_type','yardline_100','rusher_player_id','receiver_player_id',
         'passer_player_id','pass_attempt','rush_attempt','pass_touchdown','rush_touchdown','air_yards','two_point_attempt','sack']
VALUES=['rz_targets','ez_targets','rz_carries','gl_carries','x_td','qb_rz_attempts','qb_x_pass_td']

def download(refresh_current=False):
    DATA.mkdir(parents=True,exist_ok=True)
    session=requests.Session();session.trust_env=False;manifest=[]
    for year in range(2019,2027):
        path=DATA/f'play_by_play_{year}.parquet'
        url=f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet'
        if not path.exists() or (refresh_current and year==2026):
            response=session.get(url,timeout=300);response.raise_for_status();path.write_bytes(response.content)
        manifest.append({'file':path.name,'url':url,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    dump(ROOT/'reports/pbp_sources.json',{'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'sources':manifest})

def load_plays():
    frames=[pd.read_parquet(f,columns=COLUMNS) for f in sorted(DATA.glob('play_by_play_*.parquet'))]
    plays=pd.concat(frames,ignore_index=True)
    plays=plays[plays.season_type.eq('REG')&plays.play_type.isin(['pass','run'])&plays.two_point_attempt.ne(1)&plays.sack.ne(1)].copy()
    plays['bucket']=pd.cut(plays.yardline_100,BUCKETS,labels=False,include_lowest=True)
    return plays

def player_games(plays):
    early=plays.season.between(*RATE_SEASONS)
    rush=plays[plays.rush_attempt.eq(1)&plays.rusher_player_id.notna()]
    target=plays[plays.pass_attempt.eq(1)&plays.receiver_player_id.notna()]
    passes=plays[plays.pass_attempt.eq(1)&plays.passer_player_id.notna()]
    rush_rate=rush[early.loc[rush.index]].groupby('bucket').rush_touchdown.mean()
    target_rate=target[early.loc[target.index]].groupby('bucket').pass_touchdown.mean()
    pass_rate=passes[early.loc[passes.index]].groupby('bucket').pass_touchdown.mean()
    keys=['player_id','season','week']
    r=rush.assign(player_id=rush.rusher_player_id,x=rush.bucket.map(rush_rate),rz=rush.yardline_100.le(20),gl=rush.yardline_100.le(5))
    r=r.groupby(keys).agg(rz_carries=('rz','sum'),gl_carries=('gl','sum'),x_rush=('x','sum'))
    t=target.assign(player_id=target.receiver_player_id,x=target.bucket.map(target_rate),rz=target.yardline_100.le(20),
                    ez=target.air_yards.ge(target.yardline_100))
    t=t.groupby(keys).agg(rz_targets=('rz','sum'),ez_targets=('ez','sum'),x_rec=('x','sum'))
    q=passes.assign(player_id=passes.passer_player_id,x=passes.bucket.map(pass_rate),rz=passes.yardline_100.le(20))
    q=q.groupby(keys).agg(qb_rz_attempts=('rz','sum'),qb_x_pass_td=('x','sum'))
    games=pd.concat([r,t,q],axis=1).fillna(0.).reset_index()
    games['x_td']=games.x_rush+games.x_rec
    rates={'rush':rush_rate.to_dict(),'target':target_rate.to_dict(),'pass':pass_rate.to_dict(),'seasons':list(RATE_SEASONS),'buckets':BUCKETS}
    return games[keys+VALUES].astype({c:float for c in VALUES}),rates

def _prior(frame,games,group_keys,windows,prefix):
    out=pd.DataFrame(index=frame.index,columns=[f'{prefix}_{c}_mean{n}' for c in VALUES for n in windows],dtype=float)
    history={k:g for k,g in games.groupby(group_keys,sort=False)}
    for key,ids in frame.groupby(group_keys,sort=False).groups.items():
        h=history.get(key if isinstance(key,tuple) else (key,))
        if h is None:continue
        order=(h.season*100+h.week).to_numpy()
        target=(frame.loc[ids,'season']*100+frame.loc[ids,'week']).to_numpy()
        # side=left excludes the target week itself.
        idx=np.searchsorted(order,target,side='left')-1;ok=idx>=0
        for col in VALUES:
            for n in windows:
                vals=h[col].rolling(n,min_periods=1).mean().to_numpy()
                out.loc[np.asarray(ids)[ok],f'{prefix}_{col}_mean{n}']=vals[idx[ok]]
    return out

def enrich(frame,games):
    frame=frame.drop(columns=[c for c in frame if c.startswith('v7_')]).reset_index(drop=True).copy()
    games=games.sort_values(['player_id','season','week']).reset_index(drop=True)
    season=_prior(frame,games,['player_id','season'],(3,5),'v7')
    career=_prior(frame,games,['player_id'],(8,),'v7_career')
    frame=pd.concat([frame,season,career],axis=1)
    for n in (3,5):
        frame[f'v7_exp_ppr{n}']=frame[f'v2_no_td_ppr_mean{n}']+6*frame[f'v7_x_td_mean{n}'].fillna(0)+4*frame[f'v7_qb_x_pass_td_mean{n}'].fillna(0)
    frame['v7_exp_ppr_career8']=frame.v2_career_no_td_ppr_mean8+6*frame.v7_career_x_td_mean8.fillna(0)+4*frame.v7_career_qb_x_pass_td_mean8.fillna(0)
    frame['v7_td_luck3']=frame.fantasy_points_ppr_mean3-frame.v7_exp_ppr3
    frame['v7_exp_ppr3_x_implied']=frame.v7_exp_ppr3*frame.v6_implied_team_total/22.5
    cols=[c for c in frame if c.startswith('v7_')]
    return frame,cols

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-current',action='store_true')
    download(parser.parse_args().refresh_current)
