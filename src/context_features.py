"""Team opportunity and position context using strictly earlier source weeks."""
import numpy as np
import pandas as pd

def _prior_lookup(frame, source, keys, value_columns, prefix, windows=(3,5)):
    result=pd.DataFrame(index=frame.index)
    grouped={key:group.sort_values('week') for key,group in source.groupby(keys,sort=False)}
    names=[]
    for col in value_columns:
        for n in windows:
            name=f'v4_{prefix}_{col}_mean{n}';names.append(name);result[name]=np.nan
    for key, ids in frame.groupby(keys,sort=False).groups.items():
        if key not in grouped:continue
        history=grouped[key]
        # side=left explicitly excludes the target week, even when raw future outcomes exist.
        indices=np.searchsorted(history.week.to_numpy(),frame.loc[ids,'week'].to_numpy(),side='left')-1
        valid=indices>=0
        for col in value_columns:
            for n in windows:
                values=history[col].rolling(n,min_periods=1).mean().to_numpy()
                result.loc[np.asarray(ids)[valid],f'v4_{prefix}_{col}_mean{n}']=values[indices[valid]]
    return result,names

def enrich(frame, stats):
    frame=frame.drop(columns=[c for c in frame if c.startswith('v4_')]).copy()
    frame=frame.reset_index(drop=True)
    team=stats.groupby(['team','season','week'],as_index=False)[['attempts','carries','targets']].sum()
    ownpos=stats.groupby(['team','position','season','week'],as_index=False).fantasy_points_ppr.sum()
    allowed=stats.groupby(['opponent_team','position','season','week'],as_index=False).fantasy_points_ppr.sum()
    allowed=allowed.rename(columns={'opponent_team':'opponent'})
    output,names=_prior_lookup(frame,team,['team','season'],['attempts','carries','targets'],'team')
    extra,more=_prior_lookup(frame,ownpos,['team','position','season'],['fantasy_points_ppr'],'ownpos',windows=(3,))
    output=pd.concat([output,extra],axis=1);names+=more
    extra,more=_prior_lookup(frame,allowed,['opponent','position','season'],['fantasy_points_ppr'],'opp',windows=(3,))
    output=pd.concat([output,extra],axis=1);names+=more
    frame=pd.concat([frame,output],axis=1)
    for col in ['attempts','carries','targets']:
        name=f'v4_{col}_share3'
        frame[name]=frame[col+'_mean3']/frame[f'v4_team_{col}_mean3'].replace(0,np.nan)
        names.append(name)
    frame['v4_points_share3']=frame.fantasy_points_ppr_mean3/frame.v4_ownpos_fantasy_points_ppr_mean3.replace(0,np.nan)
    names.append('v4_points_share3')
    return frame,names
