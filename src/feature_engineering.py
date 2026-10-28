"""V2: opportunity, touchdown-independent production, and longer player history.

All group transforms shift before rolling. Target population remains identical
to v1, including zero-point candidates and the missing-game exclusion policy.
"""
import numpy as np
import pandas as pd

KEYS = ['player_id', 'season', 'week']
RAW = ['fantasy_points_ppr', 'targets', 'carries', 'attempts', 'receptions',
       'passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds',
       'rushing_tds', 'receiving_tds', 'passing_air_yards', 'receiving_air_yards',
       'receiving_yards_after_catch', 'target_share', 'wopr', 'air_yards_share']

def enhance(data, future, stats):
    source = stats.copy()
    for col in RAW:
        source[col] = pd.to_numeric(source.get(col, 0), errors='coerce').fillna(0)
    # Keep actual source records, plus synthetic DNP candidates and forecast rows.
    missing = pd.concat([data[KEYS], future[KEYS]], ignore_index=True)
    source = pd.concat([source[KEYS + RAW], missing], ignore_index=True)
    source = source.drop_duplicates(KEYS, keep='first').sort_values(KEYS).reset_index(drop=True)
    source[RAW] = source[RAW].fillna(0)
    source['opportunities'] = source.targets + source.carries
    source['no_td_ppr'] = source.fantasy_points_ppr - 4*source.passing_tds - 6*(source.rushing_tds+source.receiving_tds)
    source['ppr_zero'] = (source.fantasy_points_ppr == 0).astype(float)
    source['no_offense'] = ((source.attempts + source.targets + source.carries) == 0).astype(float)
    source['catch_rate'] = source.receptions / source.targets.replace(0, np.nan)
    source['yards_per_target'] = source.receiving_yards / source.targets.replace(0, np.nan)
    source['yards_per_carry'] = source.rushing_yards / source.carries.replace(0, np.nan)
    engineered = []
    # Current-season history, strictly before the target week.
    group = source.groupby(['player_id', 'season'], sort=False)
    for col in ['opportunities', 'no_td_ppr', 'passing_air_yards', 'receiving_air_yards',
                'receiving_yards_after_catch', 'wopr', 'air_yards_share',
                'catch_rate', 'yards_per_target', 'yards_per_carry', 'ppr_zero', 'no_offense']:
        for window in [3, 5]:
            name = f'v2_{col}_mean{window}'
            source[name] = group[col].transform(lambda s: s.shift(1).rolling(window,min_periods=1).mean())
            engineered.append(name)
    for window in [3, 5]:
        name=f'v2_ppr_median{window}'
        source[name]=group.fantasy_points_ppr.transform(lambda s:s.shift(1).rolling(window,min_periods=1).median())
        engineered.append(name)
    for span in [3, 6]:
        name=f'v2_ppr_ewm{span}'
        source[name]=group.fantasy_points_ppr.transform(lambda s:s.shift(1).ewm(span=span,adjust=False).mean())
        engineered.append(name)
    # Career rolling form may cross season boundaries, but never target-week time.
    career=source.groupby('player_id',sort=False)
    for col in ['fantasy_points_ppr','opportunities','no_td_ppr']:
        name=f'v2_career_{col}_mean8'
        source[name]=career[col].transform(lambda s:s.shift(1).rolling(8,min_periods=1).mean())
        engineered.append(name)
    result=source[KEYS+engineered]
    def merge(frame):
        joined=frame.merge(result,on=KEYS,how='left',validate='one_to_one',sort=False)
        # Recent opportunity changes often precede persistent scoring changes.
        joined['v2_ppr_trend']=joined.fantasy_points_ppr_mean3-joined.fantasy_points_ppr_season_mean
        joined['v2_opportunity_trend']=joined.v2_opportunities_mean3-joined.v2_opportunities_mean5
        joined['v2_td_dependency']=joined.fantasy_points_ppr_mean3-joined.v2_no_td_ppr_mean3
        joined['v2_recent_consistency']=joined.ppr_std5/(1+joined.fantasy_points_ppr_mean5.abs())
        return joined
    return merge(data),merge(future),engineered+['v2_ppr_trend','v2_opportunity_trend','v2_td_dependency','v2_recent_consistency']
