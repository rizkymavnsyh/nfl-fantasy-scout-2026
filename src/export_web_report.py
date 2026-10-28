"""Add the model report section (versions, calibration, importance) to dist/data.json."""
import json
import numpy as np
import pandas as pd
from pipeline import ROOT,dump,metrics

LABELS={
 'v6_form_x_implied':'Recent form × Vegas implied team total',
 'v2_ppr_ewm3':'Recent PPR form (EWM, 3 games)',
 'v2_ppr_ewm6':'PPR form (EWM, 6 games)',
 'v5_snap_pct_lag1':'Offensive snap share, last game',
 'v5_snap_pct_mean3':'Offensive snap share, last 3',
 'v5_snap_pct_trend':'Snap share trend',
 'v2_career_fantasy_points_ppr_mean8':'Career PPR, last 8 games',
 'v2_career_no_td_ppr_mean8':'Career PPR without TDs, last 8',
 'v2_career_opportunities_mean8':'Career touches + targets, last 8',
 'fantasy_points_ppr_lag1':'PPR points, last game',
 'fantasy_points_ppr_season_mean':'PPR points, season average',
 'v6_implied_team_total':'Vegas implied team total',
 'v6_total_line':'Vegas game total',
 'v6_team_spread':'Vegas spread (team view)',
 'v2_opportunities_mean3':'Touches + targets, last 3',
 'rushing_yards_lag1':'Rushing yards, last game',
 'v2_no_offense_mean3':'Games without an offensive touch, last 3',
 'carries_lag1':'Carries, last game',
 'v2_no_td_ppr_mean3':'PPR without TDs, last 3',
 'carries_season_mean':'Carries, season average',
 'target_share_season_mean':'Target share, season average',
 'target_share_lag1':'Target share, last game',
 'opponent_allowed_mean5':'Opponent PPR allowed to position, last 5',
 'v5_chances_per_snap3':'Touches + targets per snap',
 'history_games':'Games of history this season',
 'week':'Week of season','position':'Position',
}

def run():
    path=ROOT/'dist/data.json';payload=json.loads(path.read_text(encoding='utf-8'))
    m=json.loads((ROOT/'reports/metrics.json').read_text(encoding='utf-8'))
    vc=m['version_comparison'];v5=json.loads((ROOT/'reports/v5_comparison.json').read_text(encoding='utf-8'))['full_cohort']['model']
    versions=[{'version':'V1','note':'Rolling stats + gradient boosting','mae':vc['old']['mae'],'rmse':vc['old']['rmse'],'r2':vc['old']['r2']},
              {'version':'V2','note':'Opportunity and touchdown-free form','mae':vc['v2']['mae'],'rmse':vc['v2']['rmse'],'r2':vc['v2']['r2']},
              {'version':'V3','note':'LightGBM ensemble balanced for MAE and RMSE','mae':vc['new']['mae'],'rmse':vc['new']['rmse'],'r2':vc['new']['r2']},
              {'version':'V5','note':'Prior-week snap share and injury reports','mae':v5['mae'],'rmse':v5['rmse'],'r2':v5['r2']},
              {'version':'V6','note':'Vegas spread and implied team totals','mae':m['test']['mae'],'rmse':m['test']['rmse'],'r2':m['test']['r2']}]
    test=pd.read_csv(ROOT/'reports/predictions_test.csv',usecols=['position','season','week','prediction','target','baseline'])
    edges=np.unique(np.quantile(test.prediction,np.linspace(0,1,11)))
    test['bin']=pd.cut(test.prediction,edges,include_lowest=True)
    calibration=[{'predicted':float(g.prediction.mean()),'actual':float(g.target.mean()),'n':int(len(g))} for _,g in test.groupby('bin',observed=True)]
    sample=test.sample(700,random_state=7)
    scatter=[[round(float(r.prediction),1),round(float(r.target),1),r.position] for r in sample.itertuples()]
    importance=json.loads((ROOT/'reports/v6_feature_importance.json').read_text(encoding='utf-8'))['importance']
    top=[{'feature':k,'label':LABELS.get(k,k),'share':v,'new':k.startswith('v6_')} for k,v in list(importance.items())[:12]]
    by_position={pos:{'n':r['model']['n'],'model':r['model']['mae'],'baseline':r['baseline']['mae'],'previous':r.get('v5',{}).get('mae')} for pos,r in m['by_position'].items()}
    cv=next(c for c in m['cv_candidates'] if c['model']==m['selected_model'])
    payload['report']={'versions':versions,'calibration':calibration,'scatter':scatter,'importance':top,'by_position':by_position,
      'validation':{'mae':cv['cv_mae'],'rmse':cv['cv_rmse'],'folds':cv['folds']},
      'test':metrics(test.target,test.prediction),'baseline':metrics(test.target,test.baseline),
      'tolerance':m['tolerance'],'interval':m['interval'],'feature_count':m['feature_count'],'selected_model':m['selected_model']}
    dump(path,payload)
    print('Web report:',len(versions),'versions,',len(calibration),'calibration bins,',len(top),'features',flush=True)

if __name__=='__main__':run()
