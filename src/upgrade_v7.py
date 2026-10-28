"""V7: red-zone usage and expected touchdowns from play-by-play. Same full-population replay as V6."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from datetime import datetime,timezone
import hashlib,json
import joblib
import numpy as np
import pandas as pd
from pipeline import ROOT,dump,metrics
from forecast_models import FantasyEnsemble,make_model
from pregame_features import enrich as enrich_pregame,load_sources,fantasy_pool
from market_features import enrich as enrich_market,load_games
from redzone_features import enrich as enrich_redzone,load_plays,player_games
from improve_v4 import tolerance
from upgrade_v6 import importance

def compare(frame):
    return {'n':len(frame),'model':metrics(frame.target,frame.prediction),
      'v6':metrics(frame.target,frame.v6_prediction),'baseline':metrics(frame.target,frame.baseline),
      'tolerance':{'model':tolerance(frame.target,frame.prediction),'v6':tolerance(frame.target,frame.v6_prediction),
                   'baseline':tolerance(frame.target,frame.baseline)}}

def build_data():
    snaps,injuries=load_sources()
    data=pd.read_csv(ROOT/'reports/feature_dataset.csv')
    data,_=enrich_pregame(data,snaps,injuries)
    data,_=enrich_market(data,load_games())
    games,rates=player_games(load_plays())
    data,extra=enrich_redzone(data,games)
    data['fantasy_pool']=fantasy_pool(data)
    return data,extra,rates

def run():
    old=json.loads((ROOT/'reports/metrics.json').read_text(encoding='utf-8'))
    assert old.get('version')=='v6','V7 is evaluated against the active V6 model.'
    bundle=joblib.load(ROOT/'models/evaluation_model.joblib')
    data,extra,rates=build_data()
    cols=old['features']+extra
    specs=dict(bundle['model'].specifications);base=dict(bundle['model'].weights)
    specs.update({
      'v7_redzone_hurdle':{'columns':cols,'params':{'family':'hurdle','loss':'regression','leaves':15,'iterations':600,'minimum':60,'learning_rate':.03,'reg_lambda':40.}},
      'v7_redzone_mean':{'columns':cols,'params':{'family':'lightgbm','loss':'regression','leaves':15,'iterations':650,'minimum':60,'learning_rate':.025,'reg_lambda':40.}}})
    names=[n for n in specs if n.startswith('v7_')]
    blends={'v6_reference':base}
    for name in names:
        blends[name]={name:1.}
        for w in [.25,.5,.75]:
            blends[f'v6+{int(w*100)}pct_{name}']={**{k:v*(1-w) for k,v in base.items()},name:w}
    for w in [.25,.5,.75]:blends[f'redzone_hurdle+{int(w*100)}pct_redzone_mean']={'v7_redzone_hurdle':1-w,'v7_redzone_mean':w}
    coverage={str(int(y)):float(g.v7_x_td_mean3.notna().mean()) for y,g in data.groupby('season')}
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'specifications':specs,'blends':blends,
      'added_features':extra,'validation_seasons':[2023,2024],'td_rates':{k:{str(b):v for b,v in r.items()} if isinstance(r,dict) else r for k,r in rates.items()},
      'selection':'Minimize full-cohort mean CV RMSE, requiring MAE <= V6 and RMSE <= 99.75% of V6. Otherwise retain V6.',
      'time_policy':'Play-by-play aggregates from games strictly before the target week. Touchdown rates by field-position bucket estimated on 2019-2022 only.',
      'pbp_coverage_by_season':coverage,
      'feature_code_sha256':hashlib.sha256((ROOT/'src/redzone_features.py').read_bytes()).hexdigest()}
    dump(ROOT/'reports/v7_plan.json',plan)
    folds={y:data[data.season.eq(y)] for y in [2023,2024]};cache=np.load(ROOT/'reports/cv_v6_predictions.npz');cv={}
    for year,va in folds.items():
        train=data[data.season<year]
        for name in base:cv[(name,year)]=cache[f'{name}_{year}']
        for name in names:
            spec=specs[name];model=make_model(spec['columns'],**spec['params']).fit(train[cols],train.target)
            cv[(name,year)]=np.maximum(0,model.predict(va[cols]));s=metrics(va.target,cv[(name,year)])
            print(f'CV {year} {name}: MAE {s["mae"]:.5f} RMSE {s["rmse"]:.5f}',flush=True)
    candidates=[]
    for name,weights in blends.items():
        fs=[]
        for year,va in folds.items():
            pred=sum(w*cv[(n,year)] for n,w in weights.items())
            fs.append({'validation_season':year,'train_n':int((data.season<year).sum()),**metrics(va.target,pred)})
        candidates.append({'model':name,'weights':weights,'folds':fs,
          'cv_mae':float(np.mean([r['mae'] for r in fs])),'cv_rmse':float(np.mean([r['rmse'] for r in fs]))})
    reference=next(r for r in candidates if r['model']=='v6_reference')
    eligible=[r for r in candidates if r['cv_mae']<=reference['cv_mae'] and r['cv_rmse']<=reference['cv_rmse']*.9975]
    chosen=min(eligible,key=lambda r:r['cv_rmse']) if eligible else reference
    validation={'reference':reference,'selected':chosen,'candidates':candidates,'promoted':bool(eligible),
                'locked_before_replay_at_utc':datetime.now(timezone.utc).isoformat(),'rule':plan['selection']}
    dump(ROOT/'reports/v7_validation.json',validation)
    np.savez_compressed(ROOT/'reports/cv_v7_predictions.npz',**{f'{n}_{year}':v for (n,year),v in cv.items()})
    print('SELECTED',json.dumps({k:chosen[k] for k in ['model','weights','cv_mae','cv_rmse']}),'REFERENCE',reference['cv_mae'],reference['cv_rmse'],flush=True)
    if not eligible:
        print('V7 not promoted. V6 retained.',flush=True);return validation
    train=data[data.season<=2024];test=data[data.season>=2025].copy();frozen=pd.read_csv(ROOT/'reports/predictions_test.csv')
    assert np.array_equal(test[['player_id','season','week']].to_numpy(),frozen[['player_id','season','week']].to_numpy())
    assert np.array_equal(test.target,frozen.target)
    test['v6_prediction']=frozen.prediction.to_numpy()
    model=FantasyEnsemble(specs,chosen['weights']).fit(train[cols],train.target)
    test['prediction']=model.predict(test[cols])
    calibration=[]
    for year,va in folds.items():
        part=va[['position','target']].copy();part['prediction']=sum(w*cv[(n,year)] for n,w in chosen['weights'].items());calibration.append(part)
    calibration=pd.concat(calibration)
    bands={pos:float(np.quantile(abs(g.target-g.prediction),.8)) for pos,g in calibration.groupby('position')}
    test['band']=test.position.map(bands);test['low']=np.maximum(0,test.prediction-test.band);test['high']=test.prediction+test.band
    comparison={'promoted':True,'full_cohort':compare(test),'fantasy_pool':compare(test[test.fantasy_pool]),
                'features':extra,'time_note':plan['time_policy'],'validation':{'v6':reference,'selected':chosen},
                'replay_status':'2025/26 previously inspected for earlier versions. V7 selection used 2023/24 only; test scored once after selection.'}
    dump(ROOT/'reports/v7_comparison.json',comparison)
    weights=importance(model)
    dump(ROOT/'reports/v7_feature_importance.json',{'definition':'Blend-weighted share of LightGBM split gain on the 2019-2024 evaluation model. Not a causal effect.','importance':weights})
    print('REPLAY',json.dumps({k:comparison[k]['model'] for k in ['full_cohort','fantasy_pool']}),'V6',json.dumps(comparison['full_cohort']['v6']),flush=True)
    report={**old,'version':'v7','selected_model':chosen['model'],'features':cols,'feature_count':len(cols),
      'created_at':datetime.now(timezone.utc).isoformat(),'cv_candidates':candidates,
      'train':metrics(train.target,model.predict(train[cols])),'test':metrics(test.target,test.prediction),
      'baseline':metrics(test.target,test.baseline),'redzone_comparison':comparison,
      'feature_importance':dict(list(weights.items())[:20]),
      'by_season':{str(int(y)):{'model':metrics(g.target,g.prediction),'baseline':metrics(g.target,g.baseline),'v6':metrics(g.target,g.v6_prediction)} for y,g in test.groupby('season')},
      'by_position':{pos:{'model':metrics(g.target,g.prediction),'baseline':metrics(g.target,g.baseline),'v6':metrics(g.target,g.v6_prediction)} for pos,g in test.groupby('position')},
      'tolerance':comparison['full_cohort']['tolerance'],
      'interval':{'nominal':.8,'coverage_test':float(((test.target>=test.low)&(test.target<=test.high)).mean()),
         'half_width_by_position':bands,'description':'Empirical historical CV residuals, not an individual guarantee.'}}
    report['mae_improvement_pct']=100*(report['baseline']['mae']-report['test']['mae'])/report['baseline']['mae']
    report['protocol']={**old['protocol'],'selection':plan['selection'],'time_policy':old['protocol']['time_policy']+' '+plan['time_policy']}
    dump(ROOT/'reports/metrics.json',report);dump(ROOT/'reports/v7_metrics.json',report)
    joblib.dump({'model':model,'name':chosen['model'],'columns':cols},ROOT/'models/evaluation_model.joblib')
    test.to_csv(ROOT/'reports/predictions_test.csv',index=False);test.to_csv(ROOT/'reports/v7_predictions_test.csv',index=False)
    return comparison

if __name__=='__main__':run()
