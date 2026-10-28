"""Role and previous-week injury features. Same full-population replay as V3."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from datetime import datetime,timezone
import hashlib,json
import joblib
import numpy as np
import pandas as pd
from pipeline import ROOT,dump,metrics
from forecast_models import FantasyEnsemble,make_model
from pregame_features import enrich,load_sources,fantasy_pool
from improve_v4 import tolerance

def compare(frame):
    return {'n':len(frame),'model':metrics(frame.target,frame.prediction),
      'v3':metrics(frame.target,frame.v3_prediction),'baseline':metrics(frame.target,frame.baseline),
      'tolerance':{'model':tolerance(frame.target,frame.prediction),'v3':tolerance(frame.target,frame.v3_prediction),
                   'baseline':tolerance(frame.target,frame.baseline)}}

def run():
    old=json.loads((ROOT/'reports/v3/metrics.json').read_text(encoding='utf-8'))
    bundle=joblib.load(ROOT/'reports/v3/evaluation_model.joblib')
    snaps,injuries=load_sources()
    data=pd.read_csv(ROOT/'reports/feature_dataset.csv')
    data,extra=enrich(data,snaps,injuries);cols=old['features']+extra
    data['fantasy_pool']=fantasy_pool(data)
    specs=dict(bundle['model'].specifications);base=bundle['model'].weights
    specs.update({
      'v5_role_mean':{'columns':cols,'params':{'family':'lightgbm','loss':'regression','leaves':15,'iterations':650,'minimum':60,'learning_rate':.025,'reg_lambda':40.}},
      'v5_role_fair':{'columns':cols,'params':{'family':'lightgbm','loss':'fair','leaves':15,'iterations':600,'minimum':60,'learning_rate':.03,'reg_lambda':30.}},
      'v5_role_hurdle':{'columns':cols,'params':{'family':'hurdle','loss':'regression','leaves':15,'iterations':600,'minimum':60,'learning_rate':.03,'reg_lambda':40.}},
      'v5_role_position':{'columns':cols,'params':{'family':'position','loss':'regression','leaves':7,'iterations':600,'minimum':50,'learning_rate':.025,'reg_lambda':45.}}})
    names=[n for n in specs if n.startswith('v5_')]
    blends={'v3_reference':base}
    for name in names:
        blends[name]={name:1.}
        for w in [.25,.5,.75]:
            blends[f'v3+{int(w*100)}pct_{name}']={**{k:v*(1-w) for k,v in base.items()},name:w}
    for companion in ['v5_role_fair','v5_role_hurdle','v5_role_position']:
        for w in [.25,.5,.75]:blends[f'role_mean+{int(w*100)}pct_{companion}']={'v5_role_mean':1-w,companion:w}
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'specifications':specs,'blends':blends,
      'added_features':extra,'validation_seasons':[2023,2024],
      'selection':'Minimize full-cohort mean CV RMSE, requiring MAE <= V3 and RMSE <= 99.75% of V3. Otherwise retain V3.',
      'fantasy_pool_rule':'QB prior mean3 attempts>=15 or prior snap percentage>=0.5. RB/WR/TE prior mean3 chances>=5 or snap mean3>=0.25. Snap age<=2 weeks when known.',
      'cohort_policy':'Full cohort retained for all headline model comparisons. Fantasy-pool results reported separately with the identical subset for V3, V5 and baseline.',
      'time_policy':'Strictly prior-week snaps and injury reports. Dated injury updates after the target date excluded. Archived snap first-publication timestamps are unavailable, so historical results are replay rather than certified point-in-time reconstruction.',
      'source_manifest_sha256':hashlib.sha256((ROOT/'reports/pregame_sources.json').read_bytes()).hexdigest()}
    plan['feature_code_sha256']=hashlib.sha256((ROOT/'src/pregame_features.py').read_bytes()).hexdigest()
    plan['snap_history_coverage_by_season']={str(int(y)):float(g.v5_snap_history_available.mean()) for y,g in data.groupby('season')}
    dump(ROOT/'reports/v5_plan.json',plan)
    folds={y:data[data.season.eq(y)] for y in [2023,2024]};cache=np.load(ROOT/'reports/cv_v3_predictions.npz');cv={}
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
    reference=next(r for r in candidates if r['model']=='v3_reference')
    eligible=[r for r in candidates if r['cv_mae']<=reference['cv_mae'] and r['cv_rmse']<=reference['cv_rmse']*.9975]
    chosen=min(eligible,key=lambda r:r['cv_rmse']) if eligible else reference
    validation={'reference':reference,'selected':chosen,'candidates':candidates,'promoted':bool(eligible),
                 'locked_before_replay_at_utc':datetime.now(timezone.utc).isoformat(),'rule':plan['selection']}
    dump(ROOT/'reports/v5_validation.json',validation)
    np.savez_compressed(ROOT/'reports/cv_v5_predictions.npz',**{f'{n}_{year}':v for (n,year),v in cv.items()})
    print('SELECTED',json.dumps(chosen),flush=True)
    train=data[data.season<=2024];test=data[data.season>=2025].copy();frozen=pd.read_csv(ROOT/'reports/v3/predictions_test.csv')
    assert np.array_equal(test[['player_id','season','week']].to_numpy(),frozen[['player_id','season','week']].to_numpy())
    assert np.array_equal(test.target,frozen.target)
    test['v3_prediction']=frozen.prediction.to_numpy()
    model=FantasyEnsemble(specs,chosen['weights']).fit(train[cols],train.target) if eligible else bundle['model']
    test['prediction']=model.predict(test[cols])
    calibration=[]
    for year,va in folds.items():
        part=va[['position','target']].copy();part['prediction']=sum(w*cv[(n,year)] for n,w in chosen['weights'].items());calibration.append(part)
    calibration=pd.concat(calibration)
    bands={pos:float(np.quantile(abs(g.target-g.prediction),.8)) for pos,g in calibration.groupby('position')}
    test['band']=test.position.map(bands);test['low']=np.maximum(0,test.prediction-test.band);test['high']=test.prediction+test.band
    comparison={'promoted':bool(eligible),'full_cohort':compare(test),'fantasy_pool':compare(test[test.fantasy_pool]),
                'definition':plan['fantasy_pool_rule'],'cohort_note':plan['cohort_policy'],'time_note':plan['time_policy'],
                'features':extra,'source_coverage':json.loads((ROOT/'reports/pregame_id_coverage.json').read_text(encoding='utf-8')),
                'validation':{'v3':reference,'selected':chosen},'replay_status':'2025/26 already inspected. Retrospective replay only.'}
    dump(ROOT/'reports/v5_comparison.json',comparison)
    test.to_csv(ROOT/'reports/v5_predictions_test.csv',index=False)
    print('REPLAY',json.dumps({k:comparison[k] for k in ['promoted','full_cohort','fantasy_pool']}),flush=True)
    if eligible:
        report={**old,'version':'v5','selected_model':chosen['model'],'features':cols,'feature_count':len(cols),
          'created_at':datetime.now(timezone.utc).isoformat(),'cv_candidates':candidates,
          'train':metrics(train.target,model.predict(train[cols])),'test':metrics(test.target,test.prediction),
          'baseline':metrics(test.target,test.baseline),'pregame_comparison':comparison,
          'by_season':{str(int(y)):{'model':metrics(g.target,g.prediction),'baseline':metrics(g.target,g.baseline)} for y,g in test.groupby('season')},
          'by_position':{pos:{'model':metrics(g.target,g.prediction),'baseline':metrics(g.target,g.baseline)} for pos,g in test.groupby('position')},
          'tolerance':comparison['full_cohort']['tolerance'],
          'interval':{'nominal':.8,'coverage_test':float(((test.target>=test.low)&(test.target<=test.high)).mean()),
             'half_width_by_position':bands,'description':'Empirical historical CV residuals, not an individual guarantee.'}}
        # Preserve V1/V2/V3 comparison verbatim as historical results. V5 has its own panel.
        report['mae_improvement_pct']=100*(report['baseline']['mae']-report['test']['mae'])/report['baseline']['mae']
        report['protocol']={**old['protocol'],'selection':plan['selection'],'time_policy':plan['time_policy'],
                            'test_status':'Previously inspected repeatedly. Retrospective replay, not an untouched independent holdout.'}
        report['forecast_snapshot']=json.loads((ROOT/'reports/forecast_snapshot.json').read_text(encoding='utf-8'))
        dump(ROOT/'reports/metrics.json',report);dump(ROOT/'reports/v5_metrics.json',report)
        joblib.dump({'model':model,'name':chosen['model'],'columns':cols},ROOT/'models/evaluation_model.joblib')
        test.to_csv(ROOT/'reports/predictions_test.csv',index=False)
    return comparison

if __name__=='__main__':run()
