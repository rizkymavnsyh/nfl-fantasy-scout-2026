"""Bounded V2 experiments. Select on 2023/24, then replay already-seen 2025/26."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from pipeline import ROOT, dump, load_data, feature_columns, metrics, export_dashboard
from feature_engineering import enhance
from forecast_models import FantasyEnsemble, make_model

def run():
    original=json.loads((ROOT/'reports/v1/metrics.json').read_text(encoding='utf-8'))
    data=pd.read_csv(ROOT/'reports/feature_dataset.csv')
    future=pd.read_csv(ROOT/'reports/forecasts_week8.csv')
    # Re-running V2 starts from the same unchanged V1 feature columns.
    data=data.drop(columns=[c for c in data if c.startswith('v2_')])
    future=future.drop(columns=[c for c in future if c.startswith('v2_')])
    stats,_=load_data()
    basecols=original['features']
    data,future,extra=enhance(data,future,stats)
    columns=basecols+extra
    specs={
      'v1_squared':{'columns':basecols,'params':{}},
      'v1_absolute':{'columns':basecols,'params':{'loss':'absolute_error','leaves':15,'iterations':220,'minimum':50,'learning_rate':.055}},
      'v2_squared':{'columns':columns,'params':{'iterations':220,'leaves':16,'minimum':45,'learning_rate':.05}},
      'v2_absolute':{'columns':columns,'params':{'loss':'absolute_error','leaves':16,'iterations':220,'minimum':45,'learning_rate':.055}},
      'v2_absolute_smooth':{'columns':columns,'params':{'loss':'absolute_error','leaves':10,'iterations':300,'minimum':65,'learning_rate':.045}},
      'v2_absolute_flexible':{'columns':columns,'params':{'loss':'absolute_error','leaves':26,'iterations':200,'minimum':40,'learning_rate':.05}},
    }
    blends={name:{name:1.} for name in specs}
    for mean in ['v1_squared','v2_squared']:
        for med in ['v1_absolute','v2_absolute','v2_absolute_smooth']:
            for weight in [.25,.5,.75]:
                blends[f'{mean}+{int(weight*100)}pct_{med}']={mean:1-weight,med:weight}
    dump(ROOT/'reports/upgrade_plan.json',{
      'created_at':datetime.now(timezone.utc).isoformat(),
      'selection_data':'2023 and 2024 expanding-season validation only',
      'replay_data':'2025 and 2026 already inspected during V1. Not a new untouched holdout.',
      'cohort':'Exact same rows, labels and baseline as V1',
      'constraint':'Validation RMSE may not increase more than 3% relative to V1. Minimize MAE among eligible candidates.',
      'specifications':specs,'blends':blends})
    cvpred={}
    folds={}
    for year in [2023,2024]:
        tr=data[data.season<year]; va=data[data.season==year]
        folds[year]=va
        for name,spec in specs.items():
            model=make_model(spec['columns'],**spec['params']).fit(tr[spec['columns']],tr.target)
            cvpred[(name,year)]=np.maximum(0,model.predict(va[spec['columns']]))
            r=metrics(va.target,cvpred[(name,year)])
            print(f'Validation {year} {name}: MAE={r["mae"]:.5f} RMSE={r["rmse"]:.5f}',flush=True)
    scores=[]
    for name,weights in blends.items():
        fold=[]
        for year,va in folds.items():
            p=sum(weight*cvpred[(key,year)] for key,weight in weights.items())
            fold.append({'validation_season':year,'train_n':int((data.season<year).sum()),**metrics(va.target,p)})
        scores.append({'model':name,'weights':weights,'cv_mae':float(np.mean([f['mae'] for f in fold])),
                       'cv_rmse':float(np.mean([f['rmse'] for f in fold])),'folds':fold})
    ref=next(s for s in scores if s['model']=='v1_squared')
    valid=[s for s in scores if s['cv_rmse']<=ref['cv_rmse']*1.03]
    chosen=min(valid,key=lambda s:s['cv_mae'])
    dump(ROOT/'reports/upgrade_validation.json',{'reference':ref,'selected':chosen,'candidates':scores,
         'selection_locked_before_replay':True})
    print('SELECTED FROM VALIDATION:',chosen['model'],chosen['cv_mae'],chosen['cv_rmse'],flush=True)
    if chosen['cv_mae']>=ref['cv_mae']:
        print('No validation improvement. V1 retained.',flush=True)
        return
    train=data[data.season<=2024]
    test=data[data.season>=2025].copy()
    frozen_old=pd.read_csv(ROOT/'reports/v1/predictions_test.csv')
    assert len(test)==len(frozen_old)
    assert np.allclose(test.target,frozen_old.target)
    assert (test[['player_id','season','week']].to_numpy()==frozen_old[['player_id','season','week']].to_numpy()).all()
    evaluation=FantasyEnsemble(specs,chosen['weights']).fit(train[columns],train.target)
    test['prediction']=evaluation.predict(test[columns])
    calibration=[]
    for year,va in folds.items():
        frame=va[['position','target']].copy()
        frame['prediction']=sum(w*cvpred[(key,year)] for key,w in chosen['weights'].items())
        calibration.append(frame)
    calibration=pd.concat(calibration)
    bands={pos:float(np.quantile(abs(g.target-g.prediction),.8)) for pos,g in calibration.groupby('position')}
    test['band']=test.position.map(bands)
    test['low']=np.maximum(0,test.prediction-test.band);test['high']=test.prediction+test.band
    report={**original,'created_at':datetime.now(timezone.utc).isoformat(),
      'version':'v2','selected_model':chosen['model'],'features':columns,'feature_count':len(columns),
      'train':metrics(train.target,evaluation.predict(train[columns])),
      'test':metrics(test.target,test.prediction),'baseline':metrics(test.target,test.baseline),
      'cv_candidates':scores,
      'by_season':{str(int(y)):{'model':metrics(g.target,g.prediction),'baseline':metrics(g.target,g.baseline)} for y,g in test.groupby('season')},
      'by_position':{pos:{'model':metrics(g.target,g.prediction),'baseline':metrics(g.target,g.baseline)} for pos,g in test.groupby('position')},
      'interval':{'nominal':.8,'coverage_test':float(((test.target>=test.low)&(test.target<=test.high)).mean()),
        'description':'Empirical absolute-error bands from 2023/24 development folds, reused for selection. Not individual guarantees.',
        'half_width_by_position':bands},
      'version_comparison':{'old':original['test'],'new':metrics(test.target,test.prediction),
        'cv_mae_old':ref['cv_mae'],'cv_mae_new':chosen['cv_mae'],
        'evaluation_status':'Retrospective replay of the already-inspected V1 test set. New future weeks required for independent confirmation.'},
    }
    report['mae_improvement_pct']=100*(report['baseline']['mae']-report['test']['mae'])/report['baseline']['mae']
    report['version_comparison']['mae_reduction_from_v1_pct']=100*(original['test']['mae']-report['test']['mae'])/original['test']['mae']
    report['protocol']={**original['protocol'],
        'selection':'V2 selected only on expanding 2023/24 validation; RMSE degradation capped at 3%.',
        'test_status':'Previously inspected during V1, now retrospective replay, not an untouched independent holdout.',
        'injury_info':'No live injury reports or official active roster filter'}
    dump(ROOT/'reports/upgrade_comparison.json',report['version_comparison'])
    print('REPLAY:',json.dumps(report['test']),'OLD:',json.dumps(original['test']),flush=True)
    # Promote only after locked validation selection. Test replay does not choose
    # between candidates. All comparisons and limitations remain in the report.
    joblib.dump({'model':evaluation,'columns':columns,'name':chosen['model']},ROOT/'models/evaluation_model.joblib')
    production=FantasyEnsemble(specs,chosen['weights']).fit(data[columns],data.target)
    joblib.dump({'model':production,'columns':columns,'name':chosen['model']},ROOT/'models/forecast_model.joblib')
    future['prediction']=production.predict(future[columns])
    future['low']=np.maximum(0,future.prediction-future.position.map(bands));future['high']=future.prediction+future.position.map(bands)
    future.to_csv(ROOT/'reports/forecasts_week8.csv',index=False)
    data.to_csv(ROOT/'reports/feature_dataset.csv',index=False)
    test.to_csv(ROOT/'reports/predictions_test.csv',index=False)
    dump(ROOT/'reports/metrics.json',report)
    export_dashboard(data,future,test,report)
    print('V2 forecast and dashboard data exported.',flush=True)
    return report

if __name__=='__main__':run()
