"""V3: prioritize RMSE while retaining most of V2's MAE improvement.

The 2025/26 data has been inspected and remains a retrospective replay.
All hyperparameters and selection rules below are recorded before validation.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
from pipeline import ROOT,dump,metrics,export_dashboard
from forecast_models import make_model,FantasyEnsemble

def run():
    v1=json.loads((ROOT/'reports/v1/metrics.json').read_text(encoding='utf-8'))
    v2=json.loads((ROOT/'reports/v2/metrics.json').read_text(encoding='utf-8'))
    data=pd.read_csv(ROOT/'reports/feature_dataset.csv')
    future=pd.read_csv(ROOT/'reports/forecasts_week8.csv')
    cols=v2['features']
    specs={
      'v1_squared':{'columns':v1['features'],'params':{}},
      'v2_median':{'columns':cols,'params':{'loss':'absolute_error','leaves':26,'iterations':200,'minimum':40,'learning_rate':.05}},
    }
    settings=[
      ('lgb_mean15','regression',15,650,60,.025,40.,.9),
      ('lgb_mean31','regression',31,500,80,.03,60.,.9),
      ('lgb_mean7','regression',7,900,90,.025,50.,.9),
      ('lgb_huber','huber',15,600,60,.035,30.,.9),
      ('lgb_fair','fair',15,600,60,.03,30.,.9),
      ('lgb_median15','regression_l1',15,700,60,.025,30.,.9),
      ('lgb_median31','regression_l1',31,500,80,.03,40.,.9),
      ('lgb_quantile60','quantile',15,700,60,.03,35.,.6),
    ]
    for name,loss,leaves,iters,minimum,lr,reg,alpha in settings:
        specs[name]={'columns':cols,'params':{'family':'lightgbm','loss':loss,'leaves':leaves,
          'iterations':iters,'minimum':minimum,'learning_rate':lr,'reg_lambda':reg,'alpha':alpha}}
    blends={name:{name:1.} for name in specs}
    for mean in ['v1_squared','lgb_mean15','lgb_mean31','lgb_mean7']:
        for robust in ['v2_median','lgb_median15','lgb_fair']:
            for fraction in [.25,.5,.75]:
                blends[f'{mean}+{int(fraction*100)}pct_{robust}']={mean:1-fraction,robust:fraction}
    dump(ROOT/'reports/rebalance_plan.json',{'created_at':datetime.now(timezone.utc).isoformat(),
      'validation_seasons':[2023,2024],
      'selection':'Minimize validation RMSE subject to validation MAE <= 105% of V2, and validation RMSE <= V1. If empty, minimize RMSE subject to MAE limit and disclose failure to dominate V1.',
      'replay_status':'2025/26 repeatedly inspected. Retrospective comparison, no untouched holdout claim.',
      'specifications':specs,'blends':blends})
    fold_data={year:data[data.season==year] for year in [2023,2024]}
    cv={}
    for year,va in fold_data.items():
        tr=data[data.season<year]
        for name,spec in specs.items():
            model=make_model(spec['columns'],**spec['params']).fit(tr[spec['columns']],tr.target)
            cv[(name,year)]=np.maximum(0,model.predict(va[spec['columns']]))
            r=metrics(va.target,cv[(name,year)])
            print(f'CV {year} {name}: MAE {r["mae"]:.4f} RMSE {r["rmse"]:.4f}',flush=True)
    scores=[]
    for name,weights in blends.items():
        folds=[]
        for year,va in fold_data.items():
            p=sum(w*cv[(key,year)] for key,w in weights.items())
            folds.append({'validation_season':year,'train_n':int((data.season<year).sum()),**metrics(va.target,p)})
        scores.append({'model':name,'weights':weights,'cv_mae':float(np.mean([f['mae'] for f in folds])),
                       'cv_rmse':float(np.mean([f['rmse'] for f in folds])),'folds':folds})
    reference=next(c for c in scores if c['model']=='v1_squared')
    median=next(c for c in scores if c['model']=='v2_median')
    limited=[c for c in scores if c['cv_mae']<=median['cv_mae']*1.05]
    pareto=[c for c in limited if c['cv_rmse']<=reference['cv_rmse']]
    chosen=min(pareto or limited,key=lambda c:c['cv_rmse'])
    dump(ROOT/'reports/rebalance_validation.json',{'selected':chosen,'reference_v1':reference,
       'reference_v2':median,'candidates':scores,'constraints_met':bool(pareto),'locked_before_replay':True})
    np.savez_compressed(ROOT/'reports/cv_v3_predictions.npz',**{f'{name}_{year}':p for (name,year),p in cv.items()})
    print('SELECTED',chosen['model'],chosen['cv_mae'],chosen['cv_rmse'],flush=True)
    train=data[data.season<=2024]
    test=data[data.season>=2025].copy()
    frozen=pd.read_csv(ROOT/'reports/v1/predictions_test.csv')
    assert len(test)==len(frozen) and np.allclose(test.target,frozen.target)
    assert (test[['player_id','season','week']].to_numpy()==frozen[['player_id','season','week']].to_numpy()).all()
    evaluation=FantasyEnsemble(specs,chosen['weights']).fit(train[cols],train.target)
    test['prediction']=evaluation.predict(test[cols])
    cal=[]
    for year,va in fold_data.items():
        part=va[['position','target']].copy()
        part['prediction']=sum(w*cv[(key,year)] for key,w in chosen['weights'].items())
        cal.append(part)
    cal=pd.concat(cal)
    bands={pos:float(np.quantile(abs(g.target-g.prediction),.8)) for pos,g in cal.groupby('position')}
    test['band']=test.position.map(bands);test['low']=np.maximum(0,test.prediction-test.band);test['high']=test.prediction+test.band
    r={**v2,'version':'v3','created_at':datetime.now(timezone.utc).isoformat(),'selected_model':chosen['model'],
      'train':metrics(train.target,evaluation.predict(train[cols])),'test':metrics(test.target,test.prediction),
      'cv_candidates':scores,
      'by_season':{str(int(y)):{'model':metrics(g.target,g.prediction),'baseline':metrics(g.target,g.baseline)} for y,g in test.groupby('season')},
      'by_position':{pos:{'model':metrics(g.target,g.prediction),'baseline':metrics(g.target,g.baseline)} for pos,g in test.groupby('position')},
      'interval':{'nominal':.8,'coverage_test':float(((test.target>=test.low)&(test.target<=test.high)).mean()),
        'description':'Empirical development-fold residual band, not an individual guarantee.','half_width_by_position':bands},
      'version_comparison':{'old':v1['test'],'v2':v2['test'],'new':metrics(test.target,test.prediction),
        'cv_mae_old':reference['cv_mae'],'cv_mae_new':chosen['cv_mae'],'cv_rmse_old':reference['cv_rmse'],'cv_rmse_new':chosen['cv_rmse'],
        'evaluation_status':'Repeated retrospective replay on already-inspected 2025/26 data. Independent future-week verification still required.'}}
    r['protocol']={**v2['protocol'],'selection':'V3 minimizes historical validation RMSE subject to retaining MAE improvement. See rebalance_plan.json.',
      'test_status':'Previously inspected repeatedly. Retrospective replay, not an untouched independent holdout.'}
    r['mae_improvement_pct']=100*(r['baseline']['mae']-r['test']['mae'])/r['baseline']['mae']
    r['version_comparison']['mae_reduction_from_v1_pct']=100*(v1['test']['mae']-r['test']['mae'])/v1['test']['mae']
    r['version_comparison']['rmse_reduction_from_v2_pct']=100*(v2['test']['rmse']-r['test']['rmse'])/v2['test']['rmse']
    print('REPLAY',json.dumps(r['test']),flush=True)
    joblib.dump({'model':evaluation,'name':chosen['model'],'columns':cols},ROOT/'models/evaluation_model.joblib')
    production=FantasyEnsemble(specs,chosen['weights']).fit(data[cols],data.target)
    joblib.dump({'model':production,'name':chosen['model'],'columns':cols},ROOT/'models/forecast_model.joblib')
    future['prediction']=production.predict(future[cols]);future['low']=np.maximum(0,future.prediction-future.position.map(bands));future['high']=future.prediction+future.position.map(bands)
    future.to_csv(ROOT/'reports/forecasts_week8.csv',index=False)
    test.to_csv(ROOT/'reports/predictions_test.csv',index=False)
    dump(ROOT/'reports/metrics.json',r)
    dump(ROOT/'reports/rebalance_comparison.json',r['version_comparison'])
    export_dashboard(data,future,test,r)
    print('V3 exported',flush=True)
    return r

if __name__=='__main__':run()
