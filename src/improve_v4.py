"""Bounded V4 experiment. Select on historical CV, then one retrospective replay."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from datetime import datetime,timezone
import hashlib,json
import joblib
import numpy as np
import pandas as pd
from pipeline import ROOT,dump,metrics,load_data
from forecast_models import FantasyEnsemble,make_model
from context_features import enrich

def tolerance(y,p):
    y=np.asarray(y);p=np.asarray(p);mask=y!=0
    return {'definition':'Fraction with absolute error <= fixed PPR-point tolerance. Not classification accuracy.',
       'within_3':float((abs(y-p)<=3).mean()),'within_5':float((abs(y-p)<=5).mean()),
       'nonzero_n':int(mask.sum()),'nonzero_within_5':float((abs(y[mask]-p[mask])<=5).mean())}

def run(stage2=False):
    old=json.loads((ROOT/'reports/v3/metrics.json').read_text(encoding='utf-8'))
    old_bundle=joblib.load(ROOT/'reports/v3/evaluation_model.joblib')
    data=pd.read_csv(ROOT/'reports/feature_dataset.csv')
    stats,_=load_data()
    data,extra=enrich(data,stats);cols=old['features']+extra
    specs=dict(old_bundle['model'].specifications)
    specs.update({
      'v4_context_mean':{'columns':cols,'params':{'family':'lightgbm','loss':'regression','leaves':15,'iterations':650,'minimum':60,'learning_rate':.025,'reg_lambda':40.}},
      'v4_context_fair':{'columns':cols,'params':{'family':'lightgbm','loss':'fair','leaves':15,'iterations':600,'minimum':60,'learning_rate':.03,'reg_lambda':30.}},
      'v4_context31':{'columns':cols,'params':{'family':'lightgbm','loss':'regression','leaves':31,'iterations':500,'minimum':80,'learning_rate':.03,'reg_lambda':60.}},
      'v4_position_mean':{'columns':cols,'params':{'family':'position','loss':'regression','leaves':7,'iterations':600,'minimum':50,'learning_rate':.025,'reg_lambda':45.}},
      'v4_position_fair':{'columns':cols,'params':{'family':'position','loss':'fair','leaves':7,'iterations':600,'minimum':50,'learning_rate':.03,'reg_lambda':35.}},
      'v4_hurdle':{'columns':cols,'params':{'family':'hurdle','loss':'regression','leaves':15,'iterations':600,'minimum':60,'learning_rate':.03,'reg_lambda':40.}},
    })
    base=old_bundle['model'].weights
    blends={'v3_reference':base}
    new=[n for n in specs if n.startswith('v4_')]
    for name in new:
        blends[name]={name:1.}
        for weight in [.25,.5,.75]:
            blends[f'v3+{int(weight*100)}pct_{name}']={**{k:v*(1-weight) for k,v in base.items()},name:weight}
    for name in ['v4_position_mean','v4_hurdle']:
        for weight in [.25,.5,.75]:
            blends[f'v4_context+{int(weight*100)}pct_{name}']={'v4_context_mean':1-weight,name:weight}
    if stage2:
        for label,used in [('base',old['features']),('context',cols)]:
            for decay in ([.65,.85] if label=='base' else [.65]):
                for loss in ['regression','fair']:
                    name=f'v4_recent_{label}_{int(decay*100)}_{loss}'
                    specs[name]={'columns':used+['season'],'params':{'family':'recency','season_decay':decay,
                      'loss':loss,'leaves':15,'iterations':650 if loss=='regression' else 600,
                      'minimum':60,'learning_rate':.025 if loss=='regression' else .03,'reg_lambda':40. if loss=='regression' else 30.}}
                    blends[name]={name:1.}
                    for weight in [.25,.5,.75]:
                        blends[f'v3+{int(weight*100)}pct_{name}']={**{k:v*(1-weight) for k,v in base.items()},name:weight}
                mean=f'v4_recent_{label}_{int(decay*100)}_regression';fair=f'v4_recent_{label}_{int(decay*100)}_fair'
                for weight in [.25,.5]:
                    blends[f'{mean}+{int(weight*100)}pct_fair']={mean:1-weight,fair:weight}
        cols=cols+['season']
        # Phase 1 predictions are retained; phase 2 is recorded before its fits.
        new=[n for n in specs if n.startswith('v4_')]
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'validation_seasons':[2023,2024],
      'specifications':specs,'blends':blends,'features_added':extra,
      'selection':'Minimum mean CV RMSE among candidates with CV MAE <= V3 and CV RMSE at least 0.25% lower than V3. Otherwise retain V3.',
      'tolerances_fixed_before_evaluation':[3,5],
      'scope':'Repeated historical development and retrospective replay. No independent holdout claim.',
      'feature_dataset_sha256':hashlib.sha256((ROOT/'reports/feature_dataset.csv').read_bytes()).hexdigest()}
    plan['phase']=2 if stage2 else 1
    plan['season_input_note']='For recency models, season controls fit weights and is not a predictor.'
    dump(ROOT/('reports/v4_plan_phase2.json' if stage2 else 'reports/v4_plan.json'),plan)
    prior=np.load(ROOT/'reports/cv_v3_predictions.npz')
    cv={};folds={year:data[data.season.eq(year)] for year in [2023,2024]}
    cached=np.load(ROOT/'reports/cv_v4_predictions.npz') if stage2 else None
    for year,va in folds.items():
        train=data[data.season<year]
        for name in base:cv[(name,year)]=prior[f'{name}_{year}']
        assert all(len(cv[(name,year)])==len(va) for name in base)
        for name in new:
            if cached is not None and f'{name}_{year}' in cached:
                cv[(name,year)]=cached[f'{name}_{year}'];continue
            spec=specs[name];model=make_model(spec['columns'],**spec['params']).fit(train[spec['columns']],train.target)
            cv[(name,year)]=np.maximum(0,model.predict(va[spec['columns']]))
            score=metrics(va.target,cv[(name,year)])
            print(f'CV {year} {name}: MAE {score["mae"]:.5f}, RMSE {score["rmse"]:.5f}',flush=True)
    scores=[]
    for name,weights in blends.items():
        results=[]
        for year,va in folds.items():
            p=sum(weight*cv[(model,year)] for model,weight in weights.items())
            results.append({'validation_season':year,'train_n':int((data.season<year).sum()),**metrics(va.target,p),
                            'tolerance':tolerance(va.target,p)})
        scores.append({'model':name,'weights':weights,'folds':results,
          'cv_mae':float(np.mean([s['mae'] for s in results])),'cv_rmse':float(np.mean([s['rmse'] for s in results]))})
    reference=next(s for s in scores if s['model']=='v3_reference')
    eligible=[s for s in scores if s['cv_mae']<=reference['cv_mae'] and s['cv_rmse']<=reference['cv_rmse']*.9975]
    selected=min(eligible,key=lambda r:r['cv_rmse']) if eligible else reference
    result={'reference':reference,'selected':selected,'candidates':scores,'selection_locked_at_utc':datetime.now(timezone.utc).isoformat(),
            'promoted':bool(eligible),'rule':plan['selection']}
    dump(ROOT/('reports/v4_validation_phase2.json' if stage2 else 'reports/v4_validation.json'),result)
    np.savez_compressed(ROOT/'reports/cv_v4_predictions.npz',**{f'{n}_{year}':p for (n,year),p in cv.items()})
    print('SELECTED',json.dumps(selected),flush=True)
    if not eligible:
        print('V3 retained: no candidate meets both CV requirements.',flush=True);return result
    train=data[data.season<=2024];test=data[data.season>=2025].copy()
    frozen=pd.read_csv(ROOT/'reports/v3/predictions_test.csv')
    assert len(test)==len(frozen) and np.allclose(test.target,frozen.target)
    assert np.array_equal(test[['player_id','season','week']].to_numpy(),frozen[['player_id','season','week']].to_numpy())
    evaluation=FantasyEnsemble(specs,selected['weights']).fit(train[cols],train.target)
    test['prediction']=evaluation.predict(test[cols])
    calibration=[]
    for year,va in folds.items():
        part=va[['position','target']].copy()
        part['prediction']=sum(w*cv[(n,year)] for n,w in selected['weights'].items());calibration.append(part)
    calibration=pd.concat(calibration)
    bands={pos:float(np.quantile(abs(g.target-g.prediction),.8)) for pos,g in calibration.groupby('position')}
    test['band']=test.position.map(bands);test['low']=np.maximum(0,test.prediction-test.band);test['high']=test.prediction+test.band
    report={**old,'version':'v4','created_at':datetime.now(timezone.utc).isoformat(),'selected_model':selected['model'],
      'features':cols,'feature_count':len([c for c in cols if c!='season']),'fit_metadata_columns':['season'] if stage2 else [],'train':metrics(train.target,evaluation.predict(train[cols])),
      'test':metrics(test.target,test.prediction),'cv_candidates':scores,
      'by_season':{str(int(y)):{'model':metrics(g.target,g.prediction),'baseline':metrics(g.target,g.baseline)} for y,g in test.groupby('season')},
      'by_position':{pos:{'model':metrics(g.target,g.prediction),'baseline':metrics(g.target,g.baseline)} for pos,g in test.groupby('position')},
      'interval':{'nominal':.8,'coverage_test':float(((test.target>=test.low)&(test.target<=test.high)).mean()),
                  'description':'Empirical CV residual bands, not individual probability guarantees.','half_width_by_position':bands},
      'tolerance':{'model':tolerance(test.target,test.prediction),'v3':tolerance(frozen.target,frozen.prediction),
                   'baseline':tolerance(test.target,test.baseline)},
      'v4_comparison':{'v3':old['test'],'v4':metrics(test.target,test.prediction),'cv_v3':reference,'cv_v4':selected,
                       'status':'Same already-inspected 2025/26 replay. Selection used 2023/24 only.'}}
    report['version_comparison']={**old['version_comparison'],'v3':old['test'],'new':report['test']}
    report['version_comparison']['rmse_reduction_from_v2_pct']=100*(report['version_comparison']['v2']['rmse']-report['test']['rmse'])/report['version_comparison']['v2']['rmse']
    report['mae_improvement_pct']=100*(report['baseline']['mae']-report['test']['mae'])/report['baseline']['mae']
    report['protocol']={**old['protocol'],'selection':plan['selection'],
                       'test_status':'Previously inspected repeatedly. Retrospective replay, not an untouched independent holdout.'}
    dump(ROOT/'reports/v4_metrics.json',report)
    test.to_csv(ROOT/'reports/v4_predictions_test.csv',index=False)
    # Promotion is determined by locked CV results, never replay performance.
    joblib.dump({'model':evaluation,'name':selected['model'],'columns':cols},ROOT/'models/evaluation_model.joblib')
    test.to_csv(ROOT/'reports/predictions_test.csv',index=False)
    dump(ROOT/'reports/metrics.json',report)
    print('REPLAY',json.dumps({'v3':old['test'],'v4':report['test'],'tolerance':report['tolerance']}),flush=True)
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--stage2',action='store_true')
    run(parser.parse_args().stage2)
