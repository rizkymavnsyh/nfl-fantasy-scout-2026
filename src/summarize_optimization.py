"""Publish the experiment outcome, including rejected candidates and fixed tolerances."""
import json,hashlib
import pandas as pd
import numpy as np
import joblib
from pipeline import ROOT,dump
from improve_v4 import tolerance

def summarize():
    for folder in (ROOT/'reports/frozen').iterdir():
        if not folder.is_dir():continue
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        for name,key in [('forecasts.csv','forecast_sha256'),('lineups.json','lineups_sha256')]:
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==manifest[key]
    validation=json.loads((ROOT/'reports/v4_validation_phase2.json').read_text(encoding='utf-8'))
    report=json.loads((ROOT/'reports/metrics.json').read_text(encoding='utf-8'))
    prediction=pd.read_csv(ROOT/'reports/predictions_test.csv')
    old=pd.read_csv(ROOT/'reports/v3/predictions_test.csv')
    if not validation['promoted']:
        assert np.array_equal(prediction.prediction,old.prediction)
        assert np.array_equal(prediction.target,old.target)
    bundle=joblib.load(ROOT/'models/evaluation_model.joblib')
    sample=prediction.sample(50,random_state=42)
    assert np.allclose(bundle['model'].predict(sample[bundle['columns']]),sample.prediction,atol=1e-10)
    report['tolerance']={'model':tolerance(prediction.target,prediction.prediction),
                         'baseline':tolerance(prediction.target,prediction.baseline),
                         'by_position':{p:{'model':tolerance(g.target,g.prediction),
                                           'baseline':tolerance(g.target,g.baseline)} for p,g in prediction.groupby('position')}}
    ref=validation['reference'];best=min(validation['candidates'],key=lambda r:r['cv_rmse'])
    summary={'promoted':validation['promoted'],'retained_version':report['version'],
      'candidate_count':len(validation['candidates']),'reference_cv':{k:ref[k] for k in ['cv_mae','cv_rmse']},
      'lowest_rmse_candidate':{k:best[k] for k in ['model','cv_mae','cv_rmse']},'selection_rule':validation['rule'],
      'approaches':['12 new features from earlier team/position opportunity data',
                    'Position-specific regression','Nonzero-probability times conditional points',
                    'Recency sample weights with 0.65 and 0.85 per-season decay','Fixed blends with V3'],
      'status':'No candidate met both validation requirements. V3 predictions and locked snapshots retained.' if not validation['promoted'] else 'Promoted using historical validation only.',
      'data_boundary':'2023/24 development folds only. Failed candidates were not replayed on 2025/26 for selection.',
      'tolerance':report['tolerance'],'frozen_snapshots_unchanged':True,
      'test_prediction_sha256':hashlib.sha256((ROOT/'reports/predictions_test.csv').read_bytes()).hexdigest()}
    dump(ROOT/'reports/optimization_summary.json',summary)
    dump(ROOT/'reports/metrics.json',report);dump(ROOT/'dist/metrics.json',report)
    payload=json.loads((ROOT/'dist/data.json').read_text(encoding='utf-8'))
    payload['metrics']=report;payload['optimization_summary']=summary;dump(ROOT/'dist/data.json',payload)
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':summarize()
