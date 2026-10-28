"""Serializable forecasting ensemble, chosen using historical validation only."""
import sys
from pathlib import Path
if sys.platform=='win32':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor'))
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def make_model(columns, loss='squared_error', leaves=20, iterations=180, minimum=35, learning_rate=.045,
               family='hist', reg_lambda=25., alpha=.9, season_decay=.8):
    if family=='recency':
        return RecencyRegressor(columns,season_decay=season_decay,loss=loss,leaves=leaves,
          iterations=iterations,minimum=minimum,learning_rate=learning_rate,reg_lambda=reg_lambda,alpha=alpha)
    if family in ('position','hurdle'):
        cls=PositionRegressor if family=='position' else HurdleRegressor
        return cls(columns,loss=loss,leaves=leaves,iterations=iterations,minimum=minimum,
                   learning_rate=learning_rate,reg_lambda=reg_lambda,alpha=alpha)
    numeric=[c for c in columns if c!='position']
    prep=ColumnTransformer([
        ('numbers',Pipeline([('fill',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),numeric),
        ('position',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['position'])])
    if family=='lightgbm':
        from lightgbm import LGBMRegressor
        model=LGBMRegressor(objective=loss,n_estimators=iterations,num_leaves=leaves,
          min_child_samples=minimum,learning_rate=learning_rate,reg_lambda=reg_lambda,
          colsample_bytree=.9,random_state=42,n_jobs=1,verbosity=-1,deterministic=True,
          force_col_wise=True,alpha=alpha)
        return Pipeline([('prepare',prep),('model',model)])
    return Pipeline([('prepare',prep),('model',HistGradientBoostingRegressor(
        loss=loss,max_iter=iterations,max_leaf_nodes=leaves,min_samples_leaf=minimum,
        learning_rate=learning_rate,l2_regularization=25.,early_stopping=False,random_state=42))])

class PositionRegressor:
    """One regularized regression model per position, using the same pre-game inputs."""
    def __init__(self,columns,**params):self.columns=columns;self.params=params;self.models={}
    def fit(self,X,y):
        y=np.asarray(y)
        for pos in sorted(X.position.unique()):
            mask=X.position.eq(pos).to_numpy()
            self.models[pos]=make_model(self.columns,family='lightgbm',**self.params).fit(X.loc[mask,self.columns],y[mask])
        return self
    def predict(self,X):
        out=np.zeros(len(X))
        for pos,model in self.models.items():
            mask=X.position.eq(pos).to_numpy()
            if mask.any():out[mask]=model.predict(X.loc[mask,self.columns])
        return out

class HurdleRegressor:
    """P(nonzero PPR) times expected PPR conditional on a nonzero result."""
    def __init__(self,columns,**params):self.columns=columns;self.params=params
    def fit(self,X,y):
        from lightgbm import LGBMClassifier
        y=np.asarray(y);has_points=y!=0
        classifier=make_model(self.columns,family='lightgbm',**self.params)
        classifier.steps[-1]=('model',LGBMClassifier(n_estimators=350,num_leaves=7,min_child_samples=70,
          learning_rate=.035,reg_lambda=35.,n_jobs=1,random_state=42,verbosity=-1,deterministic=True,force_col_wise=True))
        self.classifier=classifier.fit(X[self.columns],has_points.astype(int))
        self.regressor=make_model(self.columns,family='lightgbm',**self.params).fit(X.loc[has_points,self.columns],y[has_points])
        return self
    def predict(self,X):
        return self.classifier.predict_proba(X[self.columns])[:,1]*self.regressor.predict(X[self.columns])

class RecencyRegressor:
    """Season is fit-weight metadata only; it is excluded from predictor columns."""
    def __init__(self,columns,season_decay=.8,**params):
        self.columns=[c for c in columns if c!='season'];self.season_decay=season_decay;self.params=params
    def fit(self,X,y):
        weights=self.season_decay**(X.season.max()-X.season.to_numpy())
        self.model=make_model(self.columns,family='lightgbm',**self.params)
        self.model.fit(X[self.columns],y,model__sample_weight=weights)
        return self
    def predict(self,X):return self.model.predict(X[self.columns])

class FantasyEnsemble:
    def __init__(self, specifications, weights):
        self.specifications=specifications
        self.weights=weights
        self.models={}
    def fit(self,X,y):
        for name,weight in self.weights.items():
            if weight<=0:continue
            spec=self.specifications[name]
            cols=spec['columns']
            model=make_model(cols,**spec['params'])
            model.fit(X[cols],y)
            self.models[name]=model
        return self
    def predict(self,X):
        total=np.zeros(len(X))
        for name,model in self.models.items():
            total+=self.weights[name]*np.maximum(0,model.predict(X[self.specifications[name]['columns']]))
        return total
