import numpy as np,pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
D=pd.read_csv('data/underdog_2026_ablation_enriched.csv')
print(D[['underdog_possession_gap','elo_gap_abs','underdog_home','underdog_goal_diff','xg_gap','linebreak_gap','finalthird_gap']].describe().round(2))
print(D[['underdog_home','underdog_possession_gap']].head())
print('missing',D.isna().sum()[D.isna().sum()>0])
T=D.underdog_possession_gap.to_numpy(float)
base=['elo_gap_abs','underdog_home']
models={'ridge':lambda:make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13))), 'forest':lambda:RandomForestRegressor(n_estimators=45,min_samples_leaf=8,max_features=1.0,random_state=43,n_jobs=1)}
rows=[]
for name,cols in [('prematch',base),('plus_territory',base+['linebreak_gap','finalthird_gap']),('plus_territory_directness',base+['linebreak_gap','finalthird_gap','underdog_directness_gap'])]:
 X=D[cols].astype(float).to_numpy()
 for outcome in ['linebreak_gap','finalthird_gap','xg_gap','underdog_goal_diff']:
  if outcome in cols:continue
  Y=D[outcome].to_numpy(float)
  for modelname,make in models.items():
   for seed in [11,37,71]:
    pt=np.zeros(len(D));py=np.zeros(len(D))
    for train,test in KFold(3,shuffle=True,random_state=seed).split(X):
     tmodel=make();ymodel=make();tmodel.fit(X[train],T[train]);ymodel.fit(X[train],Y[train]);pt[test]=tmodel.predict(X[test]);py[test]=ymodel.predict(X[test])
    rt=T-pt;ry=Y-py
    fit=sm.OLS(ry,sm.add_constant(rt)).fit(cov_type='HC3')
    rows.append({'controls':name,'outcome':outcome,'learner':modelname,'seed':seed,'effect_per_1pp':fit.params[1],'effect_per_10pp':fit.params[1]*10,'p_value_HC3':fit.pvalues[1],'ci_low_10pp':fit.conf_int()[1,0]*10,'ci_high_10pp':fit.conf_int()[1,1]*10,'treatment_residual_sd':rt.std(),'outcome_residual_sd':ry.std(),'n':len(D)})
r=pd.DataFrame(rows);r.to_csv('data/dml_possession_2026_all_runs.csv',index=False)
s=r.groupby(['controls','outcome','learner']).agg(effect_10pp_median=('effect_per_10pp','median'),effect_10pp_min=('effect_per_10pp','min'),effect_10pp_max=('effect_per_10pp','max'),p_median=('p_value_HC3','median'),p_min=('p_value_HC3','min'),p_max=('p_value_HC3','max'),ci_low_median=('ci_low_10pp','median'),ci_high_median=('ci_high_10pp','median')).reset_index()
s.to_csv('data/dml_possession_2026_summary.csv',index=False)
print(s.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
