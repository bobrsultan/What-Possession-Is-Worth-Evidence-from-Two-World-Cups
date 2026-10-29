import pandas as pd,numpy as np,statsmodels.api as sm
from scipy.stats import f
from sklearn.model_selection import KFold,cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
p='data/'
a=pd.read_csv(p+'underdog_2026_match_data_elo50.csv'); raw=pd.read_csv(p+'team_key_stats.csv'); raw=raw[raw.match_team_id.notna()].copy();raw['v']=pd.to_numeric(raw.value,errors='coerce'); w=raw.pivot_table(index='match_team_id',columns='metric',values='v',aggfunc='first'); teams=pd.read_csv(p+'match_teams.csv'); teams=teams.join(w,on='match_team_id');h=teams[teams.home_away=='home'].set_index('match_id');aw=teams[teams.home_away=='away'].set_index('match_id');
for met,col in [('xG (Expected Goals)','xg'),('Completed Line Breaks','linebreak'),('Receptions in the Final Third','finalthird'),('Forced Turnovers','turnovers'),('Attempts at Goal (On Target)','shots'),('Total Passes (Complete)','passes')]:
 if met in h:
  gap=h[met]-aw[met]; a[col+'_gap']=np.where(a.underdog_home,a.match_id.map(gap),-a.match_id.map(gap))
# goals, xg, shots, linebreak, turnovers, final third; directness as explanatory predictor for all except shots (mechanical overlap)
outputs=['underdog_goal_diff','xg_gap','linebreak_gap','finalthird_gap','turnovers_gap']
rows=[];kf=KFold(10,shuffle=True,random_state=42)
for y in outputs:
 for base_name,base in [('Elo only',['elo_gap_abs']),('Elo + directness',['elo_gap_abs','underdog_directness_gap'])]:
  dat=a[[y,'underdog_possession_gap']+base].dropna(); Y=dat[y]; x0=sm.add_constant(dat[base]); x1=sm.add_constant(dat[base+['underdog_possession_gap']]); fit0=sm.OLS(Y,x0).fit();fit1=sm.OLS(Y,x1).fit(cov_type='HC3');fit1classic=sm.OLS(Y,x1).fit();ff,pv,df=fit1classic.compare_f_test(fit0)
  pred0=cross_val_predict(LinearRegression(),dat[base],Y,cv=kf);pred1=cross_val_predict(LinearRegression(),dat[base+['underdog_possession_gap']],Y,cv=kf)
  rows.append(dict(outcome=y,baseline=base_name,n=len(dat),R2_without=fit0.rsquared,R2_with=fit1.rsquared,delta_R2=fit1.rsquared-fit0.rsquared,partial_F_p=pv,possession_coefficient=fit1.params['underdog_possession_gap'],possession_HC3_p=fit1.pvalues['underdog_possession_gap'],RMSE_without=np.sqrt(mean_squared_error(Y,pred0)),RMSE_with=np.sqrt(mean_squared_error(Y,pred1)),RMSE_change=np.sqrt(mean_squared_error(Y,pred1))-np.sqrt(mean_squared_error(Y,pred0))))
res=pd.DataFrame(rows);res.to_csv(p+'possession_ablation_2026.csv',index=False);a.to_csv(p+'underdog_2026_ablation_enriched.csv',index=False);print(res.to_string(index=False,float_format=lambda z:f'{z:.4f}'))
