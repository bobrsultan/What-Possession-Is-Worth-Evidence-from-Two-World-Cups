import pandas as pd, numpy as np
import statsmodels.formula.api as smf
p='data/'
ratings={'Algeria':1760,'Argentina':2115,'Australia':1777,'Austria':1830,'Belgium':1893,'Bosnia and Herzegovina':1595,'Brazil':1991,'Cabo Verde':1578,'Canada':1788,'Colombia':1982,'Congo DR':1652,'Croatia':1911,'Curaçao':1434,'Czechia':1740,"Côte d'Ivoire":1696,'Ecuador':1938,'Egypt':1696,'England':2021,'France':2063,'Germany':1932,'Ghana':1511,'Haiti':1548,'IR Iran':1772,'Iraq':1607,'Japan':1906,'Jordan':1680,'Korea Republic':1758,'Mexico':1875,'Morocco':1827,'Netherlands':1948,'New Zealand':1562,'Norway':1914,'Panama':1730,'Paraguay':1833,'Portugal':1986,'Qatar':1421,'Saudi Arabia':1576,'Scotland':1782,'Senegal':1860,'South Africa':1518,'Spain':2157,'Sweden':1712,'Switzerland':1891,'Tunisia':1628,'Türkiye':1910,'USA':1726,'Uruguay':1892,'Uzbekistan':1714}
m=pd.read_csv(p+'possession_directness_2026_match_data.csv')
# Validate team columns
print(m.columns.tolist())
m['home_elo']=m.home_team.map(ratings);m['away_elo']=m.away_team.map(ratings)
assert m[['home_elo','away_elo']].notna().all().all()
m['elo_gap_abs']=(m.home_elo-m.away_elo).abs()
m['underdog_home']=m.home_elo<m.away_elo
m['underdog_goal_diff']=np.where(m.underdog_home,m.goal_difference,-m.goal_difference)
m['underdog_possession_gap']=np.where(m.underdog_home,m.possession_gap,-m.possession_gap)
m['underdog_directness_gap']=np.where(m.underdog_home,m.directness_gap,-m.directness_gap)
m['underdog_win']=(m.underdog_goal_diff>0).astype(int)
for cutoff in [0,50,100]:
 d=m[m.elo_gap_abs>=cutoff].copy()
 for col in ['underdog_possession_gap','underdog_directness_gap','elo_gap_abs']:
  d[col+'_c']=d[col]-d[col].mean()
 f=smf.ols('underdog_goal_diff ~ underdog_possession_gap_c * underdog_directness_gap_c + elo_gap_abs_c',data=d).fit(cov_type='HC3')
 print('\nCUTOFF',cutoff,'n',len(d),'wins',d.underdog_win.sum())
 print(pd.DataFrame({'beta':f.params,'p':f.pvalues,'lo':f.conf_int()[0],'hi':f.conf_int()[1]}).round(5).to_string());print('r2',f.rsquared)
 if cutoff==50:
  d.to_csv(p+'underdog_2026_match_data_elo50.csv',index=False)
  pd.DataFrame({'term':f.params.index,'coefficient':f.params.values,'p_value':f.pvalues.values,'ci_lower':f.conf_int()[0].values,'ci_upper':f.conf_int()[1].values}).to_csv(p+'underdog_2026_interaction_results.csv',index=False)
  print('2026 upset avg',d.groupby('underdog_win')[['underdog_possession_gap','underdog_directness_gap']].mean())
pd.DataFrame([{'team':k,'elo_2026_06_09':v} for k,v in ratings.items()]).to_csv(p+'pre_tournament_elo_2026.csv',index=False)
