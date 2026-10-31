"""Aggregate validation-selected head recipes (A-F) for scarce (scx_) and noise (rcx_) regimes. usage: python agg_ext.py [RESULTS_DIR] [OUT.json]  (defaults: ../results, analysis/agg_ext.json)"""
import json,sys,glob,numpy as np
from scipy import stats
R=sys.argv[1] if len(sys.argv)>1 else str(__import__("pathlib").Path(__file__).resolve().parents[1]/"results");OUT=sys.argv[2] if len(sys.argv)>2 else str(__import__("pathlib").Path(__file__).resolve().parent/"agg_ext.json");out={}
RECS='ABCDEF'
def agg(prefix,tag,ds,key):
    fs=sorted(glob.glob(f'{R}/{prefix}_{ds}_s*_{tag}.json'));D=[json.load(open(f)) for f in fs]
    if len(D)<3: return None
    res=dict(n=len(D))
    for var in ('nopre','full'):
        r=max(RECS,key=lambda r:np.mean([d[f'{var}_{r}']['val'] for d in D]));res[var+'_recipe']=r
        for m in 'PRN': res[f'{var}_{m}']=float(np.mean([d[f'{var}_{r}'][m] for d in D]))
        res[var+'_R_seeds']=[d[f'{var}_{r}']['R'] for d in D]
    for m in 'PRN': res[f'xsim_{m}']=float(np.mean([d['xsim'][m] for d in D]))
    res['xsim_R_seeds']=[d['xsim']['R'] for d in D]
    for var in ('nopre','full'):
        for m in 'PRN': res[f'{var}_{m}_gain%']=100*(res[f'{var}_{m}']/res[f'xsim_{m}']-1)
        a=np.array(res[var+'_R_seeds']);b=np.array(res['xsim_R_seeds']);res[var+'_p_R']=float(stats.ttest_rel(a,b).pvalue) if len(a)>2 else None
    return res
for ds in ('beauty','lastfm','ml10m'):
    for k in (20,40):
        r=agg('scx',f'k{k}',ds,'');
        if r: out[f'scarce_{ds}_{k}']=r
    for n in (10,20,30):
        r=agg('rcx',f'n{n}',ds,'')
        if r: out[f'noise_{ds}_{n}']=r
json.dump(out,open(OUT,"w"),indent=1)
for k,v in out.items(): print(k,'full',v['full_recipe'],{m:round(v[f'full_{m}_gain%'],1) for m in 'PRN'},'| nopre',v['nopre_recipe'],{m:round(v[f'nopre_{m}_gain%'],1) for m in 'PRN'},'p_full',round(v['full_p_R'],3))
