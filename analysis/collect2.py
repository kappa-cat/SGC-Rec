"""Aggregates results/ into analysis/numbers2.json: five-seed means and standard deviations of all methods and ablations, paired t / Wilcoxon tests with Holm correction, calibration diagnostics, training cost and the Amazon-Beauty user-activity groups."""
import json,numpy as np
from pathlib import Path
from scipy import stats
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
R=ROOT/'results'; DS=['beauty','lastfm','ml10m']; SE=range(5)
J=lambda p:json.load(open(p))
M={(d,s):J(R/f'main_{d}_s{s}.json') for d in DS for s in SE}
import os
B={(d,s):J(R/(f'base2_{d}_s{s}.json' if os.path.exists(R/f'base2_{d}_s{s}.json') else f'base_{d}_s{s}.json')) for d in DS for s in SE}
N_base2=all(os.path.exists(R/f'base2_{d}_s{s}.json') for d in DS for s in SE);print('base2 complete:',N_base2)
N={'cfg':{d:J(R/f'cfg_{d}.json') for d in DS}}
var=['xsim','nopre','cat_nopre_only','cat_prior_only','full','full_kappa0','full_nohead','nohead_rx','gauss_affinity_only','initonly','novarpath','v0']
bl=['UserCF','ItemCF','NCF','DeepFM','LightGCN','SGL','SimGCL','LightGCL','HCCF','SEPT']
for d in DS:
    for v in var:
        for m in 'PRN':
            xs=[M[d,s][v][m] for s in SE]; N[f'{d}.{v}.{m}']=(float(np.mean(xs)),float(np.std(xs,ddof=1)),xs)
        N[f'{d}.{v}.E']=float(np.mean([M[d,s][v]['entropy'] for s in SE]))
    for v in bl:
        for m in 'PRN':
            xs=[B[d,s][v][m] for s in SE]; N[f'{d}.{v}.{m}']=(float(np.mean(xs)),float(np.std(xs,ddof=1)),xs)
        N[f'{d}.{v}.E']=float(np.mean([B[d,s][v]['entropy'] for s in SE]))
        N[f'{d}.{v}.time']=float(np.mean([B[d,s][v]['train_time'] for s in SE])); N[f'{d}.{v}.mb']=float(np.mean([B[d,s][v]['peak_mb'] for s in SE]))
    N[f'{d}.prior_epochs']=[M[d,s]['prior_epochs_selected'] for s in SE]
    N[f'{d}.spearman']=float(np.mean([M[d,s]['calib']['spearman'] for s in SE])); N[f'{d}.M2']=float(np.mean([M[d,s]['calib']['M2'] for s in SE]))
    tm=[M[d,s]['time'] for s in SE]
    N[f'{d}.time']={k:float(np.mean([t[k] for t in tm])) for k in tm[0]}
    N[f'{d}.xsim_epochs']=float(np.mean([M[d,s]['xsim_epochs'] for s in SE]))
def tests(a,b):
    out={};ps=[]
    for d in DS:
        for m in 'PRN':
            x=np.array(N[f'{d}.{a}.{m}'][2]);y=np.array(N[f'{d}.{b}.{m}'][2])
            p=stats.ttest_rel(x,y).pvalue;w=stats.wilcoxon(x,y).pvalue
            out[f'{d}.{m}']=dict(gain=(x.mean()/y.mean()-1)*100,p=p,w=w,pos=int((x>y).sum())); ps.append((f'{d}.{m}',p))
    ps.sort(key=lambda t:t[1]);run=0
    for i,(k,p) in enumerate(ps):
        run=max(run,min(1,(len(ps)-i)*p)); out[k]['holm']=run
    return out
N['tests']={f'{a}_vs_{b}':tests(a,b) for a,b in [('full','xsim'),('nopre','xsim'),('full','nopre'),('cat_prior_only','cat_nopre_only'),('initonly','nopre'),('full','v0'),('full','novarpath'),('full','full_kappa0')]}
# strongest baseline per column (excluding xsim / sgc-rec)
# sparsity by user degree (Amazon-Beauty)
groups=[(0,10),(10,20),(20,50),(50,10**9)]
z={s:np.load(ROOT/'data'/'beauty.npz',allow_pickle=True) for s in [0]}[0]
acc={g:{k:[] for k in ['xR','xP','xN','fR','fP','fN','n']} for g in range(4)}
for s in SE:
    m=np.load(R/f'main_beauty_s{s}.npz'); nu=int(z['nu'])
    u=z[f'u_{s}'];tag=z[f'tag_{s}'];deg=np.bincount(u[tag==0],minlength=nu)
    dg=deg[m['users']]
    for gi,(a,b) in enumerate(groups):
        mk=(dg>=a)&(dg<b)
        for q in 'RPN': acc[gi]['x'+q].append(m['xsim_'+q][mk].mean()); acc[gi]['f'+q].append(m['full_'+q][mk].mean())
        acc[gi]['n'].append(int(mk.sum()))
N['sparsity']={str(g):{k:float(np.mean(v)) for k,v in acc[g].items()} for g in acc}
N['sens']={d:J(R/f'sens_{d}.json') for d in DS}
json.dump(N,open(HERE/'numbers2.json','w'),indent=1,default=float)
pct=lambda a,b:(a/b-1)*100
for d in DS:
    print(d,{v:round(N[f'{d}.{v}.R'][0],4) for v in ['xsim','nopre','full']+bl})
print({k:{kk:round(vv['gain'],2) for kk,vv in t.items() if kk.endswith('.R')} for k,t in N['tests'].items()})
for d in DS: print(d,N[f'{d}.time'],N[f'{d}.xsim_epochs'])
print(N['sparsity'])
