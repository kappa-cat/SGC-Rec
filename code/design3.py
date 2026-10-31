"""Second validation-only search of the prior pathway: alpha in {.5,.75,1} x anchor variants, selected by mean validation Recall of the FULL model
over seeds 0-2 (val split only; the test split is never touched). python design3.py ds ... ; writes results/design3_<ds>.json and overwrites results/cfg_<ds>.json."""
import sys,json,numpy as np,torch
from sgc2 import *
KAP=[0,.02,.05,.10,.15,.25,.35,.5]
A=[('init',dict(anchor=0.,anchor_deg=False))]
for nm,o in [('u.5',dict(anchor=.5,anchor_deg=False)),('u2',dict(anchor=2.,anchor_deg=False)),('d.5',dict(anchor=.5,anchor_deg=True)),('d2',dict(anchor=2.,anchor_deg=True)),('d8',dict(anchor=8.,anchor_deg=True)),('d32',dict(anchor=32.,anchor_deg=True))]: A.append((nm,o))
ALPHAS=[.5,.75,1.]
V=[(f'{n}_a{a}',a,o) for a in ALPHAS for n,o in A if not(n=='init' and a!=.75)]
for ds in sys.argv[1:]:
    rows={};seeds=[0,1,2]
    for seed in seeds:
        d=Data(ds,seed);Z,_=get_xsim(d,ds,seed);ex=Ev(Z,d.nu)
        pr=fit_prior(d,seed,social_weight=0.);gev=EnergyEv(pr,d.nu)
        for name,a,o in V:
            er=Ev(train_cat(d,ds,seed,4,pr['mu'],a,None,**o),d.nu)
            h,sc,_=fit_head(d,er,ex,seed,unc=pr['unc'],gev=gev,free=False)
            rows.setdefault(name,[]).append([float(evaluate(d,sc(k),'val')['R'].mean()) for k in KAP]);print(ds,seed,name,np.round(rows[name][-1],4).tolist(),flush=True)
    mean={k:np.mean(v,0) for k,v in rows.items()};best=None
    for k,a,o in V:
        j=int(np.argmax(mean[k]));val=mean[k][j]
        if best is None or val>best[0]+1e-9: best=(val,k,KAP[j],a,o)
    _,name,kap,a,o=best
    cfg=dict(variant=name,alpha=a,pscale=None,free=False,kappa=kap,prior_epochs=8,cat_epochs=4,social_weight=0.,anchor=o['anchor'],anchor_deg=o['anchor_deg'])
    json.dump(dict(kappas=KAP,rows=rows,selected=cfg),open(HERE/'results'/f'design3_{ds}.json','w'),indent=1);json.dump(cfg,open(HERE/'results'/f'cfg_{ds}.json','w'),indent=1);print('SELECTED',ds,cfg,flush=True)
