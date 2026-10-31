"""Validation-only search over the strength of the prior anchor (MAP regularisation of the catalogue expert
towards the pre-trained mean). Val seeds 0-2. Writes results/design2_<ds>.json and overwrites results/cfg_<ds>.json."""
import sys,json,numpy as np,torch
from sgc2 import *
KAP=[0,.02,.05,.10,.15,.25,.35,.5]
V=[('init',dict(anchor=0.,anchor_deg=False)),('u.05',dict(anchor=.05,anchor_deg=False)),('u.5',dict(anchor=.5,anchor_deg=False)),
   ('d.5',dict(anchor=.5,anchor_deg=True)),('d2',dict(anchor=2.,anchor_deg=True)),('d8',dict(anchor=8.,anchor_deg=True))]
for ds in sys.argv[1:]:
    rows={};seeds=[0,1,2]
    def add(name,seed,sc):
        rows.setdefault(name,[]).append([float(evaluate(d,sc(k),'val')['R'].mean()) for k in KAP]);print(ds,seed,name,np.round(rows[name][-1],4).tolist(),flush=True)
    for seed in seeds:
        d=Data(ds,seed);Z,_=get_xsim(d,ds,seed);ex=Ev(Z,d.nu)
        b=build_nopre(d,ds,seed,{});h,sc,_=fit_head(d,b['er'],ex,seed);add('nopre',seed,sc)
        pr=fit_prior(d,seed,social_weight=0.);gev=EnergyEv(pr,d.nu)
        for name,o in V:
            er=Ev(train_cat(d,ds,seed,4,pr['mu'],.75,None,**o),d.nu)
            h,sc,_=fit_head(d,er,ex,seed,unc=pr['unc'],gev=gev,free=False);add(name,seed,sc)
    mean={k:np.mean(v,0) for k,v in rows.items()};best=None
    for k,_ in V:
        j=int(np.argmax(mean[k]));val=mean[k][j]
        if best is None or val>best[0]+1e-9: best=(val,k,KAP[j])
    _,name,kap=best;o=dict(V)[name]
    cfg=dict(variant=name,alpha=.75,pscale=None,free=False,kappa=kap,prior_epochs=8,cat_epochs=4,social_weight=0.,anchor=o['anchor'],anchor_deg=o['anchor_deg'])
    json.dump(dict(kappas=KAP,rows=rows,selected=cfg),open(HERE/'results'/f'design2_{ds}.json','w'),indent=1);json.dump(cfg,open(HERE/'results'/f'cfg_{ds}.json','w'),indent=1);print('SELECTED',ds,cfg,flush=True)
