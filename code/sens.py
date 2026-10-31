"""validation sensitivity: kappa (val seeds 0-2) and catalogue horizon (val seed 0)"""
import sys,json
from sgc2 import *
KAP=[0,.02,.05,.10,.15,.25,.35,.5]
for ds in sys.argv[1:]:
    cfg=json.load(open(HERE/'results'/f'cfg_{ds}.json'));out=dict(kappas=KAP,rows=[],horizon=[])
    for seed in (0,1,2):
        d=Data(ds,seed);Z,_=get_xsim(d,ds,seed);ex=Ev(Z,d.nu)
        b0=build_nopre(d,ds,seed,cfg);h0,sc0,_=fit_head(d,b0['er'],ex,seed);fb=build_full(d,ds,seed,cfg);h,sc,_=fit_head(d,fb['er'],ex,seed,unc=fb['pr']['unc'],gev=fb['gev'],free=cfg['free'])
        out['rows'].append(dict(seed=seed,nopre=[float(evaluate(d,sc0(k),'val')['R'].mean()) for k in KAP],full=[float(evaluate(d,sc(k),'val')['R'].mean()) for k in KAP]));print(ds,seed,out['rows'][-1],flush=True)
        if seed==0:
            for ep in (2,4,6):
                b0=build_nopre(d,ds,0,cfg,ep);h0,sc0,_=fit_head(d,b0['er'],ex,0);Zf=train_cat(d,ds,0,ep,fb['pr']['mu'],cfg['alpha'],cfg['pscale'],anchor=cfg.get('anchor',0.),anchor_deg=cfg.get('anchor_deg',False));erf=Ev(Zf,d.nu);hf,scf,_=fit_head(d,erf,ex,0,unc=fb['pr']['unc'],gev=fb['gev'],free=cfg['free'])
                out['horizon'].append(dict(epochs=ep,full=float(evaluate(d,scf(cfg['kappa']),'val')['R'].mean()),nopre=float(evaluate(d,sc0(cfg['kappa']),'val')['R'].mean())));print(out['horizon'][-1],flush=True)
    json.dump(out,open(HERE/'results'/f'sens_{ds}.json','w'),indent=1)
