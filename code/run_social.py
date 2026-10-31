"""Last-FM friendship ablation: python run_social.py seeds(csv). Test-split analysis, configuration fixed from validation.
nomix    : friends are not mixed into the catalogue expert (social mixing off); the friend count may still enter the prior features
nofriend : no friendship information anywhere (no mixing, zero social-degree feature)"""
import sys,json,numpy as np
from sgc2 import *
ds='lastfm';seeds=[int(x) for x in sys.argv[1].split(',')];cfg=json.load(open(HERE/'results'/f'cfg_{ds}.json'));kap=cfg['kappa']
for seed in seeds:
    f=HERE/'results'/f'soc_{ds}_s{seed}.json'
    if f.exists(): continue
    d=Data(ds,seed);out=dict(dataset=ds,seed=seed)
    def ev(name,dd,scorer):
        out[name]=summ(evaluate(dd,scorer,'test'),dd);print(ds,seed,name,{k:round(v,4) for k,v in out[name].items()},flush=True)
    Z,_=get_xsim(d,ds,seed);ex=Ev(Z,d.nu)
    b0=build_nopre(d,ds,seed,cfg,social=False);h0,sc0,_=fit_head(d,b0['er'],ex,seed);ev('nopre_nomix',d,sc0(kap));ev('cat_nopre_nomix',d,b0['er'].raw)
    fb=build_full(d,ds,seed,cfg,social=False);h,sc,_=fit_head(d,fb['er'],ex,seed,unc=fb['pr']['unc'],gev=fb['gev'],free=cfg['free']);ev('full_nomix',d,sc(kap));ev('cat_prior_nomix',d,fb['er'].raw)
    d2=Data(ds,seed);d2.drop_social()
    fb=build_full(d2,ds,seed,cfg,social=False);h,sc,_=fit_head(d2,fb['er'],ex,seed,unc=fb['pr']['unc'],gev=fb['gev'],free=cfg['free']);ev('full_nofriend',d2,sc(kap));ev('cat_prior_nofriend',d2,fb['er'].raw)
    json.dump(out,open(f,'w'),indent=1)
