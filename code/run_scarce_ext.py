"""scarce-data regime with re-tuned fusion head: python run_scarce_ext.py ds seeds(csv) keep(percent)
For every head recipe both the validation score (cross-fitted for recipes fitted on validation pairs) and the test score are logged; the recipe is chosen from validation scores of seeds 0-2 only."""
import sys,json,numpy as np
from sgc3 import *
ds=sys.argv[1];seeds=[int(x) for x in sys.argv[2].split(',')];kp=int(sys.argv[3]);cfg=json.load(open(HERE/'results'/f'cfg_{ds}.json'));kap=cfg['kappa']
for seed in seeds:
    f=HERE/'results'/f'scx_{ds}_s{seed}_k{kp}.json'
    if f.exists(): continue
    d=Data(ds,seed,keep=kp/100);out=dict(dataset=ds,seed=seed,keep=kp);arr={}
    Z,_=get_xsim(d,ds,seed,0.,f'_k{kp}');ex=Ev(Z,d.nu)
    def rec(name,sv,st):
        ev=evaluate(d,st,'test');out[name]=dict(val=float(evaluate(d,sv,'val')['R'].mean()) if sv is not None else None,**summ(ev,d))
        arr.update({name+'_R':ev['R'],name+'_N':ev['N'],'users':ev['users']})
    rec('xsim',ex.raw,ex.raw)
    b0=build_nopre(d,ds,seed,cfg);rec('cat_nopre_only',b0['er'].raw,b0['er'].raw)
    for r,(sv,st) in recipe_scorers(d,b0['er'],ex,seed,None,None,cfg['free'],kap).items(): rec('nopre_'+r,sv,st)
    fb=build_full(d,ds,seed,cfg);rec('cat_prior_only',fb['er'].raw,fb['er'].raw)
    for r,(sv,st) in recipe_scorers(d,fb['er'],ex,seed,fb['pr']['unc'],fb['gev'],cfg['free'],kap).items(): rec('full_'+r,sv,st)
    print(ds,seed,kp,{k:round(v['R'],4) for k,v in out.items() if isinstance(v,dict)},flush=True)
    np.savez_compressed(HERE/'results'/f'scx_{ds}_s{seed}_k{kp}.npz',**arr);json.dump(out,open(f,'w'),indent=1)
