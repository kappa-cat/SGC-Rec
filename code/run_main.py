"""test-split SGC-Rec and ablations: python run_main.py ds seeds(csv)"""
import sys,json,numpy as np
from sgc2 import *
ds=sys.argv[1];seeds=[int(x) for x in sys.argv[2].split(',')];cfg=json.load(open(HERE/'results'/f'cfg_{ds}.json'));kap=cfg['kappa']
for seed in seeds:
    f=HERE/'results'/f'main_{ds}_s{seed}.json'
    if f.exists() and json.load(open(f)).get('cfg')==cfg and 'nohead_rx' in json.load(open(f)) and 'cat_prior_only_R' in np.load(f.with_suffix('.npz')).files: continue
    d=Data(ds,seed);out=dict(dataset=ds,seed=seed,kappa=kap,cfg=cfg);arr={}
    def ev(name,scorer,keep=False):
        e=evaluate(d,scorer,'test');out[name]=summ(e,d);print(ds,seed,name,{k:round(v,4) for k,v in out[name].items()},flush=True)
        if keep: arr.update({name+'_R':e['R'],name+'_P':e['P'],name+'_N':e['N'],'users':e['users']})
    Z,xi=get_xsim(d,ds,seed);ex=Ev(Z,d.nu);ev('xsim',ex.raw,True);out['t_xsim']=xi['train_time'];out['mem_xsim']=xi['peak_mb'];out['xsim_epochs']=xi['epochs']
    b0=build_nopre(d,ds,seed,cfg);h0,sc0,th0=fit_head(d,b0['er'],ex,seed);ev('nopre',sc0(kap),True);ev('cat_nopre_only',b0['er'].raw,True)
    fb=build_full(d,ds,seed,cfg);pr=fb['pr'];er=fb['er'];gev=fb['gev']
    with Stage() as sh: h,sc,th=fit_head(d,er,ex,seed,unc=pr['unc'],gev=gev,free=cfg['free'])
    ev('full',sc(kap),True);ev('full_kappa0',sc(0.))
    ev('full_nohead',lambda u:(er.full(u)+ex.full(u)+gev.full(u))/3);ev('nohead_rx',lambda u:(er.full(u)+ex.full(u))/2);ev('cat_prior_only',er.raw,True);ev('gauss_affinity_only',gev.full)
    out['calib']=calib(d,h,er,ex,gev,pr['unc'],seed);out['head']=dict(ps=float(h.ps),var=h.var.tolist(),gate=h.gate.tolist(),ls=h.ls.tolist())
    out['prior_epochs_selected']=pr['epochs']
    hi,sci,_=fit_head(d,er,ex,seed,free=cfg['free']);ev('initonly',sci(kap))
    hn,scn,_=fit_head(d,er,ex,seed,unc=None,gev=gev,free=cfg['free']);ev('novarpath',scn(kap))
    # linear-head prior (simpler pre-training design used as a reference)
    mu,unc0,_,meta=fit_prior_legacy(d,components=2,epochs=8,seed=seed,beta_kl=1e-3,objective='distributional',social_weight=0.)  # linear-head prior: 8 epochs on every dataset, no friendship term
    Zl=train_cat(d,ds,seed,4,mu,.75,None);erl=Ev(Zl,d.nu);hl,scl,_=fit_head(d,erl,ex,seed,unc=unc0,gev=CosEv(mu,d.nu),free=False);ev('v0',scl(kap),True)
    out['time']=dict(xsim=xi['train_time'],prior=fb['t_prior'],cat=fb['t_cat'],head=sh.sec,total_sgc=xi['train_time']+fb['t_prior']+fb['t_cat']+sh.sec,peak_mb_xsim=xi['peak_mb'],peak_mb_sgc=max(xi['peak_mb'],fb['m_prior'],fb['m_cat'],sh.mb))
    np.savez_compressed(HERE/'results'/f'main_{ds}_s{seed}.npz',**arr);json.dump(out,open(f,'w'),indent=1)
