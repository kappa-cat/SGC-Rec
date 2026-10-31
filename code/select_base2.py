"""validation-seed-0 hyper-parameter search for the contrastive baselines (8 configurations each; SEPT 4)"""
import sys,json,itertools
from sgc2 import *
GRID={
 'SGL':[dict(lr=lr,lam=lam,drop=dr) for lr in(.001,.002) for lam in(.02,.1) for dr in(.1,.3)],
 'SimGCL':[dict(lr=lr,lam=lam,eps=ep) for lr in(.001,.002) for lam in(.05,.2) for ep in(.1,.2)],
 'LightGCL':[dict(lr=lr,lam1=l1,tau=t) for lr in(.001,.002) for l1 in(.02,.2) for t in(.2,.5)],
 'HCCF':[dict(lr=lr,ssl=s,tau=t) for lr in(.001,.002) for s in(1e-3,1e-2) for t in(.5,1.)],
 'SEPT':[dict(lr=lr,ss=s) for lr in(.001,.002) for s in(.005,.05)],
}
f=HERE/'results'/'baseline_cfg2.json';out=json.load(open(f)) if f.exists() else {}
for ds in sys.argv[1:]:
    d=Data(ds,0);out.setdefault(ds,{})
    for n,cfgs in GRID.items():
        best=None
        for c in cfgs:
            c=dict(c);lr=c.pop('lr')
            m,info=train_model(n,d,0,dict(d=64,L=2,reg=1e-4,**c),lr,max_epochs=150)
            print(ds,n,lr,c,round(info['best_val'],4),info['epochs'],flush=True)
            if best is None or info['best_val']>best[0]: best=(info['best_val'],dict(lr=lr,**c))
            del m;torch.cuda.empty_cache()
        out[ds][n]=best[1];print('SELECTED',ds,n,best,flush=True)
        json.dump(out,open(f,'w'),indent=1)
