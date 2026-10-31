"""Fusion-head variants for the scarce-data regime: the head can be fitted on training pairs (default) or on held-out validation pairs."""
from sgc2 import *

def _val_pairs(d,parity=None):
    co=d.val.tocoo();u=co.row.astype(np.int64);i=co.col.astype(np.int64)
    if parity is not None: m=(u%2)==parity;u,i=u[m],i[m]
    return torch.from_numpy(u).to(DEV),torch.from_numpy(i).to(DEV)

def fit_head_gen(d,er,ex,seed,src='train',parity=None,epochs=40,l2=1e-4,unc=None,gev=None,free=True,g0=0.0):
    lud,lid=degs(d);pu=pi=None
    if unc is not None: pu,pi=unc[:d.nu],unc[d.nu:]
    torch.manual_seed(1271+seed);h=GHead(free).to(DEV)
    with torch.no_grad(): h.gate[0]=g0   # initial gate bias (0 = default); negative starts closer to the interaction expert
    opt=torch.optim.Adam(h.parameters(),lr=.02);g=torch.Generator(device=DEV).manual_seed(3817+seed)
    if src=='val':
        vu,vi=_val_pairs(d,parity);known=torch.cat([d.train_key,vu*d.ni+vi]).sort().values
    def fwd(u,i): return h(er.pair(u,i),ex.pair(u,i),lud[u],lid[i],pu[u] if pu is not None else None,pi[i] if pu is not None else None,gev.pair(u,i) if gev is not None else None)
    for _ in range(epochs):
        if src=='train': u,i,j=d.sample_epoch(g)
        else:
            perm=torch.randperm(len(vu),device=DEV,generator=g);u,i=vu[perm],vi[perm];j=torch.randint(0,d.ni,(len(u),),device=DEV,generator=g)
            for _r in range(50):
                key=u*d.ni+j;p=torch.searchsorted(known,key).clamp(max=len(known)-1);bad=known[p]==key
                if not bad.any(): break
                j[bad]=torch.randint(0,d.ni,(int(bad.sum()),),device=DEV,generator=g)
        for a in range(0,len(u),4096):
            ub,ib,jb=u[a:a+4096],i[a:a+4096],j[a:a+4096];mp,vp=fwd(ub,ib);mn,vn=fwd(ub,jb)
            delta=mp-mn;V=vp+vn;loss=F.softplus(-delta).mean()+.05*(V.log()+(1-delta).square()/V).mean()+l2*sum(q.square().sum() for q in h.parameters())
            opt.zero_grad();loss.backward();opt.step()
    h.eval();return h

def head_scorer(d,h,er,ex,unc,gev,k):
    lud,lid=degs(d);pu,pi=(unc[:d.nu],unc[d.nu:]) if unc is not None else (None,None)
    def f(u):
        with torch.no_grad():
            mu,v=h(er.full(u),ex.full(u),lud[u,None],lid[None],pu[u,None] if pu is not None else None,pi[None] if pu is not None else None,gev.full(u) if gev is not None else None);return mu-k*v.sqrt()
    return f

RECIPES={'A':dict(src='train',epochs=40,l2=1e-4),'B':dict(src='train',epochs=10,l2=1e-2),'C':dict(src='val',epochs=40,l2=1e-4),'D':dict(src='val',epochs=20,l2=1e-2),'E':dict(src='val',epochs=40,l2=1e-4,g0=-3.0),'F':dict(src='val',epochs=20,l2=1e-2,g0=-3.0)}

def recipe_scorers(d,er,ex,seed,unc,gev,free,k):
    """returns {recipe: (val_scorer, test_scorer)}; validation-fitted recipes use 2-fold user-disjoint cross-fitting for the validation score"""
    out={}
    for name,r in RECIPES.items():
        if r['src']=='train':
            h=fit_head_gen(d,er,ex,seed,unc=unc,gev=gev,free=free,**r);s=head_scorer(d,h,er,ex,unc,gev,k);out[name]=(s,s)
        else:
            hs=[fit_head_gen(d,er,ex,seed,parity=p,unc=unc,gev=gev,free=free,**r) for p in (0,1)];ss=[head_scorer(d,h,er,ex,unc,gev,k) for h in hs]
            def cv(u,ss=ss): return torch.where((u%2==0)[:,None],ss[1](u),ss[0](u))
            hall=fit_head_gen(d,er,ex,seed,parity=None,unc=unc,gev=gev,free=free,**r);out[name]=(cv,head_scorer(d,hall,er,ex,unc,gev,k))
    return out
