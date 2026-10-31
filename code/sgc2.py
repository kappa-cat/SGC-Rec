"""SGC-Rec pipeline (PyTorch / CUDA): Gaussian-mixture prior with sampled-softmax expected-likelihood energy, full-catalogue expert and Gaussian decision head."""
import json,time,os,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from scipy.stats import spearmanr
from common import HERE,DEV,Data,evaluate,entropy
from models import MODELS,Head,normalise,raw_features
import models4  # registers SGCRecR
from gaussian_pretrain import fit_gaussian_prior as fit_prior_legacy
(HERE/'results'/'cache').mkdir(parents=True,exist_ok=True)

def sync():
    if DEV.type=='cuda': torch.cuda.synchronize()
class Stage:
    """wall-clock seconds and peak allocated GPU MB of one stage"""
    def __init__(s): s.sec=0.;s.mb=0.
    def __enter__(s):
        if DEV.type=='cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        sync(); s.t=time.perf_counter(); return s
    def __exit__(s,*a):
        sync(); s.sec=time.perf_counter()-s.t; s.mb=torch.cuda.max_memory_allocated()/2**20 if DEV.type=='cuda' else 0.

XS={'beauty':dict(lr=.002,lam=.10,eps=.10,tau=.20,L=3),'lastfm':dict(lr=.001,lam=.05,eps=.20,tau=.15,L=2),'ml10m':dict(lr=.001,lam=.05,eps=.20,tau=.15,L=2)}
TAUR={'beauty':.15,'lastfm':.10,'ml10m':.15}

def train_model(name,d,seed,args,lr,bs=2048,patience=10,max_epochs=None,min_epochs=0):
    gen=torch.Generator(device=DEV).manual_seed(seed);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    m=MODELS[name](d,dict(args),gen,rng)
    if not getattr(m,'trainable',True):
        with Stage() as st: m.fit()
        return m,dict(train_time=st.sec,epochs=0,best_val=float('nan'),peak_mb=st.mb)
    m.build();m.pretrain()
    opt=torch.optim.Adam(m.parameters(),lr=lr);best=-1.;bad=0;state=None;T=0.;peak=0.;maxe=max_epochs or m.max_epochs;ep=0;bep=0
    for ep in range(maxe):
        with Stage() as st:
            m.train();m.before_epoch(ep);u,i,j=d.sample_epoch(gen,m.n_neg);step=bs*m.n_neg
            for a in range(0,len(u),step):
                loss=m.loss(u[a:a+step],i[a:a+step],j[a:a+step]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        T+=st.sec;peak=max(peak,st.mb)
        m.eval()
        with torch.no_grad(): r=float(evaluate(d,m.scorer(),'val')['R'].mean())
        if r>best: best=r;bad=0;bep=ep+1;state={k:v.detach().clone() for k,v in m.state_dict().items()}
        else:
            bad+=1
            if bad>=patience and ep+1>=min_epochs: break
    m.load_state_dict(state);m.eval()
    return m,dict(train_time=T,epochs=ep+1,best_val=best,peak_mb=peak,best_epoch=bep)

def get_xsim(d,ds,seed,noise=0.,tag='',min_epochs=0):
    f=HERE/'results'/'cache'/f'xsim_{ds}_s{seed}_n{int(noise*100)}{tag}.pt'
    if f.exists():
        z=torch.load(f,map_location=DEV,weights_only=True);return z['Z'],z['info']
    a=dict(XS[ds]);lr=a.pop('lr');a.update(d=64,reg=1e-4,lstar=1)
    m,info=train_model('XSimGCL',d,seed,a,lr,bs=2048,max_epochs=150,min_epochs=min_epochs)
    with torch.no_grad(): Z=m.embed().detach()
    torch.save(dict(Z=Z,info=info),f);return Z,info

class Ev:
    """standardized dot-product evidence from an embedding matrix"""
    def __init__(s,Z,nu):
        s.U,s.I=Z[:nu],Z[nu:];mean=s.I.mean(0);C=(s.I-mean).T@(s.I-mean)/s.I.shape[0]
        s.o=s.U@mean;s.s=torch.sqrt(torch.einsum('bd,de,be->b',s.U,C,s.U).clamp_min(1e-8))
    def pair(s,u,i): return ((s.U[u]*s.I[i]).sum(1)-s.o[u])/s.s[u]
    def full(s,u): return (s.U[u]@s.I.T-s.o[u,None])/s.s[u,None]
    def raw(s,u): return s.U[u]@s.I.T

def energy_pair(ma,qa,mb,qb):
    v=(qa+qb).clamp_min(1e-4);return -(((ma-mb)**2).mean(-1)/v+v.log())
def energy_all(mu_u,q_u,mu_i,q_i):
    d=mu_u.shape[1];dist=((mu_u**2).sum(1)[:,None]+(mu_i**2).sum(1)[None]-2*mu_u@mu_i.T).clamp_min(0)/d
    v=(q_u[:,None]+q_i[None]).clamp_min(1e-4);return -(dist/v+v.log())

class EnergyEv:
    """per-user standardized Gaussian expected-likelihood affinity"""
    def __init__(s,pr,nu):
        s.mu_u,s.mu_i,s.qu,s.qi=pr['mu'][:nu],pr['mu'][nu:],pr['q'][:nu],pr['q'][nu:];s.nu=nu
        o=torch.zeros(nu,device=DEV);sd=torch.ones(nu,device=DEV)
        with torch.no_grad():
            for a in range(0,nu,256):
                b=torch.arange(a,min(a+256,nu),device=DEV);E=energy_all(s.mu_u[b],s.qu[b],s.mu_i,s.qi);o[b]=E.mean(1);sd[b]=E.std(1).clamp_min(1e-6)
        s.o,s.s=o,sd
    def raw(s,u): return energy_all(s.mu_u[u],s.qu[u],s.mu_i,s.qi)
    def pair(s,u,i): return (energy_pair(s.mu_u[u],s.qu[u],s.mu_i[i],s.qi[i])-s.o[u])/s.s[u]
    def full(s,u): return (s.raw(u)-s.o[u,None])/s.s[u,None]

class CosEv:
    """bounded affinity: standardized cosine of the prior means (linear-head prior ablation)"""
    def __init__(s,mu,nu):
        s.e=Ev(F.normalize(mu,dim=1),nu)
    def raw(s,u): return s.e.raw(u)
    def pair(s,u,i): return s.e.pair(u,i)
    def full(s,u): return s.e.full(u)

def fit_prior(d,seed,epochs=8,K=64,P=2048,bs=512,lr=4e-3,beta_kl=1e-3,C=2,dim=64,social_weight=0.,select=True,verbose=False):
    gen=torch.Generator(device=DEV).manual_seed(1601+seed);torch.manual_seed(1601+seed)
    uf,itf=raw_features(d);xu,xi=normalise(uf,'cdf'),normalise(itf,'cdf')
    hu=Head(xu.shape[1],dim,C,'sph',gen).to(DEV);hi=Head(xi.shape[1],dim,C,'sph',gen).to(DEV)
    s=nn.Parameter(torch.zeros((),device=DEV));opt=torch.optim.Adam(list(hu.parameters())+list(hi.parameters())+[s],lr=lr)
    src,dst=d.tu,d.ti;n=len(src);tk=d.train_key;best=(-1.,None,0);hist=[];zero=lambda B:torch.zeros(B,dtype=torch.long,device=DEV)
    tic=time.perf_counter()
    def enc():
        mu_u,var_u=hu(xu);mu_i,var_i=hi(xi);return mu_u,var_u.mean(1),mu_i,var_i.mean(1)  # scalar variance = average over dimensions
    for ep in range(epochs):
        perm=torch.randperm(n,device=DEV,generator=gen)
        for st in range(0,n,bs):
            e=perm[st:st+bs];u,i=src[e],dst[e];B=len(e)
            mu_u,qu,mu_i,qi=enc()
            pool=torch.randint(0,d.ni,(P,),device=DEV,generator=gen);j=pool[torch.randint(0,P,(B,K),device=DEV,generator=gen)]
            pos=energy_pair(mu_u[u],qu[u],mu_i[i],qi[i]);neg=energy_pair(mu_u[u][:,None,:],qu[u][:,None],mu_i[j],qi[j])
            key=u[:,None]*d.ni+j;pp=torch.searchsorted(tk,key.reshape(-1)).clamp(max=len(tk)-1).reshape(B,K);neg=neg.masked_fill(tk[pp]==key,-1e9)
            l=F.cross_entropy(torch.cat([pos[:,None],neg],1)*s.exp(),zero(B))
            kl=lambda m,q:(.5*(q[:,None]+m.square()-1-q[:,None].log()).sum(1)).mean()
            loss=l+beta_kl*(kl(mu_u,qu)+kl(mu_i,qi));opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if social_weight>0 and d.name=='lastfm' and len(d.social):
            su=torch.from_numpy(d.social[:,0]).long().to(DEV);sv=torch.from_numpy(d.social[:,1]).long().to(DEV);so=torch.randperm(len(su),device=DEV,generator=gen)
            for st in range(0,len(so),bs):
                e=so[st:st+bs];u,v=su[e],sv[e];B=len(e);mu_u,qu,_,_=enc();j=torch.randint(0,d.nu,(B,K),device=DEV,generator=gen)
                pos=energy_pair(mu_u[u],qu[u],mu_u[v],qu[v]);neg=energy_pair(mu_u[u][:,None,:],qu[u][:,None],mu_u[j],qu[j])
                l=social_weight*F.cross_entropy(torch.cat([pos[:,None],neg],1)*s.exp(),zero(B));opt.zero_grad(set_to_none=True);l.backward();opt.step()
        with torch.no_grad():
            mu_u,qu,mu_i,qi=enc();r=float(evaluate(d,lambda ub:energy_all(mu_u[ub],qu[ub],mu_i,qi),'val')['R'].mean());hist.append(r)
            if (not select) or r>best[0]: best=(r,(mu_u.detach().clone(),qu.detach().clone(),mu_i.detach().clone(),qi.detach().clone()),ep+1)
    mu_u,qu,mu_i,qi=best[1];sd=lambda x:(x-x.mean())/x.std().clamp_min(1e-8)
    sync();return dict(mu=torch.cat([mu_u,mu_i]),q=torch.cat([qu,qi]),unc=torch.cat([sd(qu),sd(qi)]),val_R=hist,epochs=best[2],seconds=time.perf_counter()-tic)

def train_cat(d,ds,seed,epochs=4,prior_mu=None,alpha=1.,pscale=3.,lr=.002,bs=1024,social=None,anchor=0.,anchor_deg=False):
    args=dict(d=64,L=2,reg=1e-4,tau_r=TAUR[ds],balance=.5,ego=1,mask_positive=1,social=(.5 if (ds=='lastfm' if social is None else social) else 0))
    gen=torch.Generator(device=DEV).manual_seed(907+seed);torch.manual_seed(907+seed)
    m=MODELS['SGCRecR'](d,args,gen,np.random.default_rng(907+seed));m.build()
    if prior_mu is not None:
        with torch.no_grad():
            sc=pscale if pscale else m.E.norm(dim=1).mean().clamp_min(1e-8);s0=sc  # s = mean norm of the INITIAL table
            m.E.mul_(1-alpha).add_(F.normalize(prior_mu,dim=1)*sc,alpha=alpha)
    opt=torch.optim.Adam(m.parameters(),lr=lr)
    if prior_mu is not None and anchor>0:
        with torch.no_grad():
            tgt=(alpha*F.normalize(prior_mu,dim=1)*s0).detach()  # anchor target alpha*normalize(m)*s
            dg=torch.cat([torch.tensor(np.asarray(d.R.sum(1)).ravel(),device=DEV),torch.tensor(np.asarray(d.R.sum(0)).ravel(),device=DEV)]).float()
            w=anchor/(1+dg/dg.median().clamp_min(1.)) if anchor_deg else torch.full_like(dg,anchor)
    for _ in range(epochs):
        u,i,j=d.sample_epoch(gen)
        for a in range(0,len(u),bs):
            loss=m.loss(u[a:a+bs],i[a:a+bs],j[a:a+bs])
            if prior_mu is not None and anchor>0: loss=loss+(w*(m.E-tgt).square().sum(1)).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    m.eval()
    with torch.no_grad(): Z=m.embed().detach()
    return Z

class GHead(nn.Module):
    def __init__(s,free=True):
        super().__init__();s.free=free;s.ls=nn.Parameter(torch.zeros(2));s.gate=nn.Parameter(torch.tensor([0.,.5]));s.ps=nn.Parameter(torch.tensor(0.));s.var=nn.Parameter(torch.tensor([-1.,.25,0.,-.25,-2.]))
    def forward(s,zr,zx,ud,idg,pu=None,pi=None,zg=None):
        g=torch.sigmoid(s.gate[0]+s.gate[1]*ud);mu=g*zr*s.ls[0].exp()+(1-g)*zx*s.ls[1].exp()
        if zg is not None: mu=mu+(s.ps.exp() if s.free else .1*torch.sigmoid(s.ps))*zg
        lv=s.var[0]+s.var[1]*(zr-zx).abs()+s.var[2]*ud+s.var[3]*idg
        if pu is not None: lv=lv+F.softplus(s.var[4])*(pu+pi)
        return mu,F.softplus(lv)+1e-4

def degs(d):
    du=torch.tensor(np.asarray(d.R.sum(1)).ravel(),device=DEV).float();di=torch.tensor(np.asarray(d.R.sum(0)).ravel(),device=DEV).float()
    lud=(du+1).log();lid=(di+1).log();return (lud-lud.mean())/lud.std(),(lid-lid.mean())/lid.std()

def fit_head(d,er,ex,seed,epochs=40,unc=None,gev=None,free=True):
    lud,lid=degs(d);pu=pi=None
    if unc is not None: pu,pi=unc[:d.nu],unc[d.nu:]
    torch.manual_seed(1271+seed);h=GHead(free).to(DEV);opt=torch.optim.Adam(h.parameters(),lr=.02);g=torch.Generator(device=DEV).manual_seed(3817+seed)
    def fwd(u,i,pair,ud,idg,pu_,pi_):
        return h(er.pair(u,i),ex.pair(u,i),ud,idg,pu_,pi_,gev.pair(u,i) if gev is not None else None)
    sync();tic=time.perf_counter()
    for _ in range(epochs):
        u,i,j=d.sample_epoch(g)
        for a in range(0,len(u),4096):
            ub,ib,jb=u[a:a+4096],i[a:a+4096],j[a:a+4096]
            mp,vp=fwd(ub,ib,None,lud[ub],lid[ib],pu[ub] if pu is not None else None,pi[ib] if pu is not None else None)
            mn,vn=fwd(ub,jb,None,lud[ub],lid[jb],pu[ub] if pu is not None else None,pi[jb] if pu is not None else None)
            delta=mp-mn;V=vp+vn;loss=F.softplus(-delta).mean()+.05*(V.log()+(1-delta).square()/V).mean()+1e-4*sum(q.square().sum() for q in h.parameters())
            opt.zero_grad();loss.backward();opt.step()
    h.eval();sync();sec=time.perf_counter()-tic
    def scorer(k):
        def f(u):
            with torch.no_grad():
                mu,v=h(er.full(u),ex.full(u),lud[u,None],lid[None],pu[u,None] if pu is not None else None,pi[None] if pu is not None else None,gev.full(u) if gev is not None else None);return mu-k*v.sqrt()
        return f
    return h,scorer,sec

def calib(d,h,er,ex,gev,unc,seed):
    lud,lid=degs(d);pu,pi=(unc[:d.nu],unc[d.nu:]) if unc is not None else (None,None)
    co=d.val.tocoo();u=co.row.astype(np.int64);i=co.col.astype(np.int64);rng=np.random.default_rng(99+seed);j=rng.integers(0,d.ni,len(u),dtype=np.int64);known=(d.R+d.val).tocsr()
    for _ in range(100):
        bad=np.asarray(known[u,j]).ravel()>0
        if not bad.any():break
        j[bad]=rng.integers(0,d.ni,bad.sum())
    U=torch.tensor(u,device=DEV);I=torch.tensor(i,device=DEV);J=torch.tensor(j,device=DEV)
    with torch.no_grad():
        def f(it): return h(er.pair(U,it),ex.pair(U,it),lud[U],lid[it],pu[U] if pu is not None else None,pi[it] if pu is not None else None,gev.pair(U,it) if gev is not None else None)
        mp,vp=f(I);mn,vn=f(J);V=(vp+vn).cpu().numpy();sq=(1-(mp-mn)).square().cpu().numpy()
    half=len(V)//2;T=np.mean(sq[:half]/V[:half]);rho=spearmanr(T*V[half:],sq[half:]).statistic
    return dict(spearman=float(rho),M2=float(np.mean(sq[half:]/(T*V[half:]))))

def summ(ev,d): return dict(P=float(ev['P'].mean()),R=float(ev['R'].mean()),N=float(ev['N'].mean()),entropy=entropy(d,ev['top']))

def build_full(d,ds,seed,cfg,epochs=None,social=None):
    """Gaussian prior -> catalogue expert -> evidence; returns a dict of objects"""
    ep=epochs or cfg.get('cat_epochs',4)
    with Stage() as sp: pr=fit_prior(d,seed,epochs=cfg.get('prior_epochs',8),social_weight=cfg.get('social_weight',0.))
    with Stage() as sc: Z=train_cat(d,ds,seed,ep,pr['mu'],cfg.get('alpha',1.),cfg.get('pscale',3.),anchor=cfg.get('anchor',0.),anchor_deg=cfg.get('anchor_deg',False),social=social)
    return dict(pr=pr,Z=Z,er=Ev(Z,d.nu),gev=EnergyEv(pr,d.nu),t_prior=sp.sec,m_prior=sp.mb,t_cat=sc.sec,m_cat=sc.mb)

def build_nopre(d,ds,seed,cfg,epochs=None,social=None):
    with Stage() as sc: Z=train_cat(d,ds,seed,epochs or cfg.get('cat_epochs',4),None,social=social)
    return dict(Z=Z,er=Ev(Z,d.nu),t_cat=sc.sec,m_cat=sc.mb)
