"""Linear-head Gaussian pre-training (used for the linear-head prior ablation)."""
import math, time
import torch
import torch.nn.functional as F
from common import DEV
from models import Head, normalise, raw_features

def _standardize(x):
    return (x-x.mean())/x.std().clamp_min(1e-8)

def gaussian_energy(a, va, b, vb):
    """Analytic expected-likelihood energy for diagonal Gaussians."""
    v=(va+vb).clamp_min(1e-4)
    return -(((a-b).square()/v)+v.log()).mean(1)

def fit_gaussian_prior(data, dim=64, components=2, epochs=8, seed=0,
                       beta_kl=1e-3, lr=2e-3, batch_size=4096,
                       objective='distributional', social_weight=.2):
    """Fit CDF-conditioned diagonal mixtures using training edges only."""
    gen=torch.Generator(device=DEV).manual_seed(1601+seed);torch.manual_seed(1601+seed)
    uf,itf=raw_features(data);xu,xi=normalise(uf,'cdf'),normalise(itf,'cdf')
    hu=Head(xu.shape[1],dim,components,'diag',gen).to(DEV)
    hi=Head(xi.shape[1],dim,components,'diag',gen).to(DEV)
    opt=torch.optim.Adam(list(hu.parameters())+list(hi.parameters()),lr=lr)
    src=torch.from_numpy(data.tr_u).long().to(DEV);dst=torch.from_numpy(data.tr_i).long().to(DEV)
    # Exclude one-off CUDA context initialization from the stage timer.
    if DEV.type=='cuda':
        torch.ones(1,device=DEV).square();torch.cuda.synchronize()
    tic=time.perf_counter()
    for _ in range(epochs):
        order=torch.randperm(len(src),device=DEV,generator=gen)
        for start in range(0,len(order),batch_size):
            e=order[start:start+batch_size];u,i=src[e],dst[e]
            j=torch.randint(0,data.ni,(len(e),),device=DEV,generator=gen)
            # BPR negatives must be unobserved for the sampled user.  Rejection
            # uses the training CSR only, preserving the split boundary.
            for _reject in range(100):
                uu=u.detach().cpu().numpy().copy(); jj=j.detach().cpu().numpy().copy()
                bad=torch.from_numpy((data.R[uu,jj].A1>0)).to(DEV)
                if not bad.any(): break
                j[bad]=torch.randint(0,data.ni,(int(bad.sum()),),device=DEV,generator=gen)
            mu_u,var_u=hu(xu);mu_i,var_i=hi(xi)
            if objective=='distributional':
                # Expected-likelihood score for diagonal Gaussians; it trains
                # both means and variances on every pair.
                pos=gaussian_energy(mu_u[u],var_u[u],mu_i[i],var_i[i])
                neg=gaussian_energy(mu_u[u],var_u[u],mu_i[j],var_i[j])
            else:
                zu=mu_u[u]+var_u[u].sqrt()*torch.randn(mu_u[u].shape,device=DEV,generator=gen)
                zi=mu_i[i]+var_i[i].sqrt()*torch.randn(mu_i[i].shape,device=DEV,generator=gen)
                zj=mu_i[j]+var_i[j].sqrt()*torch.randn(mu_i[j].shape,device=DEV,generator=gen)
                pos=(zu*zi).sum(1)/math.sqrt(dim);neg=(zu*zj).sum(1)/math.sqrt(dim)
            uu,ii=torch.unique(u),torch.unique(torch.cat([i,j]))
            ku=.5*(var_u[uu]+mu_u[uu].square()-1-var_u[uu].log()).sum(1).mean()
            ki=.5*(var_i[ii]+mu_i[ii].square()-1-var_i[ii].log()).sum(1).mean()
            loss=F.softplus(-(pos-neg)).mean()+beta_kl*(ku+ki)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        # Native Last-FM friendships supervise the user Gaussians; the term is
        # inactive on datasets whose auxiliary user graph is derived from interactions.
        if data.name=='lastfm' and len(data.social) and social_weight>0:
            su=torch.from_numpy(data.social[:,0]).long().to(DEV);sv=torch.from_numpy(data.social[:,1]).long().to(DEV)
            so=torch.randperm(len(su),device=DEV,generator=gen)
            for start in range(0,len(so),batch_size):
                e=so[start:start+batch_size];u,v=su[e],sv[e];j=torch.randint(0,data.nu,(len(e),),device=DEV,generator=gen)
                for _reject in range(100):
                    uu=u.detach().cpu().numpy().copy();jj=j.detach().cpu().numpy().copy();bad=torch.from_numpy((data.S[uu,jj].A1>0)).to(DEV)
                    if not bad.any():break
                    j[bad]=torch.randint(0,data.nu,(int(bad.sum()),),device=DEV,generator=gen)
                mu_u,var_u=hu(xu);sp=gaussian_energy(mu_u[u],var_u[u],mu_u[v],var_u[v]);sn=gaussian_energy(mu_u[u],var_u[u],mu_u[j],var_u[j]);loss=social_weight*F.softplus(-(sp-sn)).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    with torch.no_grad():
        mu_u,var_u=hu(xu);mu_i,var_i=hi(xi);mu=torch.cat([mu_u,mu_i]).detach()
        unc=torch.cat([_standardize(var_u.mean(1)),_standardize(var_i.mean(1))]).detach()
    if DEV.type=='cuda':torch.cuda.synchronize()
    state={'user_head':hu.state_dict(),'item_head':hi.state_dict()}
    meta=dict(norm='empirical_cdf',components=components,covariance='diagonal',objective=objective,epochs=epochs,beta_kl=beta_kl,social_weight=social_weight if data.name=='lastfm' else 0,seconds=time.perf_counter()-tic)
    return mu,unc,state,meta

def inject_prior(embedding,mean,weight=.15):
    """Blend the learned mean into an expert at the Xavier norm scale."""
    with torch.no_grad():
        scale=embedding.norm(dim=1).mean().clamp_min(1e-8)
        embedding.mul_(1-weight).add_(F.normalize(mean,dim=1)*scale,alpha=weight)
