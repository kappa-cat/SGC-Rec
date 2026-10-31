"""Single-encoder SGC-Rec variants with full-catalogue ranking.

Social edges are used only on native-social data, with training-overlap shrinkage.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from common import DEV, xavier, sp_to_torch, sym_norm
from models import Base, MODELS, propagate, noisy_prop
from models3 import item_item_block


class SGCRecR(Base):
    max_epochs = 150

    def build(self):
        d = self.data
        self.E = xavier(d.n, self.d, self.gen)
        if self.a.get('item_bias', 0):
            self.item_bias = torch.nn.Parameter(torch.zeros(d.ni, device=DEV))
        if self.a.get('ii_k',0):
            J=d.joint().tolil()
            J[d.nu:,d.nu:]=item_item_block(d,self.a['ii_k'],self.a.get('ii_w',1.0))
            self.A=sp_to_torch(sym_norm(J.tocsr()))
        else:
            self.A = d.norm_adj()
        deg = np.asarray(d.R.sum(1)).ravel()
        self.degree = torch.tensor(deg, device=DEV)
        self.S = None
        if self.a.get('social', 0) > 0:
            if d.name != 'lastfm':
                raise ValueError('Native-social branch is only valid for Last-FM in this package')
            C = (d.R @ d.R.T).tocsr()
            S = d.S.tocoo()
            common = np.asarray(C[S.row, S.col]).ravel()
            union = deg[S.row] + deg[S.col] - common
            # Beta(1,1) posterior mean for binary preference overlap, used as a
            # reliability weight for each friendship.
            reliability = (common + 1) / (union + 2)
            W = sp.csr_matrix((reliability, (S.row, S.col)), shape=S.shape)
            mass = np.asarray(W.sum(1)).ravel()
            self.S = sp_to_torch(sp.diags(1 / np.maximum(mass, 1e-12)) @ W)
            gate = mass / (mass + np.sqrt(np.maximum(deg, 1)))
            self.gate = torch.tensor(gate[:, None], dtype=torch.float32, device=DEV)

    def embed(self):
        X = propagate(self.E, self.A, self.L, include_ego=bool(self.a.get('ego', 1)))
        nrm = bool(self.a.get('norm', 1))
        if nrm: X = F.normalize(X, dim=1)                      # normalized catalogue representation h^r
        if self.S is not None:                                  # social mixing: mix the NORMALIZED representations, then re-normalize
            U = X[:self.nu]
            g = self.a['social'] * self.gate
            X = torch.cat([U + g * (torch.sparse.mm(self.S, U) - U), X[self.nu:]])
            if nrm: X = F.normalize(X, dim=1)
        return X

    def loss(self, u, i, j):
        X = self.embed()
        logits = X[u] @ X[self.nu:].T / self.a.get('tau_r', .1)
        if hasattr(self, 'item_bias'): logits = logits + self.item_bias[None]
        # Exclude OTHER known positives from the denominator. Labels from
        # validation/test never participate in negative masks or gradients.
        if self.a.get('mask_positive', 1):
            ex = self.data.R[u.detach().cpu().numpy()].tocoo()
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[torch.tensor(ex.row, device=DEV).long(), torch.tensor(ex.col, device=DEV).long()] = True
            mask[torch.arange(len(u), device=DEV), i] = False
            logits = logits.masked_fill(mask, -torch.inf)
        losses = F.cross_entropy(logits, i, reduction='none')
        w = self.degree[u].clamp_min(1).pow(-self.a.get('balance', .5))
        w = w / w.mean()
        return (w * losses).mean() + self.reg_loss(self.E, u, i, j)

    @torch.no_grad()
    def scorer(self):
        X = self.embed()
        if hasattr(self, 'item_bias'):
            return lambda u: X[u] @ X[self.nu:].T / self.a.get('tau_r', .1) + self.item_bias[None]
        return self.dot_scorer(X)


MODELS['SGCRecR'] = SGCRecR


class SGCRecJoint(SGCRecR):
    """One-encoder joint listwise/contrastive variant.

    Predictive disagreement is measured between the recommendation embedding
    and the retained intermediate layer. It changes both the contrastive
    weight and, when rho > 0, the ranking score. Thus its ablation is exact:
    rho=uw=0 removes every disagreement-dependent operation.
    """
    def representation(self, train):
        eps = self.a.get('eps', .1) if train else 0.
        X, Xcl = noisy_prop(self.E, self.A, self.L, eps,
                            keep_layer=self.a.get('lstar', 1))
        X, Xcl = F.normalize(X, dim=1), F.normalize(Xcl, dim=1)
        if self.S is not None:
            g = self.a['social'] * self.gate
            def social_fuse(Z):
                U = Z[:self.nu]
                U = F.normalize(U + g*(torch.sparse.mm(self.S,U)-U),dim=1)
                return torch.cat([U,Z[self.nu:]])
            X, Xcl = social_fuse(X), social_fuse(Xcl)
        return X, Xcl

    def uncertainty(self, X, Xcl):
        return (1 - (X * Xcl).sum(1)).clamp_min(0).detach()

    def loss(self, u, i, j):
        X, Xcl = self.representation(True)
        q = self.uncertainty(X, Xcl)
        logits = X[u] @ X[self.nu:].T / self.a.get('tau_r', .1)
        rho = self.a.get('rho', 0.)
        if rho:
            logits = logits - rho * torch.sqrt(q[u,None] + q[self.nu:][None] + 1e-8)
        ex = self.data.R[u.detach().cpu().numpy()].tocoo()
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[torch.tensor(ex.row,device=DEV).long(),torch.tensor(ex.col,device=DEV).long()] = True
        mask[torch.arange(len(u),device=DEV),i] = False
        ce = F.cross_entropy(logits.masked_fill(mask,-torch.inf),i,reduction='none')
        degree_w = self.degree[u].clamp_min(1).pow(-self.a.get('balance',.5))
        degree_w = degree_w / degree_w.mean()
        uu,ii=torch.unique(u),torch.unique(i)+self.nu
        def nce(nodes):
            a,b=X[nodes],Xcl[nodes]
            z=(torch.logsumexp(a@b.T/self.a.get('tau',.2),1)
               -(a*b).sum(1)/self.a.get('tau',.2))
            if self.a.get('uw',0):
                w=(1+self.a['uw']*q[nodes]);w=w/w.mean()
                z=z*w
            return z.mean()
        return (degree_w*ce).mean()+self.a.get('lam',.05)*(nce(uu)+nce(ii))+self.reg_loss(self.E,u,i,j)

    @torch.no_grad()
    def scorer(self):
        X,Xcl=self.representation(False);q=self.uncertainty(X,Xcl);rho=self.a.get('rho',0.)
        return lambda u: X[u]@X[self.nu:].T/self.a.get('tau_r',.1)-rho*torch.sqrt(q[u,None]+q[self.nu:][None]+1e-8)


MODELS['SGCRecJoint'] = SGCRecJoint


class SGCRecDisagreement(SGCRecR):
    """Deterministic, low-cost structural uncertainty calibration variant."""
    def rep_unc(self):
        X=self.E;layers=[X] if self.a.get('ego',1) else [];kept=None
        for level in range(self.L):
            X=torch.sparse.mm(self.A,X);layers.append(X)
            if level==self.a.get('lstar',1)-1:kept=X
        final=torch.stack(layers).mean(0)
        final=F.normalize(final,dim=1);kept=F.normalize(kept,dim=1)
        if self.S is not None:
            g=self.a['social']*self.gate
            def fuse(Z):
                U=Z[:self.nu]
                return torch.cat([F.normalize(U+g*(torch.sparse.mm(self.S,U)-U),dim=1),Z[self.nu:]])
            final,kept=fuse(final),fuse(kept)
        q=(1-(final*kept).sum(1)).clamp_min(0)
        return final,q

    def adjusted(self,X,q,u):
        scores=X[u]@X[self.nu:].T/self.a.get('tau_r',.1)
        # Standardized item instability gives rho a comparable meaning across data.
        qi=q[self.nu:];z=(qi-qi.mean())/qi.std().clamp_min(1e-8)
        return scores-self.a.get('rho',.1)*scores.std(1,keepdim=True)*z[None]

    def loss(self,u,i,j):
        X,q=self.rep_unc();logits=self.adjusted(X,q,u)
        ex=self.data.R[u.detach().cpu().numpy()].tocoo()
        mask=torch.zeros_like(logits,dtype=torch.bool)
        mask[torch.tensor(ex.row,device=DEV).long(),torch.tensor(ex.col,device=DEV).long()]=True
        mask[torch.arange(len(u),device=DEV),i]=False
        ce=F.cross_entropy(logits.masked_fill(mask,-torch.inf),i,reduction='none')
        w=self.degree[u].clamp_min(1).pow(-self.a.get('balance',.5));w=w/w.mean()
        return (w*ce).mean()+self.reg_loss(self.E,u,i,j)

    @torch.no_grad()
    def scorer(self):
        X,q=self.rep_unc()
        return lambda u:self.adjusted(X,q,u)


MODELS['SGCRecDisagreement']=SGCRecDisagreement


class SGCRecFusion(SGCRecR):
    """XSimGCL backbone plus a small catalogue-ranking auxiliary objective.

    It keeps the BPR and intermediate-layer contrastive
    terms. Structural instability is consumed through `rho`; `alpha=0,rho=0`
    is the matched backbone control within exactly the same implementation.
    """
    def reps(self, train):
        X,Xcl=noisy_prop(self.E,self.A,self.L,self.a.get('eps',.2) if train else 0,
                         keep_layer=self.a.get('lstar',1))
        if self.S is not None:
            g=self.a.get('social',0)*self.gate
            def fuse(Z):
                U=Z[:self.nu]
                return torch.cat([U+g*(torch.sparse.mm(self.S,U)-U),Z[self.nu:]])
            X,Xcl=fuse(X),fuse(Xcl)
        return X,Xcl

    def qscore(self,X,Xcl,u):
        base=X[u]@X[self.nu:].T
        rho=self.a.get('rho',0)
        if not rho:return base
        A,B=F.normalize(X,dim=1),F.normalize(Xcl,dim=1)
        q=(1-(A*B).sum(1)).clamp_min(0).detach();qi=q[self.nu:]
        z=(qi-qi.mean())/qi.std().clamp_min(1e-8)
        return base-rho*base.std(1,keepdim=True)*z[None]

    def loss(self,u,i,j):
        X,Xcl=self.reps(True);nu=self.nu
        pos=(X[u]*X[nu+i]).sum(1);neg=(X[u]*X[nu+j]).sum(1)
        rank=F.softplus(-(pos-neg)).mean()+self.reg_loss(self.E,u,i,j)
        uu,ii=torch.unique(u),torch.unique(i)+nu
        def nce(nodes):
            a,b=F.normalize(X[nodes],dim=1),F.normalize(Xcl[nodes],dim=1)
            return (torch.logsumexp(a@b.T/self.a.get('tau',.15),1)
                    -(a*b).sum(1)/self.a.get('tau',.15)).mean()
        loss=rank+self.a.get('lam',.05)*(nce(uu)+nce(ii))
        alpha=self.a.get('alpha',0)
        if alpha:
            logits=self.qscore(X,Xcl,u)/self.a.get('tau_r',.2)
            ex=self.data.R[u.detach().cpu().numpy()].tocoo()
            mask=torch.zeros_like(logits,dtype=torch.bool)
            mask[torch.tensor(ex.row,device=DEV).long(),torch.tensor(ex.col,device=DEV).long()]=True
            mask[torch.arange(len(u),device=DEV),i]=False
            ce=F.cross_entropy(logits.masked_fill(mask,-torch.inf),i,reduction='none')
            w=self.degree[u].clamp_min(1).pow(-self.a.get('balance',.5));w=w/w.mean()
            loss=loss+alpha*(w*ce).mean()
        return loss

    @torch.no_grad()
    def scorer(self):
        X,Xcl=self.reps(False)
        return lambda u:self.qscore(X,Xcl,u)


MODELS['SGCRecFusion']=SGCRecFusion
