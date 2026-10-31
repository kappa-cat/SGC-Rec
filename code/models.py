"""SGC-Rec and baselines in PyTorch."""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *

spmm = torch.sparse.mm


def propagate(E, A, L, include_ego=True):
    layers = [E] if include_ego else []
    X = E
    for _ in range(L):
        X = spmm(A, X); layers.append(X)
    return torch.stack(layers, 0).mean(0)


def drop_edges(R, rho, rng):
    R = R.tocoo(); keep = rng.random(R.nnz) >= rho
    return sp.csr_matrix((R.data[keep], (R.row[keep], R.col[keep])), shape=R.shape)


class Base(nn.Module):
    n_neg = 1
    max_epochs = 200
    trainable = True

    def __init__(self, data, a, gen, rng):
        super().__init__()
        self.data, self.a, self.gen, self.rng = data, a, gen, rng
        self.d = a.get('d', 64); self.L = a.get('L', 2); self.reg = a.get('reg', 1e-4)
        self.nu, self.ni = data.nu, data.ni

    def before_epoch(self, ep): pass
    def pretrain(self): pass

    def reg_loss(self, E, u, i, j):
        return self.reg * (E[u].pow(2).sum() + E[self.nu + i].pow(2).sum() + E[self.nu + j].pow(2).sum()) / len(u)

    def cl_pair(self, v1, v2, u, i, tau):
        uu = torch.unique(u); ii = torch.unique(i) + self.nu
        return info_nce(v1[uu], v2[uu], tau) + info_nce(v1[ii], v2[ii], tau)

    def dot_scorer(self, X):
        U, I = X[:self.nu], X[self.nu:]
        return lambda ub: U[ub] @ I.T


# ============================================================================ graph baselines
class LightGCN(Base):
    max_epochs = 300

    def build(self):
        self.A = self.data.norm_adj()
        self.E = xavier(self.data.n, self.d, self.gen)

    def embed(self):
        return propagate(self.E, self.A, self.L)

    def loss(self, u, i, j):
        X = self.embed(); eu, ei, ej = X[u], X[self.nu + i], X[self.nu + j]
        return bpr((eu * ei).sum(1), (eu * ej).sum(1)) + self.reg_loss(self.E, u, i, j)

    @torch.no_grad()
    def scorer(self):
        return self.dot_scorer(self.embed())


class SGL(LightGCN):
    """SGL-ED (edge dropout, views rebuilt every epoch)."""
    max_epochs = 200

    def before_epoch(self, ep):
        self.views = [self.data.norm_adj(drop_edges(self.data.R, self.a.get('drop', 0.1), self.rng)) for _ in range(2)]

    def loss(self, u, i, j):
        l = super().loss(u, i, j)
        v1, v2 = propagate(self.E, self.views[0], self.L), propagate(self.E, self.views[1], self.L)
        return l + self.a.get('lam', 0.1) * self.cl_pair(v1, v2, u, i, self.a.get('tau', 0.2))


def noisy_prop(E, A, L, eps, keep_layer=None):
    X = E; layers = []; kept = None
    for l in range(L):
        X = spmm(A, X)
        if eps > 0:
            X = X + torch.sign(X) * F.normalize(torch.rand_like(X), dim=-1) * eps
        layers.append(X)
        if keep_layer is not None and l == keep_layer - 1:
            kept = X
    return torch.stack(layers, 0).mean(0), kept


class SimGCL(LightGCN):
    max_epochs = 150

    def embed(self):
        return noisy_prop(self.E, self.A, self.L, 0.0)[0]

    def loss(self, u, i, j):
        X = self.embed(); eu, ei, ej = X[u], X[self.nu + i], X[self.nu + j]
        l = bpr((eu * ei).sum(1), (eu * ej).sum(1)) + self.reg_loss(self.E, u, i, j)
        eps = self.a.get('eps', 0.1)
        v1 = noisy_prop(self.E, self.A, self.L, eps)[0]; v2 = noisy_prop(self.E, self.A, self.L, eps)[0]
        return l + self.a.get('lam', 0.5) * self.cl_pair(v1, v2, u, i, self.a.get('tau', 0.2))


class XSimGCL(SimGCL):
    def loss(self, u, i, j):
        X, Xcl = noisy_prop(self.E, self.A, self.L, self.a.get('eps', 0.2), keep_layer=self.a.get('lstar', 1))
        eu, ei, ej = X[u], X[self.nu + i], X[self.nu + j]
        l = bpr((eu * ei).sum(1), (eu * ej).sum(1)) + self.reg_loss(self.E, u, i, j)
        return l + self.a.get('lam', 0.2) * self.cl_pair(X, Xcl, u, i, self.a.get('tau', 0.15))


class LightGCL(Base):
    """Cai et al. ICLR'23, following the authors' code (rank-q SVD view, full-set denominators)."""
    max_epochs = 150

    def build(self):
        R = self.data.R
        du = np.asarray(R.sum(1)).ravel(); di = np.asarray(R.sum(0)).ravel()
        with np.errstate(divide='ignore'):
            Rn = sp.diags(np.where(du > 0, du ** -0.5, 0)) @ R @ sp.diags(np.where(di > 0, di ** -0.5, 0))
        self.Rn = sp_to_torch(Rn); self.RnT = sp_to_torch(Rn.T)
        U, s, V = torch.svd_lowrank(self.Rn, q=self.a.get('q', 5))
        self.us, self.vs, self.ut, self.vt = U * s, V * s, U.T.contiguous(), V.T.contiguous()
        self.Eu = xavier(self.nu, self.d, self.gen); self.Ei = xavier(self.ni, self.d, self.gen)

    def forward_all(self):
        Eu, Ei, Gu, Gi = [self.Eu], [self.Ei], [self.Eu], [self.Ei]
        for _ in range(self.L):
            Eu.append(spmm(self.Rn, Ei[-1])); Ei.append(spmm(self.RnT, Eu[-2]))
            Gu.append(self.us @ (self.vt @ Ei[-2])); Gi.append(self.vs @ (self.ut @ Eu[-2]))
        return sum(Eu), sum(Ei), sum(Gu), sum(Gi)

    def loss(self, u, i, j):
        Eu, Ei, Gu, Gi = self.forward_all(); tau = self.a.get('tau', 0.2)
        uu, ii = torch.unique(u), torch.unique(i)
        neg = torch.logsumexp(Gu[uu] @ Eu.T / tau, 1).mean() + torch.logsumexp(Gi[ii] @ Ei.T / tau, 1).mean()
        pos = ((Gu[uu] * Eu[uu]).sum(1) / tau).clamp(-5, 5).mean() + ((Gi[ii] * Ei[ii]).sum(1) / tau).clamp(-5, 5).mean()
        l = bpr((Eu[u] * Ei[i]).sum(1), (Eu[u] * Ei[j]).sum(1))
        reg = self.a.get('lam2', 1e-7) * (self.Eu.pow(2).sum() + self.Ei.pow(2).sum())
        return l + self.a.get('lam1', 0.2) * (neg - pos) + reg

    @torch.no_grad()
    def scorer(self):
        Eu, Ei, _, _ = self.forward_all()
        return lambda ub: Eu[ub] @ Ei.T


class HCCF(Base):
    """Xia et al. SIGIR'22, following the authors' PyTorch code."""
    max_epochs = 150

    def build(self):
        self.A = self.data.norm_adj()
        H = self.a.get('H', 128)
        self.E = xavier(self.data.n, self.d, self.gen)
        self.uH = xavier(self.d, H, self.gen); self.iH = xavier(self.d, H, self.gen)

    def forward_all(self, keep):
        nu = self.nu
        if keep < 1:
            v = self.A.values(); m = (torch.rand(v.shape, device=DEV) < keep).float() / keep
            A = torch.sparse_coo_tensor(self.A.indices(), v * m, self.A.shape).coalesce()
        else:
            A = self.A
        uu = self.E[:nu] @ self.uH; ii = self.E[nu:] @ self.iH
        if keep < 1:
            uu = F.dropout(uu, 1 - keep); ii = F.dropout(ii, 1 - keep)
        lats = [self.E]; g, h = [], []
        for _ in range(self.L):
            t = spmm(A, lats[-1])
            hh = torch.cat([uu @ (uu.T @ lats[-1][:nu]), ii @ (ii.T @ lats[-1][nu:])], 0)
            g.append(t); h.append(hh); lats.append(t + hh)
        return sum(lats), g, h

    def loss(self, u, i, j):
        X, g, h = self.forward_all(self.a.get('keep', 0.5))
        eu, ei, ej = X[u], X[self.nu + i], X[self.nu + j]
        l = bpr((eu * ei).sum(1), (eu * ej).sum(1)); tau = self.a.get('tau', 1.0)
        ssl = sum(self.cl_pair(a_.detach(), b_, u, i, tau) for a_, b_ in zip(g, h))
        reg = self.a.get('reg_all', 1e-7) * sum(p.pow(2).sum() for p in self.parameters())
        return l + self.a.get('ssl', 1e-3) * ssl + reg

    @torch.no_grad()
    def scorer(self):
        return self.dot_scorer(self.forward_all(1.0)[0])


class SEPT(Base):
    """Yu et al. KDD'21, following SELFRec (ss_rate 0.005, drop 0.3, 10 pseudo-positives, warm-up)."""
    max_epochs = 150

    def build(self):
        d = self.data; self.A = d.norm_adj(); S = d.S; I = sp.eye(d.nu, dtype=np.float32, format='csr')
        self.soc = sp_to_torch(sym_norm((S @ S).multiply(S) + I)); self.shr = sp_to_torch(sym_norm((d.R @ d.R.T).multiply(S) + I))
        self.warm = self.a.get('warm', 10)
        self.E = xavier(d.n, self.d, self.gen)

    def enc(self, E, A):
        out = E; X = E
        for _ in range(self.L):
            X = F.normalize(spmm(A, X), dim=1); out = out + X
        return out

    def before_epoch(self, ep):
        self.epoch = ep
        if ep >= self.warm:
            self.aug = self.data.norm_adj(drop_edges(self.data.R, self.a.get('drop', 0.3), self.rng))

    def loss(self, u, i, j):
        X = self.enc(self.E, self.A); eu, ei, ej = X[u], X[self.nu + i], X[self.nu + j]
        l = bpr((eu * ei).sum(1), (eu * ej).sum(1)) + self.reg * 0.5 * self.E.pow(2).sum() / len(u)
        if self.epoch < self.warm:
            return l
        uu = torch.unique(u)
        aug = F.normalize(self.enc(self.E, self.aug)[uu], dim=1)
        fr = self.enc(self.E[:self.nu], self.soc); sh = self.enc(self.E[:self.nu], self.shr)
        views = [F.normalize(fr[uu], dim=1), F.normalize(sh[uu], dim=1), F.normalize(X[uu], dim=1)]
        with torch.no_grad():
            pr = [torch.softmax(v @ aug.T, 1) for v in views]
        K = self.a.get('ins', 10); ssl = 0
        for k in range(3):
            o = [x for x in range(3) if x != k]
            pos_idx = torch.topk((pr[o[0]] + pr[o[1]]) / 2, K, dim=1).indices
            v = views[k]
            pos = (v[:, None, :] * aug[pos_idx]).sum(2)
            ssl = ssl - (torch.logsumexp(pos / 0.1, 1) - torch.logsumexp(v @ aug.T / 0.1, 1)).sum()
        # SELFRec sums the BPR loss over the batch; our BPR is a batch mean, so the (summed) SSL term is
        # divided by the batch size to keep the ratio of the two terms identical to the reference code
        return l + self.a.get('ss', 0.005) * ssl / len(u)

    @torch.no_grad()
    def scorer(self):
        return self.dot_scorer(self.enc(self.E, self.A))


# ============================================================================ non-graph baselines
class NCF(Base):
    """NeuMF: GMF + MLP[128-64-32-16], log loss with 4 sampled negatives."""
    n_neg = 4; max_epochs = 100

    def build(self):
        g, d = self.gen, self.d
        self.Pg, self.Qg, self.Pm, self.Qm = xavier(self.nu, d, g), xavier(self.ni, d, g), xavier(self.nu, d, g), xavier(self.ni, d, g)
        self.mlp = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()).to(DEV)
        self.h = nn.Linear(d + 16, 1, bias=False).to(DEV)

    def logit(self, u, i):
        return self.h(torch.cat([self.Pg[u] * self.Qg[i], self.mlp(torch.cat([self.Pm[u], self.Qm[i]], 1))], 1)).squeeze(1)

    def loss(self, u, i, j):
        up, ip = u[::self.n_neg], i[::self.n_neg]
        pos, neg = self.logit(up, ip), self.logit(u, j)
        l = (F.softplus(-pos).sum() + F.softplus(neg).sum()) / (len(up) + len(u))
        return l + self.reg * (self.Pg[u].pow(2).sum() + self.Pm[u].pow(2).sum()) / len(u)

    @torch.no_grad()
    def scorer(self):
        d = self.d; W1 = self.mlp[0]
        Ai = self.Qm @ W1.weight[:, d:].T; G = self.Qg * self.h.weight[0, :d]
        def f(ub):
            out = torch.empty(len(ub), self.ni, device=DEV)
            Au = self.Pm[ub] @ W1.weight[:, :d].T + W1.bias
            for a in range(0, len(ub), 64):
                h1 = torch.relu(Au[a:a + 64, None, :] + Ai[None]); h3 = self.mlp[2:](h1)
                out[a:a + 64] = (h3 @ self.h.weight[0, d:]) + self.Pg[ub[a:a + 64]] @ G.T
            return out
        return f


class DeepFM(Base):
    """DeepFM with fields (user, item, item category); category embedding = category-weighted mean."""
    n_neg = 4; max_epochs = 100

    def build(self):
        g, d = self.gen, self.d; nc = self.data.catw.shape[1]
        self.C = sp_to_torch(self.data.catw)
        self.V = xavier(self.data.n, d, g); self.Vc = xavier(nc, d, g)
        self.w = nn.Parameter(torch.zeros(self.data.n, device=DEV)); self.wc = nn.Parameter(torch.zeros(nc, 1, device=DEV))
        self.b = nn.Parameter(torch.zeros(1, device=DEV))
        self.mlp = nn.Sequential(nn.Linear(3 * d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).to(DEV)

    def item_side(self):
        return spmm(self.C, self.Vc), spmm(self.C, self.wc)[:, 0]

    def logit(self, u, i, vc, wc):
        vu, vi, c = self.V[u], self.V[self.nu + i], vc[i]
        fm = self.w[u] + self.w[self.nu + i] + wc[i] + self.b + (vu * vi + vu * c + vi * c).sum(1)
        return fm + self.mlp(torch.cat([vu, vi, c], 1)).squeeze(1)

    def loss(self, u, i, j):
        vc, wc = self.item_side(); up, ip = u[::self.n_neg], i[::self.n_neg]
        pos, neg = self.logit(up, ip, vc, wc), self.logit(u, j, vc, wc)
        l = (F.softplus(-pos).sum() + F.softplus(neg).sum()) / (len(up) + len(u))
        return l + self.reg * self.V[u].pow(2).sum() / len(u)

    @torch.no_grad()
    def scorer(self):
        d = self.d; vc, wc = self.item_side(); Vi = self.V[self.nu:]; W1 = self.mlp[0]
        Ai = Vi @ W1.weight[:, d:2 * d].T + vc @ W1.weight[:, 2 * d:].T
        const = self.w[self.nu:] + wc + (Vi * vc).sum(1); Si = Vi + vc
        def f(ub):
            out = torch.empty(len(ub), self.ni, device=DEV); Vu = self.V[ub]
            Au = Vu @ W1.weight[:, :d].T + W1.bias
            for a in range(0, len(ub), 64):
                h = self.mlp[2:](torch.relu(Au[a:a + 64, None, :] + Ai[None])).squeeze(-1)
                out[a:a + 64] = h + Vu[a:a + 64] @ Si.T + const + self.w[ub[a:a + 64], None] + self.b
            return out
        return f


def _topk_sparse(Sim, k):
    Sim = Sim.tocsr(); Sim.setdiag(0); Sim.eliminate_zeros()
    rows, cols, vals = [], [], []
    for r in range(Sim.shape[0]):
        a, b = Sim.indptr[r], Sim.indptr[r + 1]
        if b > a:
            v, c = Sim.data[a:b], Sim.indices[a:b]
            if len(v) > k:
                t = np.argpartition(-v, k)[:k]; v, c = v[t], c[t]
            rows.append(np.full(len(v), r)); cols.append(c); vals.append(v)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=Sim.shape)


class UserCF:
    trainable = False
    def __init__(self, data, a, gen, rng): self.data, self.k = data, a.get('k', 50)
    def fit(self):
        R = self.data.R; n = np.sqrt(np.asarray(R.multiply(R).sum(1)).ravel()) + 1e-12
        Rn = sp.diags(1 / n) @ R; self.S = _topk_sparse(Rn @ Rn.T, self.k)
    def scorer(self):
        return lambda ub: torch.from_numpy((self.S[ub.cpu().numpy()] @ self.data.R).toarray()).to(DEV)


class ItemCF(UserCF):
    def fit(self):
        R = self.data.R; n = np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel()) + 1e-12
        Rn = R @ sp.diags(1 / n); self.S = _topk_sparse(Rn.T @ Rn, self.k).T.tocsr()
    def scorer(self):
        return lambda ub: torch.from_numpy((self.data.R[ub.cpu().numpy()] @ self.S).toarray()).to(DEV)


# ============================================================================ SGC-Rec
def raw_features(data):
    """Per-node raw features from TRAINING data only.
    user (d0=44): #interactions, social degree, mean popularity of its items, #distinct categories, 40-bin category profile
    item (d0=43): popularity, mean activity of its users, mean social degree of its users, 40-bin category indicator"""
    R, S, C = data.R, data.S, data.catw
    top = np.argsort(-np.asarray(R.sum(0) @ C).ravel())[:39]
    M = np.zeros((C.shape[1], 40), np.float32); M[:, 39] = 1; M[top, 39] = 0; M[top, np.arange(len(top))] = 1
    C40 = np.asarray(C @ M)
    ucnt = np.asarray(R.sum(1)).ravel(); icnt = np.asarray(R.sum(0)).ravel(); sdeg = np.asarray(S.sum(1)).ravel()
    uprof = np.asarray(R @ C40)
    uf = np.c_[ucnt, sdeg, (R @ icnt) / np.maximum(ucnt, 1), (uprof > 0).sum(1), uprof]
    RT = R.T.tocsr()
    itf = np.c_[icnt, (RT @ ucnt) / np.maximum(icnt, 1), (RT @ sdeg) / np.maximum(icnt, 1), C40]
    return uf.astype(np.float64), itf.astype(np.float64)


def normalise(X, how):
    if how == 'cdf':
        out = np.empty_like(X)
        for c in range(X.shape[1]):
            s = np.sort(X[:, c]); out[:, c] = np.searchsorted(s, X[:, c], side='right') / len(s)
        X = out
    elif how == 'log':
        X = np.log1p(X)
    elif how == 'z':
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return torch.from_numpy(X.astype(np.float32)).to(DEV)


class Head(nn.Module):
    """Amortised Gaussian-mixture head."""
    def __init__(self, d0, d, C, cov, gen):
        super().__init__()
        vd = d if cov == 'diag' else 1
        self.d, self.C, self.vd = d, C, vd
        self.Wmu, self.Wlv, self.Wpi = xavier(d0, C * d, gen), xavier(d0, C * vd, gen), xavier(d0, C, gen)
        self.bmu = nn.Parameter(torch.zeros(C * d, device=DEV)); self.blv = nn.Parameter(torch.zeros(C * vd, device=DEV))
        self.bpi = nn.Parameter(torch.zeros(C, device=DEV))

    def forward(self, X):
        n = X.shape[0]
        mu = (X @ self.Wmu + self.bmu).view(n, self.C, self.d)
        lv = (X @ self.Wlv + self.blv).view(n, self.C, self.vd).clamp(-8, 8)
        pi = torch.softmax(X @ self.Wpi + self.bpi, 1)[:, :, None]
        m = (pi * mu).sum(1)
        var = (pi * (lv.exp() + mu ** 2)).sum(1) - m ** 2          # law of total variance
        return m, var.clamp_min(1e-4).expand(n, self.d)


class SGCRec(Base):
    max_epochs = 200
    DEF = dict(C=3, gamma=0.5, eta=0.1, tau=0.11, lam1=0.4, lam2=0.2, beta_kl=1e-3, norm='cdf', cov='diag',
               gauss=True, uattn=True, varaug=True, dscore=True, pre_epochs=25, attn_drop=0.1)

    def build(self):
        o = dict(self.DEF); o.update(self.a); self.o = o
        data = self.data; d = self.d
        self.A1 = data.norm_adj(); self.A2 = data.norm_adj(S=data.S)
        def rownorm(M):
            B = (M != 0).astype(np.float32); s = np.asarray(B.sum(1)).ravel()
            return sp_to_torch(sp.diags(np.where(s > 0, 1 / np.maximum(s, 1), 0).astype(np.float32)) @ B)
        self.P1, self.P2 = rownorm(data.joint()), rownorm(data.joint(S=data.S))
        self.E = xavier(data.n, d, self.gen)
        self.Wa = xavier(d, d, self.gen); self.ba = nn.Parameter(torch.zeros(d, device=DEV))
        self.q = nn.Parameter(xavier(d, 1, self.gen).data[:, 0].clone())
        if o['gauss']:
            uf, itf = raw_features(data)
            self.Xu, self.Xi = normalise(uf, o['norm']), normalise(itf, o['norm'])
            self.hu = Head(self.Xu.shape[1], d, o['C'], o['cov'], self.gen)
            self.hi = Head(self.Xi.shape[1], d, o['C'], o['cov'], self.gen)

    def all_gauss(self):
        mu_u, v_u = self.hu(self.Xu); mu_i, v_i = self.hi(self.Xi)
        return torch.cat([mu_u, mu_i], 0), torch.cat([v_u, v_i], 0)

    def pretrain(self):
        """Heads trained to reconstruct the joint adjacency, 25 epochs."""
        o = self.o
        if not o['gauss']:
            return
        data = self.data; nu = self.nu; d = self.d
        src = torch.from_numpy(np.r_[data.tr_u, data.social[:, 0]]).to(DEV)
        dst = torch.from_numpy(np.r_[data.tr_i + nu, data.social[:, 1]]).to(DEV)
        params = list(self.hu.parameters()) + list(self.hi.parameters())
        opt = torch.optim.Adam(params, lr=self.a.get('lr', 1e-3))
        B = 4096
        for ep in range(o['pre_epochs']):
            perm = torch.randperm(len(src), device=DEV, generator=self.gen)
            for s in range(0, len(perm), B):
                e = perm[s:s + B]; k = len(e)
                a_ = torch.cat([src[e], torch.randint(0, data.n, (k,), device=DEV, generator=self.gen)])
                b_ = torch.cat([dst[e], torch.randint(0, data.n, (k,), device=DEV, generator=self.gen)])
                y = torch.cat([torch.ones(k, device=DEV), torch.zeros(k, device=DEV)])
                mu, var = self.all_gauss()
                z = mu + var.sqrt() * torch.randn_like(mu)
                logit = (z[a_] * z[b_]).sum(1) / math.sqrt(d)
                nodes = torch.unique(torch.cat([a_, b_]))
                kl = 0.5 * (var[nodes] + mu[nodes] ** 2 - 1 - var[nodes].log()).sum(1).mean()
                loss = F.binary_cross_entropy_with_logits(logit, y) + o['beta_kl'] * kl
                opt.zero_grad(); loss.backward(); opt.step()
        if self.o.get('init', 'mu') == 'mu':
            with torch.no_grad():
                self.E.data.copy_(self.all_gauss()[0])  # embedding table initialised from the pre-trained means

    def views(self):
        return [propagate(self.E, A, self.L) for A in (self.A1, self.A2)]

    def fuse(self, V, nodes, ubar, train):
        logits = []
        for r, e in enumerate(V):
            l = torch.tanh(e[nodes] @ self.Wa + self.ba) @ self.q
            if ubar is not None and self.o['uattn']:
                l = l - self.o['gamma'] * ubar[r][nodes]
            if train and self.o['attn_drop'] > 0:
                l = F.dropout(l, self.o['attn_drop'])
            logits.append(l)
        al = torch.softmax(torch.stack(logits, 1), 1)
        return sum(al[:, r:r + 1] * e[nodes] for r, e in enumerate(V))

    def score(self, eu, vu, ei, vi):
        if self.o['gauss'] and self.o['dscore']:
            v = vu + vi
            return -(((eu - ei) ** 2) / v + v.log()).sum(1)
        return (eu * ei).sum(1)

    def loss(self, u, i, j):
        o = self.o; nu = self.nu; d = self.d
        V = self.views(); ubar = None
        if o['gauss']:
            mu, var = self.all_gauss(); sd = var.sqrt(); s = sd.mean(1, keepdim=True)
            ubar = [spmm(self.P1, s)[:, 0], spmm(self.P2, s)[:, 0]]       # gradient flows to the heads
        nodes = torch.cat([u, nu + i, nu + j]); B = len(u)
        e = self.fuse(V, nodes, ubar, True); eu, ei, ej = e[:B], e[B:2 * B], e[2 * B:]
        vu = vi = vj = None
        if o['gauss']:
            vu, vi, vj = var[u], var[nu + i], var[nu + j]
        l = bpr(self.score(eu, vu, ei, vi), self.score(eu, vu, ej, vj)) + self.reg_loss(self.E, u, i, j)
        uu, ii = torch.unique(u), torch.unique(i) + nu; bn = torch.cat([uu, ii])
        if o['gauss']:
            l = l + o['lam1'] * ((self.E[bn] - mu[bn]) ** 2).mean()
            l = l + o['beta_kl'] * 0.5 * (var[bn] + mu[bn] ** 2 - 1 - var[bn].log()).sum(1).mean()
            l = l + self.reg * sum(p.pow(2).sum() for n_, p in self.named_parameters() if n_.split('.')[-1].startswith('W'))
        if o['lam2'] > 0:
            cl = 0
            for Vr in V:
                for grp in (uu, ii):
                    if o['gauss'] and o['varaug']:
                        bgt = sd[grp]                                        # node-specific budget
                    elif o['gauss']:
                        bgt = sd.detach().mean()                             # dataset-level mean (N2 removed)
                    else:
                        bgt = 1.0
                    vv = [Vr[grp] + o['eta'] * bgt * F.normalize(torch.randn(len(grp), d, device=DEV), dim=1) for _ in range(2)]
                    cl = cl + 0.5 * info_nce(vv[0], vv[1], o['tau'])
            l = l + o['lam2'] * cl
        return l

    @torch.no_grad()
    def final(self):
        V = self.views(); nodes = torch.arange(self.data.n, device=DEV)
        ubar = var = None
        if self.o['gauss']:
            mu, var = self.all_gauss(); s = var.sqrt().mean(1, keepdim=True)
            ubar = [spmm(self.P1, s)[:, 0], spmm(self.P2, s)[:, 0]]
        return self.fuse(V, nodes, ubar, False), var

    @torch.no_grad()
    def scorer(self):
        e, var = self.final(); nu = self.nu; U, I = e[:nu], e[nu:]
        if self.o['gauss'] and self.o['dscore']:
            vU, vI = var[:nu], var[nu:]
            return lambda ub: dist_score(U[ub], vU[ub], I, vI)
        return lambda ub: U[ub] @ I.T


class SGCRec2(SGCRec):
    """Single-encoder SGC-Rec variant (not used in the reported experiments). The learned node-specific diagonal covariance is shared by the three operators,
    but (i) the embedding table is Xavier-initialised and tied to the Gaussian prior by a scale-free (cosine)
    consistency term, (ii) the variance-scaled perturbation is injected at every propagation layer and the
    contrastive pair is the final vs. the first-layer representation of the same forward pass, and (iii) the
    ranking score is an uncertainty-weighted inner product  sum_c w_c e_uc e_ic,  w_c = (s / (s_uc^2 + s_ic^2))^omega
    (omega = 0 recovers the plain inner product)."""
    DEF = dict(SGCRec.DEF, init='xavier', lam1=0.1, lam2=0.2, eta=0.2, tau=0.2, omega=0.5, lstar=1, ego=1)

    def build(self):
        super().build()
        self.vbar = None
        self.n_neg = int(self.o.get('n_neg', 1))

    def budget(self, sd):
        o = self.o
        if o['gauss'] and o['varaug']:
            return sd / sd.mean().detach()            # node- and dimension-specific budget, mean 1
        return 1.0

    def noisy_views(self, sd, train):
        out, cls = [], []
        bgt = self.budget(sd) if (train and sd is not None) else 1.0
        for A in (self.A1, self.A2):
            X = self.E; layers = [X] if self.o['ego'] else []; kept = None
            for l in range(self.L):
                X = spmm(A, X)
                if train and self.o['lam2'] > 0:
                    X = X + self.o['eta'] * bgt * torch.sign(X) * F.normalize(torch.rand_like(X), dim=-1)
                layers.append(X)
                if l == self.o['lstar'] - 1:
                    kept = X
            out.append(torch.stack(layers, 0).mean(0)); cls.append(kept)
        return out, cls

    def conf(self, sd, idx, kappa):
        """detached confidence weights (s_bar / s_a)^kappa, normalised to mean 1 over the batch"""
        s = sd.detach().mean(1)
        w = (s.mean() / s[idx]) ** kappa
        return w / w.mean()

    def nce(self, v1, v2, sd, g):
        """InfoNCE; optional uncertainty-weighted anchors (uw_cl) and node-adaptive temperature (utau)"""
        o = self.o; tau = o['tau']
        v1, v2 = F.normalize(v1, dim=1), F.normalize(v2, dim=1)
        t = torch.full((len(g),), tau, device=DEV)
        if sd is not None and o.get('utau', 0) > 0:
            s = sd.detach().mean(1); t = tau * (s[g] / s.mean()) ** o['utau']
        logits = (v1 @ v2.T) / t[:, None]
        per = torch.logsumexp(logits, 1) - (v1 * v2).sum(1) / t
        if sd is not None and o.get('uw_cl', 0) > 0:
            return (self.conf(sd, g, o['uw_cl']) * per).mean()
        return per.mean()

    def weights(self, vu, vi):
        v = vu + vi
        return (self.vbar / v) ** self.o['omega']

    def score(self, eu, vu, ei, vi):
        if self.o['gauss'] and self.o['omega'] > 0:
            return (self.weights(vu, vi) * eu * ei).sum(1)
        return (eu * ei).sum(1)

    def loss(self, u, i, j):
        o = self.o; nu = self.nu
        sd = var = ubar = None
        if o['gauss']:
            mu, var = self.all_gauss(); sd = var.sqrt(); s = sd.mean(1, keepdim=True)
            ubar = [spmm(self.P1, s)[:, 0], spmm(self.P2, s)[:, 0]]
            self.vbar = 2 * var.mean().detach()
        V, Vcl = self.noisy_views(sd, True)
        nodes = torch.cat([u, nu + i, nu + j]); B = len(u)
        e = self.fuse(V, nodes, ubar, True); eu, ei, ej = e[:B], e[B:2 * B], e[2 * B:]
        vu = vi = vj = None
        if o['gauss']:
            vu, vi, vj = var[u], var[nu + i], var[nu + j]
        if o.get('rec', 'bpr') == 'ssm':
            # sampled softmax over in-batch items (cosine similarity / temperature tau_r)
            up, ip = eu[::self.n_neg], ei[::self.n_neg]
            logits = F.normalize(up, dim=1) @ F.normalize(ip, dim=1).T / o.get('tau_r', 0.1)
            l = F.cross_entropy(logits, torch.arange(len(up), device=DEV)) + self.reg_loss(self.E, u, i, j)
        else:
            if o['gauss'] and o.get('uw_bpr', 0) > 0:          # confidence-weighted BPR (weights detached, mean 1)
                wu = self.conf(sd, u, o['uw_bpr'])
                l = (wu * F.softplus(-(self.score(eu, vu, ei, vi) - self.score(eu, vu, ej, vj)))).mean() + self.reg_loss(self.E, u, i, j)
            else:
                l = bpr(self.score(eu, vu, ei, vi), self.score(eu, vu, ej, vj)) + self.reg_loss(self.E, u, i, j)
        uu, ii = torch.unique(u), torch.unique(i) + nu; bn = torch.cat([uu, ii])
        if o['gauss']:
            if o['lam1'] > 0:
                l = l + o['lam1'] * (1 - F.cosine_similarity(self.E[bn], mu[bn], dim=1)).mean()
            l = l + o['beta_kl'] * 0.5 * (var[bn] + mu[bn] ** 2 - 1 - var[bn].log()).sum(1).mean()
            l = l + self.reg * sum(p.pow(2).sum() for n_, p in self.named_parameters() if n_.split('.')[-1].startswith('W'))
        if o['lam2'] > 0:
            cl = sum(self.nce(Vr[g], Vc[g], sd, g) for Vr, Vc in zip(V, Vcl) for g in (uu, ii))
            if o.get('xview', 0) > 0:                            # cross-view (interaction vs social-aware) contrast, users
                cl = cl + o['xview'] / o['lam2'] * self.nce(V[0][uu], V[1][uu], sd, uu)
            l = l + o['lam2'] * cl
        return l

    @torch.no_grad()
    def final(self):
        sd = var = ubar = None
        if self.o['gauss']:
            mu, var = self.all_gauss(); sd = var.sqrt(); s = sd.mean(1, keepdim=True)
            ubar = [spmm(self.P1, s)[:, 0], spmm(self.P2, s)[:, 0]]
            self.vbar = 2 * var.mean()
        V, _ = self.noisy_views(sd, False)
        return self.fuse(V, torch.arange(self.data.n, device=DEV), ubar, False), var

    @torch.no_grad()
    def scorer(self):
        e, var = self.final(); nu = self.nu; U, I = e[:nu], e[nu:]
        if self.o['gauss'] and self.o['omega'] > 0:
            vU, vI = var[:nu], var[nu:]
            def f(ub, chunk=64):
                out = torch.empty(len(ub), self.ni, device=DEV)
                for a in range(0, len(ub), chunk):
                    b = ub[a:a + chunk]
                    w = (self.vbar / (vU[b][:, None, :] + vI[None])) ** self.o['omega']
                    out[a:a + chunk] = (w * U[b][:, None, :] * I[None]).sum(-1)
                return out
            return f
        if self.o.get('rec', 'bpr') == 'ssm':
            U, I = F.normalize(U, dim=1), F.normalize(I, dim=1)
        return lambda ub: U[ub] @ I.T


MODELS = dict(LightGCN=LightGCN, SGL=SGL, SimGCL=SimGCL, XSimGCL=XSimGCL, LightGCL=LightGCL, HCCF=HCCF, SEPT=SEPT,
              NCF=NCF, DeepFM=DeepFM, UserCF=UserCF, ItemCF=ItemCF, SGCRec=SGCRec, SGCRec2=SGCRec2)
