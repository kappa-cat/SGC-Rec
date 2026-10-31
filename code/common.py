"""Shared utilities (PyTorch / CUDA): data, graphs, sampler, evaluation."""
import os, math, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

# Make seed-matched runs reproducible across CPU scheduling and parallel jobs.
torch.set_num_threads(int(os.environ.get('SGC_TORCH_THREADS', '1')))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.use_deterministic_algorithms(True, warn_only=True)

HERE = Path(__file__).resolve().parents[1]
DEV = torch.device(os.environ.get('SGC_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))


def sp_to_torch(M):
    M = M.tocoo()
    idx = torch.from_numpy(np.vstack([M.row, M.col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(M.data.astype(np.float32)), M.shape).coalesce().to(DEV)


def sym_norm(A):
    A = A.tocsr().astype(np.float32)
    d = np.asarray(A.sum(1)).ravel()
    with np.errstate(divide='ignore'):
        dinv = np.where(d > 0, d ** -0.5, 0).astype(np.float32)
    return sp.diags(dinv) @ A @ sp.diags(dinv)


class Data:
    def __init__(self, name, seed, noise=0.0, keep=1.0):
        z = np.load(HERE / 'data' / f'{name}.npz', allow_pickle=True)
        self.name, self.seed = name, seed
        self.nu, self.ni = int(z['nu']), int(z['ni']); self.n = self.nu + self.ni
        self.catw = sp.csr_matrix((z['cat_data'], z['cat_indices'], z['cat_indptr']), shape=(self.ni, int(z['ncat'])))
        u, i, tag = z[f'u_{seed}'].astype(np.int64), z[f'i_{seed}'].astype(np.int64), z[f'tag_{seed}']
        self.social = z[f'social_{seed}'].astype(np.int64)
        tr = tag == 0
        self.tr_u, self.tr_i = u[tr], i[tr]
        if keep < 1.0:  # sparse-data regime: random subset of the TRAINING interactions
            rs = np.random.default_rng(20_000 + seed).random(len(self.tr_u)) < keep
            self.tr_u, self.tr_i = self.tr_u[rs], self.tr_i[rs]
        if noise > 0:  # RQ3: random user-item edges injected into the TRAINING graph only
            rng = np.random.default_rng(10_000 + seed)
            k = int(round(noise * len(self.tr_u)))
            key = np.unique(self.tr_u * self.ni + self.tr_i)
            cand = np.unique(rng.integers(0, self.nu, 3 * k) * self.ni + rng.integers(0, self.ni, 3 * k))
            cand = rng.permutation(cand[~np.isin(cand, key)])[:k]
            self.tr_u = np.r_[self.tr_u, cand // self.ni]; self.tr_i = np.r_[self.tr_i, cand % self.ni]
        self.R = sp.csr_matrix((np.ones(len(self.tr_u), np.float32), (self.tr_u, self.tr_i)), shape=(self.nu, self.ni))
        self.R.sum_duplicates(); self.R.data[:] = 1
        mk = lambda m: sp.csr_matrix((np.ones(m.sum(), np.float32), (u[m], i[m])), shape=(self.nu, self.ni))
        self.val, self.test = mk(tag == 1), mk(tag == 2)
        s = self.social
        S = sp.csr_matrix((np.ones(2 * len(s), np.float32), (np.r_[s[:, 0], s[:, 1]], np.r_[s[:, 1], s[:, 0]])), shape=(self.nu, self.nu))
        S.data[:] = 1; self.S = S
        self.train_key = torch.from_numpy(np.sort(self.tr_u * self.ni + self.tr_i)).to(DEV)
        self.tu = torch.from_numpy(self.tr_u).to(DEV); self.ti = torch.from_numpy(self.tr_i).to(DEV)

    def drop_social(self):
        """remove every friendship edge (ablation): no mixing, zero social degree"""
        self.social = np.zeros((0, 2), np.int64); self.S = sp.csr_matrix((self.nu, self.nu), dtype=np.float32)

    def joint(self, R=None, S=None):
        R = self.R if R is None else R
        Su = S if S is not None else sp.csr_matrix((self.nu, self.nu), dtype=np.float32)
        return sp.vstack([sp.hstack([Su, R]), sp.hstack([R.T, sp.csr_matrix((self.ni, self.ni), dtype=np.float32)])]).tocsr()

    def norm_adj(self, R=None, S=None):
        return sp_to_torch(sym_norm(self.joint(R, S)))

    def sample_epoch(self, gen, n_neg=1):
        perm = torch.randperm(len(self.tu), device=DEV, generator=gen)
        u = self.tu[perm].repeat_interleave(n_neg); i = self.ti[perm].repeat_interleave(n_neg)
        j = torch.randint(0, self.ni, (len(u),), device=DEV, generator=gen)
        for _ in range(100):
            key = u * self.ni + j
            pos = torch.searchsorted(self.train_key, key).clamp(max=len(self.train_key) - 1)
            bad = self.train_key[pos] == key
            if not bad.any():
                break
            j[bad] = torch.randint(0, self.ni, (int(bad.sum()),), device=DEV, generator=gen)
        return u, i, j


def xavier(n, m, gen):
    b = math.sqrt(6.0 / (n + m))
    return torch.nn.Parameter((torch.rand(n, m, device=DEV, generator=gen) * 2 - 1) * b)


def info_nce(v1, v2, tau):
    v1, v2 = F.normalize(v1, dim=1), F.normalize(v2, dim=1)
    logits = v1 @ v2.T / tau
    return (torch.logsumexp(logits, 1) - (v1 * v2).sum(1) / tau).mean()


def bpr(pos, neg):
    return F.softplus(-(pos - neg)).mean()


def dist_score(eu, su, ei, si, chunk=64):
    """Expected-likelihood score for all (user, item) pairs; su/si are variances."""
    out = torch.empty(eu.shape[0], ei.shape[0], device=DEV)
    for a in range(0, eu.shape[0], chunk):
        v = su[a:a + chunk, None, :] + si[None]
        out[a:a + chunk] = -(((eu[a:a + chunk, None, :] - ei[None]) ** 2) / v + torch.log(v)).sum(-1)
    return out


@torch.no_grad()
def evaluate(data, scorer, split='test', K=10, chunk=1024, users=None):
    target = data.test if split == 'test' else data.val
    excl = (data.R + data.val).tocsr() if split == 'test' else data.R
    cnt = np.diff(target.indptr)
    us = np.nonzero(cnt > 0)[0] if users is None else users
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    P, Rc, N, tops = [], [], [], []
    for a in range(0, len(us), chunk):
        ub = us[a:a + chunk]
        sc = scorer(torch.from_numpy(ub).to(DEV)).float()
        ex = excl[ub].tocoo()
        sc[torch.from_numpy(ex.row).to(DEV).long(), torch.from_numpy(ex.col).to(DEV).long()] = -float('inf')
        top = torch.topk(sc, K, dim=1).indices.cpu().numpy()
        hit = np.asarray(target[ub][np.repeat(np.arange(len(ub)), K), top.reshape(-1)]).reshape(len(ub), K) > 0
        nt = cnt[ub]
        P.append(hit.sum(1) / K); Rc.append(hit.sum(1) / nt)
        idcg = np.array([disc[:min(k, K)].sum() for k in nt])
        N.append((hit * disc).sum(1) / idcg); tops.append(top)
    return dict(users=us, P=np.concatenate(P), R=np.concatenate(Rc), N=np.concatenate(N), top=np.concatenate(tops))


def entropy(data, top):
    cnt = np.bincount(top.ravel(), minlength=data.ni).astype(np.float64)
    p = np.asarray(data.catw.T @ cnt).ravel(); p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())
