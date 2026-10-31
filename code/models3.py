"""Model components for single-encoder SGC-Rec variants (Gaussian heads, uncertainty-gated view attention, variance-scaled
perturbation, distribution-aware ranking) and the item-item co-occurrence block used by models4.py.
Also defines XSimGCL3, an XSimGCL control with the same loss and negative operators and no uncertainty."""
import torch
import torch.nn.functional as F
import models
from models import SGCRec2, XSimGCL, noisy_prop, bpr, info_nce
from common import DEV


def pick_negatives(model, u, cand, su, ci, temp):
    data = model.data
    key = u[:, None] * data.ni + cand
    pos = torch.searchsorted(data.train_key, key.reshape(-1)).clamp(max=len(data.train_key) - 1).view_as(key)
    sc = (su[:, None, :] * ci).sum(-1).masked_fill(data.train_key[pos] == key, -1e9)
    k = sc.argmax(1) if temp is None else torch.multinomial(torch.softmax(sc / temp[:, None], 1), 1, generator=model.gen)[:, 0]
    return cand.gather(1, k[:, None])[:, 0]


def ssm_loss(model, eu, ei, u, K, tau):
    """sampled softmax with K shared uniformly-sampled negatives, cosine similarity; tau: float or (B,) tensor"""
    J = torch.randint(0, model.ni, (K,), device=DEV, generator=model.gen)
    return ssm_logits_loss(eu, ei, model.item_rep(J), tau)


def ssm_logits_loss(eu, ei, ej, tau):
    eu, ei, ej = F.normalize(eu, dim=1), F.normalize(ei, dim=1), F.normalize(ej, dim=1)
    t = tau if torch.is_tensor(tau) else torch.full((len(eu),), float(tau), device=DEV)
    pos = (eu * ei).sum(1) / t
    neg = (eu @ ej.T) / t[:, None]
    return (torch.logsumexp(torch.cat([pos[:, None], neg], 1), 1) - pos).mean()


class SGCRec3(SGCRec2):
    DEF = dict(SGCRec2.DEF, neg='uni', M=8, T0=1.0, kappa=0.0, rec='bpr', K=1024, tau_r=0.1, kappa_r=0.0,
               views=12, fusion='attn', warm=0, prior='none', prior_r=1.0, prior_learn=1)

    def pretrain(self):
        super().pretrain()
        if self.o['prior'] == 'add' and self.o['gauss']:
            # layer-0 embedding = free residual + alpha * Gaussian prior mean mu(x);  alpha matched to the table scale
            with torch.no_grad():
                mu, _ = self.all_gauss()
                a0 = self.o['prior_r'] * self.E.norm(dim=1).mean() / mu.norm(dim=1).mean().clamp_min(1e-8)
            self.alpha = torch.nn.Parameter(torch.tensor(float(a0), device=DEV), requires_grad=bool(self.o['prior_learn']))

    def ego(self):
        if self.o['prior'] == 'add' and self.o['gauss']:
            mu, _ = self.all_gauss()
            return self.E + self.alpha * mu
        return self.E

    def before_epoch(self, ep):
        # optional linear warm-up of the contrastive weight and of the perturbation budget
        if not hasattr(self, '_base_eta'):
            self._base_eta, self._base_lam2 = self.o['eta'], self.o['lam2']
        s = 1.0 if self.o['warm'] <= 0 else min(1.0, ep / float(self.o['warm']))
        self.o['eta'], self.o['lam2'] = self._base_eta * s, self._base_lam2 * s

    def build(self):
        super().build()
        if float(self.o.get('s_w', 1.0)) != 1.0 and int(self.o.get('ii_k', 0)) == 0:
            import numpy as np, scipy.sparse as sp
            from common import sym_norm, sp_to_torch
            d = self.data; J = d.joint(S=d.S * float(self.o['s_w'])).tocsr()
            self.A2 = sp_to_torch(sym_norm(J))
        if int(self.o.get('ii_k', 0)) > 0:
            # implicit item-item block (top-k cosine co-occurrence on TRAINING data) stacked into the social-aware graph
            import numpy as np, scipy.sparse as sp
            from models import _topk_sparse
            from common import sym_norm, sp_to_torch
            d = self.data; R = d.R
            n = np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel()) + 1e-12
            Rn = R @ sp.diags(1 / n)
            T = _topk_sparse((Rn.T @ Rn).tocsr(), int(self.o['ii_k']))
            T = ((T + T.T) > 0).astype(np.float32) * float(self.o.get('ii_w', 1.0))
            J = d.joint(S=d.S).tolil(); J[d.nu:, d.nu:] = T; J = J.tocsr()
            self.A2 = sp_to_torch(sym_norm(J))
            B = (J != 0).astype(np.float32); rs = np.asarray(B.sum(1)).ravel()
            self.P2 = sp_to_torch(sp.diags(np.where(rs > 0, 1 / np.maximum(rs, 1), 0).astype(np.float32)) @ B)
        self.adjs = {12: (self.A1, self.A2), 1: (self.A1,), 2: (self.A2,)}[int(self.o['views'])]
        self.vidx = {12: (0, 1), 1: (0,), 2: (1,)}[int(self.o['views'])]

    def noisy_views(self, sd, train):
        out, cls = [], []
        bgt = self.budget(sd) if (train and sd is not None) else 1.0
        X0 = self.ego()
        for A in self.adjs:
            X = X0; layers = [X] if self.o['ego'] else []; kept = None
            for l in range(self.L):
                X = spmm_(A, X)
                if train and self.o['lam2'] > 0:
                    X = X + self.o['eta'] * bgt * torch.sign(X) * F.normalize(torch.rand_like(X), dim=-1)
                layers.append(X)
                if l == self.o['lstar'] - 1:
                    kept = X
            out.append(torch.stack(layers, 0).mean(0)); cls.append(kept)
        return out, cls

    def fuse(self, V, nodes, ubar, train):
        if ubar is not None:
            ubar = [ubar[k] for k in self.vidx]
        if len(V) == 1:
            return V[0][nodes]
        if self.o['fusion'] == 'mean':
            return sum(e[nodes] for e in V) / len(V)
        return super().fuse(V, nodes, ubar, train)

    def loss(self, u, i, j):
        o = self.o; nu = self.nu
        sd = var = ubar = None
        if o['gauss']:
            mu, var = self.all_gauss(); sd = var.sqrt(); s = sd.mean(1, keepdim=True)
            ubar = [spmm_(self.P1, s)[:, 0], spmm_(self.P2, s)[:, 0]]
            self.vbar = 2 * var.mean().detach()
        V, Vcl = self.noisy_views(sd, True)
        B = len(u)
        if o['rec'] == 'ssmdot':
            # sampled softmax, dot-product logits (keeps the norm / popularity information), K shared uniform negatives
            J = torch.randint(0, self.ni, (int(o['K']),), device=DEV, generator=self.gen)
            nodes = torch.cat([u, nu + i, nu + J])
            e = self.fuse(V, nodes, ubar, True); eu, ei, ej = e[:B], e[B:2 * B], e[2 * B:]
            pos = (eu * ei).sum(1) / o['tau_r']; neg = (eu @ ej.T) / o['tau_r']
            l = (torch.logsumexp(torch.cat([pos[:, None], neg], 1), 1) - pos).mean() + self.reg_loss(self.E, u, i, j)
        elif o['rec'] == 'ssm2':
            J = torch.randint(0, self.ni, (int(o['K']),), device=DEV, generator=self.gen)
            nodes = torch.cat([u, nu + i, nu + J])
            e = self.fuse(V, nodes, ubar, True); eu, ei, ej = e[:B], e[B:2 * B], e[2 * B:]
            tau = o['tau_r']
            if o['gauss'] and o['kappa_r'] != 0:      # uncertainty-adaptive temperature of the ranking softmax
                su = sd.detach().mean(1)[:nu]
                tau = o['tau_r'] * (su[u] / su.mean()) ** o['kappa_r']
            l = ssm_logits_loss(eu, ei, ej, tau) + self.reg_loss(self.E, u, i, j)
        elif o['fusion'] == 'score' and len(V) == 2:
            # late (score-level) fusion: y_ui = sum_r alpha_{u,r} <e_u^r, e_i^r>, alpha from the uncertainty-gated attention
            al = self.user_alpha(V, u, ubar, True)
            ps = sum(al[:, r] * (V[r][u] * V[r][nu + i]).sum(1) for r in range(2))
            ng = sum(al[:, r] * (V[r][u] * V[r][nu + j]).sum(1) for r in range(2))
            l = bpr(ps, ng) + self.reg_loss(self.E, u, i, j)
        else:
            nodes = torch.cat([u, nu + i, nu + j])
            e = self.fuse(V, nodes, ubar, True); eu, ei, ej = e[:B], e[B:2 * B], e[2 * B:]
            vu = vi = vj = None
            if o['gauss']:
                vu, vi, vj = var[u], var[nu + i], var[nu + j]
            l = bpr(self.score(eu, vu, ei, vi), self.score(eu, vu, ej, vj)) + self.reg_loss(self.E, u, i, j)
        uu, ii = torch.unique(u), torch.unique(i) + nu; bn = torch.cat([uu, ii])
        if o['gauss']:
            if o['lam1'] > 0:
                l = l + o['lam1'] * (1 - F.cosine_similarity(self.E[bn], mu[bn], dim=1)).mean()
            l = l + o['beta_kl'] * 0.5 * (var[bn] + mu[bn] ** 2 - 1 - var[bn].log()).sum(1).mean()
            l = l + self.reg * sum(p.pow(2).sum() for n_, p in self.named_parameters() if n_.split('.')[-1].startswith('W'))
        if o['lam2'] > 0:
            cl = sum(self.nce(Vr[g], Vc[g], sd, g) for Vr, Vc in zip(V, Vcl) for g in (uu, ii))
            l = l + o['lam2'] * cl
        return l

    def user_alpha(self, V, nodes, ubar, train):
        logits = []
        for r, e in enumerate(V):
            l = torch.tanh(e[nodes] @ self.Wa + self.ba) @ self.q
            if ubar is not None and self.o['uattn']:
                l = l - self.o['gamma'] * ubar[self.vidx[r]][nodes]
            if train and self.o['attn_drop'] > 0:
                l = F.dropout(l, self.o['attn_drop'])
            logits.append(l)
        return torch.softmax(torch.stack(logits, 1), 1)

    @torch.no_grad()
    def item_unc(self, var):
        # standardised log mean-variance of every item (the item-side uncertainty of the Gaussian head)
        z = var[self.nu:].mean(1).log()
        return (z - z.mean()) / z.std().clamp_min(1e-8)

    def corrected(self, U, I, zi, beta):
        # distribution-aware ranking correction:  y_ui = e_u.e_i + beta * sd_u(y_u.) * z_i
        def f(ub):
            sc = U[ub] @ I.T
            return sc + beta * sc.std(1, keepdim=True) * zi[None]
        return f

    @torch.no_grad()
    def posthoc_scorers(self):
        if not self.o['gauss'] or self.o['rec'] not in ('bpr', 'ssmdot') or self.o['omega'] > 0 or self.o['fusion'] == 'score':
            return {}
        e, var = self.final(); U, I = e[:self.nu], e[self.nu:]; zi = self.item_unc(var)
        return {f'beta{b}': self.corrected(U, I, zi, b) for b in (-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2)}

    @torch.no_grad()
    def scorer(self):
        if self.o['fusion'] == 'score' and len(self.adjs) == 2:
            sd = ubar = None
            if self.o['gauss']:
                mu, var = self.all_gauss(); sd = var.sqrt(); s = sd.mean(1, keepdim=True)
                ubar = [spmm_(self.P1, s)[:, 0], spmm_(self.P2, s)[:, 0]]
            V, _ = self.noisy_views(sd, False); nu = self.nu
            al = self.user_alpha(V, torch.arange(nu, device=DEV), ubar, False)
            return lambda ub: sum(al[ub, r:r + 1] * (V[r][ub] @ V[r][nu:].T) for r in range(2))
        if self.o.get('rank_beta', 0) and self.o['gauss'] and self.o['rec'] == 'bpr':
            e, var = self.final()
            return self.corrected(e[:self.nu], e[self.nu:], self.item_unc(var), float(self.o['rank_beta']))
        if self.o['rec'] in ('ssm', 'ssm2'):
            e, var = self.final(); U, I = F.normalize(e[:self.nu], dim=1), F.normalize(e[self.nu:], dim=1)
            return lambda ub: U[ub] @ I.T
        return super().scorer()


def spmm_(A, X):
    return torch.sparse.mm(A, X)


def item_item_block(data, k, w=1.0):
    import numpy as np, scipy.sparse as sp
    from models import _topk_sparse
    R = data.R
    n = np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel()) + 1e-12
    Rn = R @ sp.diags(1 / n)
    T = _topk_sparse((Rn.T @ Rn).tocsr(), int(k))
    return ((T + T.T) > 0).astype(np.float32) * float(w)


class XSimGCL3(XSimGCL):
    """XSimGCL control with the same loss formula, layers and item-item block in its graph."""
    def build(self):
        super().build()
        if int(self.a.get('ii_k', 0)) > 0:
            from common import sym_norm, sp_to_torch
            J = self.data.joint().tolil(); J[self.nu:, self.nu:] = item_item_block(self.data, self.a['ii_k'], self.a.get('ii_w', 1.0))
            self.A = sp_to_torch(sym_norm(J.tocsr()))
    def loss(self, u, i, j):
        a = self.a
        if a.get('rec', 'bpr') != 'ssm2':
            return super().loss(u, i, j)
        X, Xcl = noisy_prop(self.E, self.A, self.L, a.get('eps', 0.2), keep_layer=a.get('lstar', 1))
        J = torch.randint(0, self.ni, (int(a.get('K', 1024)),), device=DEV, generator=self.gen)
        l = ssm_logits_loss(X[u], X[self.nu + i], X[self.nu + J], a.get('tau_r', 0.1)) + self.reg_loss(self.E, u, i, j)
        return l + a.get('lam', 0.2) * self.cl_pair(X, Xcl, u, i, a.get('tau', 0.15))

    @torch.no_grad()
    def scorer(self):
        X = self.embed()
        if self.a.get('rec', 'bpr') == 'ssm2':
            X = F.normalize(X, dim=1)
        return self.dot_scorer(X)


models.MODELS.update(SGCRec3=SGCRec3, XSimGCL3=XSimGCL3)
