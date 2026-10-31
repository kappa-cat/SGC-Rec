"""Pre-processing for the SGC-Rec experiments.

For every dataset this writes data/<name>.npz containing
  * nu, ni                         number of users / items
  * catw  (ni x ncat, CSR arrays)  item -> category weights (rows sum to 1)
  * u_s, i_s, tag_s  (s = seed)    interactions and 0/1/2 = train/val/test tag of the
                                   80/10/10 random per-user split of seed s
  * social_s                       undirected social edge list used with seed s.
                                   Last-FM: native HetRec friendships (same for every seed).
                                   Beauty / ML-10M: implicit graph built from the TRAIN
                                   part of split s only (no validation/test information).
"""
import numpy as np, json, collections, pandas as pd, scipy.sparse as sp, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('SGC_OUT', str(ROOT / 'data')))   # output dir (use SGC_OUT=data_rebuilt to compare with shipped files)
R = os.environ.get('SGC_RAW', str(ROOT / 'data' / 'raw')).rstrip('/\\') + '/'   # raw files, see data/raw/README.md
SEEDS = list(range(15))
ML_USERS = 3000          # uniform user subsample of MovieLens-10M (CPU budget)
BEAUTY_MIN_COMMON = 3    # Beauty: users linked if they share >= 3 training items
ML_K, ML_THR = 10, 0.05  # ML: mutual top-10 Jaccard neighbours with similarity > 0.05


def core_filter(u, i, k):
    while True:
        cu = collections.Counter(u); ci = collections.Counter(i)
        m = np.array([cu[a] >= k and ci[b] >= k for a, b in zip(u, i)])
        if m.all():
            return u, i
        u, i = u[m], i[m]


def reindex(u, i):
    uu, ui = np.unique(u, return_inverse=True)
    iu, ii = np.unique(i, return_inverse=True)
    return ui.astype(np.int32), ii.astype(np.int32), uu, iu


def split(u, i, nu, seed):
    rng = np.random.default_rng(seed)
    order = np.lexsort((i, u)); u = u[order]; i = i[order]
    bounds = np.r_[0, np.cumsum(np.bincount(u, minlength=nu))]
    tag = np.zeros(len(u), np.int8)
    for x in range(nu):
        a, b = bounds[x], bounds[x + 1]; n = b - a
        if n < 3:
            continue                      # too few interactions: training only
        p = rng.permutation(n)
        nt = max(1, int(round(0.1 * n))); nv = max(1, int(round(0.1 * n)))
        tag[a + p[:nt]] = 2; tag[a + p[nt:nt + nv]] = 1
    return u, i, tag


def train_matrix(u, i, tag, nu, ni):
    m = tag == 0
    return sp.csr_matrix((np.ones(m.sum(), np.float32), (u[m], i[m])), shape=(nu, ni))


def copurchase(Rtr):
    C = (Rtr @ Rtr.T).tocoo()
    m = (C.row < C.col) & (C.data >= BEAUTY_MIN_COMMON)
    return np.stack([C.row[m], C.col[m]], 1).astype(np.int32)


def mutual_knn_jaccard(Rtr):
    inter = (Rtr @ Rtr.T).toarray()
    deg = np.asarray(Rtr.sum(1)).ravel()
    jac = inter / (deg[:, None] + deg[None, :] - inter + 1e-9)
    np.fill_diagonal(jac, 0)
    nn = np.argsort(-jac, 1)[:, :ML_K]
    A = np.zeros(jac.shape, bool)
    A[np.repeat(np.arange(len(deg)), ML_K), nn.ravel()] = True
    A = A & A.T & (jac > ML_THR)
    s, t = np.nonzero(np.triu(A, 1))
    return np.stack([s, t], 1).astype(np.int32)


def finish(name, u, i, item_cats, social_fn=None, native_social=None):
    u, i, uids, iids = reindex(np.asarray(u), np.asarray(i))
    nu, ni = len(uids), len(iids)
    # item -> category weight matrix
    vocab = {}
    rows, cols, vals = [], [], []
    for k, it in enumerate(iids):
        cs = item_cats.get(int(it), ['<none>'])
        for c in cs:
            rows.append(k); cols.append(vocab.setdefault(c, len(vocab))); vals.append(1.0 / len(cs))
    catw = sp.csr_matrix((vals, (rows, cols)), shape=(ni, len(vocab)))
    out = dict(nu=nu, ni=ni, cat_data=catw.data, cat_indices=catw.indices, cat_indptr=catw.indptr,
               ncat=len(vocab), catnames=np.array(list(vocab.keys())))
    if native_social is not None:
        m = {x: k for k, x in enumerate(uids)}
        e = [(m[a], m[b]) for a, b in native_social if a in m and b in m and a != b]
        e = np.unique(np.sort(np.array(e), 1), axis=0).astype(np.int32)
    for s in SEEDS:
        us, is_, tag = split(u, i, nu, s)
        out[f'u_{s}'] = us; out[f'i_{s}'] = is_; out[f'tag_{s}'] = tag
        out[f'social_{s}'] = e if native_social is not None else social_fn(train_matrix(us, is_, tag, nu, ni))
    OUT.mkdir(exist_ok=True); np.savez_compressed(OUT / f'{name}.npz', **out)
    s0 = out['social_0']; n = len(u); t0 = out['tag_0']
    stats = dict(dataset=name, users=nu, items=ni, interactions=n, density=n / nu / ni,
                 social_edges=[int(len(out[f'social_{s}'])) for s in SEEDS],
                 social_density=2 * len(s0) / nu / (nu - 1), categories=len(vocab),
                 test_users=int((np.bincount(out['u_0'][t0 == 2], minlength=nu) > 0).sum()),
                 train=int((t0 == 0).sum()), val=int((t0 == 1).sum()), test=int((t0 == 2).sum()))
    json.dump(stats, open(OUT / f'{name}_stats.json', 'w'), indent=1)
    print(stats)


which = sys.argv[1:] or ['beauty', 'lastfm', 'ml10m']

if 'beauty' in which:
    u, i = [], []
    for l in open(R + 'CIKM2020-S3Rec/data/Beauty.txt'):
        s = l.split(); u += [int(s[0])] * (len(s) - 1); i += [int(x) for x in s[1:]]
    attrs = json.load(open(R + 'CIKM2020-S3Rec/data/Beauty_item2attributes.json'))
    pos = collections.defaultdict(set)
    for v in attrs.values():
        for j, a in enumerate(v):
            pos[a].add(j)
    brands = {a for a, p in pos.items() if p == {0}}          # brand ids only ever occur first
    cats = {int(k): [[a for a in v if a not in brands][-1]] for k, v in attrs.items()}  # leaf category
    finish('beauty', u, i, cats, social_fn=copurchase)

if 'lastfm' in which:
    r = np.loadtxt(R + 'QRec/dataset/lastfm/ratings.txt', dtype=np.int64)
    t = np.loadtxt(R + 'QRec/dataset/lastfm/trusts.txt', dtype=np.int64)
    ta = pd.read_csv(R + 'recommender_lastFM/user_taggedartists.dat', sep='\t')
    top = (ta.groupby(['artistID', 'tagID']).size().reset_index(name='n')
             .sort_values(['artistID', 'n', 'tagID'], ascending=[True, False, True])
             .drop_duplicates('artistID'))
    cats = {int(a): [int(tg)] for a, tg in zip(top.artistID, top.tagID)}   # most frequent tag
    finish('lastfm', r[:, 0], r[:, 1], cats, native_social=[tuple(x) for x in t])

if 'ml10m' in which:
    df = pd.read_csv(R + 'movielens-bench/ml-10m/ratings.dat.gz', sep=';', names=['u', 'i', 'r', 't'])
    df = df[df.r >= 4]
    rng = np.random.default_rng(2026)
    keep = rng.choice(np.sort(df.u.unique()), size=ML_USERS, replace=False)
    df = df[df.u.isin(keep)]
    u, i = core_filter(df.u.values, df.i.values, 5)
    mv = pd.read_csv(R + 'movielens-bench/ml-10m/movies.dat', sep=';', names=['i', 'title', 'g'],
                     encoding='latin-1', quoting=3)
    cats = {int(a): g.split('|') for a, g in zip(mv.i, mv.g)}             # all genres, weight 1/n
    finish('ml10m', u, i, cats, social_fn=mutual_knn_jaccard)
