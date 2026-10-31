"""Fifteen-seed analysis of Gaussian pre-training and SGC-Rec vs XSimGCL over seeds 0-14.
Reads results/main_<ds>_s{0..14}.json and writes results/ext15_tests.json as rows
[dataset, metric, comparison, relative gain %, #positive seeds, raw paired t-test p, mean A, mean B, Holm-adjusted p].
comparison: 'expert' = catalogue expert pre-trained vs not; 'fused' = SGC-Rec vs w/o pre-training; 'xsim' = SGC-Rec vs XSimGCL.
Holm correction: over the six expert+fused comparisons of each metric, and over the three XSimGCL comparisons of each metric."""
import json,numpy as np
from pathlib import Path
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'; DS=['beauty','lastfm','ml10m']; SE=range(15)
PAIRS={'expert':('cat_prior_only','cat_nopre_only'),'fused':('full','nopre'),'xsim':('full','xsim')}
M={(d,s):json.load(open(R/f'main_{d}_s{s}.json')) for d in DS for s in SE}
rows=[]
for m in 'RN':
    block={}
    for d in DS:
        for c,(a,b) in PAIRS.items():
            x=np.array([M[d,s][a][m] for s in SE]);y=np.array([M[d,s][b][m] for s in SE])
            block[d,c]=[d,m,c,float((x.mean()/y.mean()-1)*100),float((x>y).sum()),float(stats.ttest_rel(x,y).pvalue),float(x.mean()),float(y.mean())]
    for fam in (('expert','fused'),('xsim',)):
        keys=sorted([k for k in block if k[1] in fam],key=lambda k:block[k][5]); run=0
        for i,k in enumerate(keys):
            run=max(run,min(1.,(len(keys)-i)*block[k][5])); block[k].append(run)
    rows+= [block[d,c] for d in DS for c in PAIRS]
json.dump(rows,open(R/'ext15_tests.json','w'),indent=1)
for r in rows: print(r[0],r[1],r[2],f"{r[3]:+.2f}% ({int(r[4])}/15) p={r[5]:.2g} holm={r[8]:.2g}")
