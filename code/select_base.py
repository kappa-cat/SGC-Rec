"""learning-rate selection for trainable baselines on validation seed 0"""
import sys,json
from sgc2 import *
NAMES=['LightGCN','SGL','SimGCL','LightGCL','HCCF','SEPT','NCF','DeepFM']
out={}
f=HERE/'results'/'baseline_cfg.json'
if f.exists(): out=json.load(open(f))
for ds in sys.argv[1:]:
    d=Data(ds,0);out[ds]={}
    for n in NAMES:
        best=None
        for lr in (.001,.002):
            m,info=train_model(n,d,0,dict(d=64,L=2,reg=1e-4),lr,max_epochs=150 if n not in('NCF','DeepFM') else 100)
            print(ds,n,lr,info,flush=True)
            if best is None or info['best_val']>best[0]: best=(info['best_val'],lr)
            del m;torch.cuda.empty_cache()
        out[ds][n]=best[1]
    json.dump(out,open(f,'w'),indent=1)
