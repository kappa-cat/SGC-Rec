"""test-split baselines: python run_base.py ds seeds(csv)"""
import sys,json
from sgc2 import *
NAMES=['LightGCN','SGL','SimGCL','LightGCL','HCCF','SEPT','NCF','DeepFM','UserCF','ItemCF']
ds=sys.argv[1];seeds=[int(x) for x in sys.argv[2].split(',')];bc=json.load(open(HERE/'results'/'baseline_cfg.json'))[ds]
for seed in seeds:
    f=HERE/'results'/f'base_{ds}_s{seed}.json'
    if f.exists(): continue
    d=Data(ds,seed);out=dict(dataset=ds,seed=seed)
    for n in NAMES:
        m,info=train_model(n,d,seed,dict(d=64,L=2,reg=1e-4),bc.get(n,.001),max_epochs=150 if n not in('NCF','DeepFM') else 100)
        with torch.no_grad(): out[n]=dict(summ(evaluate(d,m.scorer(),'test'),d),**info)
        print(ds,seed,n,out[n],flush=True);del m;torch.cuda.empty_cache()
    json.dump(out,open(f,'w'),indent=1)
