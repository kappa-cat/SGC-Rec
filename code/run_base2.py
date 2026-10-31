"""test-split contrastive baselines with validation-selected configurations: python run_base2.py ds seeds(csv)"""
import sys,json
from sgc2 import *
NAMES=['SGL','SimGCL','LightGCL','HCCF','SEPT']
ds=sys.argv[1];seeds=[int(x) for x in sys.argv[2].split(',')];bc=json.load(open(HERE/'results'/'baseline_cfg2.json'))[ds]
for seed in seeds:
    f=HERE/'results'/f'base2_{ds}_s{seed}.json'
    if f.exists(): continue
    out=json.load(open(HERE/'results'/f'base_{ds}_s{seed}.json'));d=Data(ds,seed)
    for n in NAMES:
        c=dict(bc[n]);lr=c.pop('lr')
        m,info=train_model(n,d,seed,dict(d=64,L=2,reg=1e-4,**c),lr,max_epochs=150)
        with torch.no_grad(): out[n]=dict(summ(evaluate(d,m.scorer(),'test'),d),**info)
        print(ds,seed,n,out[n],flush=True);del m;torch.cuda.empty_cache()
    json.dump(out,open(f,'w'),indent=1)
