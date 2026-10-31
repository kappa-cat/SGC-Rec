"""Compare npz files rebuilt by prep.py from the raw data with the shipped ones (all seeds).
   SGC_OUT=data_rebuilt python prep.py          # rebuild into data_rebuilt/
   python check_data.py data_rebuilt            # compare with data/"""
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; new = ROOT / sys.argv[1] if len(sys.argv) > 1 else ROOT / 'data_rebuilt'
ok = True
for n in ('beauty', 'lastfm', 'ml10m'):
    a, b = np.load(ROOT / 'data' / f'{n}.npz', allow_pickle=True), np.load(new / f'{n}.npz', allow_pickle=True)
    keys = [k for k in b.files]
    bad = [k for k in keys if k not in a.files or a[k].shape != b[k].shape or not np.array_equal(a[k], b[k])]
    print(n, 'OK (identical on %d arrays)' % len(keys) if not bad else f'DIFF in {bad[:8]}'); ok &= not bad
print('ALL IDENTICAL' if ok else 'MISMATCH')
