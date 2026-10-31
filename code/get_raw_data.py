"""Download the public raw datasets and lay them out exactly as prep.py expects (data/raw/...).
Run ONCE on a machine with internet access:   python get_raw_data.py   (then:  python prep.py  /  see check_data.py)

  Amazon-Beauty (5-core, S3Rec release)  github.com/RUCAIBox/CIKM2020-S3Rec          -> raw/CIKM2020-S3Rec/data/
  Last-FM ratings + friendships          github.com/Coder-Yu/QRec  (dataset/lastfm)  -> raw/QRec/dataset/lastfm/
  Last-FM tags (HetRec-2011)             files.grouplens.org/datasets/hetrec2011     -> raw/recommender_lastFM/
  MovieLens-10M                          files.grouplens.org/datasets/movielens      -> raw/movielens-bench/ml-10m/
                                         (ratings.dat / movies.dat are converted from '::' to ';' separators, as read by prep.py)
"""
import io, gzip, zipfile, urllib.request
from pathlib import Path
RAW = Path(__file__).resolve().parents[1] / 'data' / 'raw'
GH = 'https://raw.githubusercontent.com/'
def fetch(url):
    print('GET', url, flush=True)
    with urllib.request.urlopen(url, timeout=300) as r: return r.read()
def put(rel, data):
    p = RAW / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data); print('  ->', p, len(data), 'bytes')
for f in ('Beauty.txt', 'Beauty_item2attributes.json'):
    put(f'CIKM2020-S3Rec/data/{f}', fetch(GH + f'RUCAIBox/CIKM2020-S3Rec/master/data/{f}'))
for f in ('ratings.txt', 'trusts.txt'):
    put(f'QRec/dataset/lastfm/{f}', fetch(GH + f'Coder-Yu/QRec/master/dataset/lastfm/{f}'))
z = zipfile.ZipFile(io.BytesIO(fetch('https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip')))
put('recommender_lastFM/user_taggedartists.dat', z.read('user_taggedartists.dat'))
z = zipfile.ZipFile(io.BytesIO(fetch('https://files.grouplens.org/datasets/movielens/ml-10m.zip')))
conv = lambda b: b.replace(b'::', b';')
put('movielens-bench/ml-10m/ratings.dat.gz', gzip.compress(conv(z.read('ml-10M100K/ratings.dat'))))
put('movielens-bench/ml-10m/movies.dat', conv(z.read('ml-10M100K/movies.dat')))
print('done. Next: python prep.py   (writes data/*.npz; use SGC_OUT=data_rebuilt to keep the shipped files untouched)')
