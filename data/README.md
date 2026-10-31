# Datasets

`beauty.npz`, `lastfm.npz` and `ml10m.npz` are the pre-processed datasets used in the experiments. Each file contains the
user/item counts, the item-category matrix and, for every seed 0–14, the per-user 80%/10%/10% train/validation/test split
(`u_<s>`, `i_<s>`, `tag_<s>` with tag 0/1/2 = train/validation/test) and the auxiliary user graph (`social_<s>`).
`<name>_stats.json` summarizes each file.

| File | Users | Items | Interactions | Auxiliary user graph | Categories |
|---|---:|---:|---:|---|---|
| `beauty.npz` | 22,363 | 12,101 | 198,502 | co-purchase (≥ 3 shared training items) | 195 leaf categories |
| `lastfm.npz` | 1,892 | 17,632 | 92,834 | declared friendships (native) | 1,040 top tags |
| `ml10m.npz` | 2,978 | 3,903 | 214,220 | mutual 10-NN Jaccard on training data | 19 genres |

## Rebuilding from the original sources

The raw files are not redistributed here. They can be downloaded from the original public releases and processed with:

```bash
cd code
python get_raw_data.py                       # downloads into data/raw/
SGC_OUT=data_rebuilt python prep.py          # Windows: set SGC_OUT=data_rebuilt && python prep.py
python check_data.py data_rebuilt            # verifies that the rebuilt files are identical to data/*.npz
```

Sources:
- Amazon-Beauty (5-core) — RUCAIBox/CIKM2020-S3Rec release
- Last-FM ratings and friendships — Coder-Yu/QRec release; Last-FM tags — HetRec 2011 (GroupLens)
- MovieLens-10M — GroupLens (ratings ≥ 4 treated as positive; fixed subsample of 2,978 users and 3,903 movies)

Please respect the terms of use of the original datasets (in particular the GroupLens MovieLens and HetRec licenses,
which permit research use and require citation of the original data).
