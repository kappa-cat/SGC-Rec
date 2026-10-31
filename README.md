# SGC-Rec

SGC-Rec is a graph recommendation model that combines (1) an amortized Gaussian-mixture pre-training stage on
empirical-CDF node features, (2) two complementary graph experts — an XSimGCL interaction expert and a Gaussian-initialized
full-catalogue LightGCN expert — and (3) a ten-parameter heteroscedastic Gaussian decision head that ranks items by the
lower-confidence score `s_ui = mu_ui − kappa * sqrt(v_ui)`.

All methods (SGC-Rec and the baselines UserCF, ItemCF, NCF, DeepFM, LightGCN, SGL, SimGCL, LightGCL, XSimGCL, HCCF, SEPT)
are implemented in one PyTorch code base and share the same data splits, training loop and full-catalogue evaluation.

## Repository layout

```
code/        model and experiment code
  common.py            data loading, full-catalogue evaluation (Precision/Recall/NDCG@10, entropy)
  models.py            baselines and the two graph experts
  models3.py, models4.py  auxiliary model components used by the experts
  gaussian_pretrain.py linear-head Gaussian prior (ablation "linear-head prior")
  sgc2.py              SGC-Rec pipeline: Gaussian pre-training, catalogue expert, decision head, calibration
  sgc3.py              decision-head recipes A–F used in the stressed regimes
  prep.py, get_raw_data.py, check_data.py   dataset construction (see data/README.md)
  run_*.py, select_*.py, design*.py, sens.py   experiment entry points (see below)
data/        pre-processed datasets (beauty.npz, lastfm.npz, ml10m.npz), splits for seeds 0–14
results/     result files of all experiments
analysis/    scripts that aggregate results/ into tables, statistical tests and figures
```

## Requirements

Python ≥ 3.10 and a CUDA GPU. Install the dependencies with `pip install -r requirements.txt`
(experiments were run with PyTorch 2.11 + CUDA 12.8 on one NVIDIA RTX 5060 Laptop GPU, 8 GB).

## Running the experiments

All commands are run from `code/`; `<ds>` is `beauty`, `lastfm` or `ml10m`. Each script writes its output to `results/`.
Hyper-parameters are selected on validation data only; the test split is used once the configuration is fixed.

| Step | Command | Output |
|---|---|---|
| Baseline learning rates (validation seed 0) | `python select_base.py <ds>` | `baseline_cfg.json` |
| Contrastive-baseline search (validation seed 0) | `python select_base2.py <ds>` | `baseline_cfg2.json` |
| Baselines (test, seeds 0–4) | `python run_base.py <ds> 0,1,2,3,4` and `python run_base2.py <ds> 0,1,2,3,4` | `base_*`, `base2_*` |
| Gaussian-prior configuration (validation seeds 0–2) | `python design2.py beauty lastfm ml10m`, then `python design3.py beauty ml10m` (Last-FM keeps the `design2` selection) | `design2_*`, `design3_*`, `cfg_<ds>.json` |
| SGC-Rec, XSimGCL and all ablations (test) | `python run_main.py <ds> 0,1,2,3,4` (add seeds `5,...,14` for the 15-seed analysis) | `main_<ds>_s<seed>.json/.npz` |
| κ and catalogue-horizon sensitivity (validation) | `python sens.py <ds>` | `sens_<ds>.json` |
| Last-FM friendship ablation | `python run_social.py 0,1,2,3,4` | `soc_lastfm_s*.json` |
| Random-edge injection (10/20/30%) | `python run_robust_ext.py <ds> 0,1,2 10,20,30` | `rcx_*` |
| Scarce training data (20/40% retained) | `python run_scarce_ext.py <ds> 0,1,2 20` and `... 40` | `scx_*` |

Then, from `analysis/`:

```bash
python collect2.py      # means, paired t / Wilcoxon tests with Holm correction, user-activity groups -> numbers2.json
python agg_ext.py       # validation-selected head recipes for the stressed regimes -> agg_ext.json
python ext15_tests.py   # fifteen-seed analysis of Gaussian pre-training -> results/ext15_tests.json
python make_figures.py  # result figures -> analysis/figs/
```

The result files of all experiments are included in `results/`, so the analysis scripts can be run directly without
re-training. `run_main.py` caches the trained XSimGCL embeddings in `results/cache/` (not included).

## Main results (Recall@10, mean over five seeds)

| Model | Amazon-Beauty | Last-FM | MovieLens-10M |
|---|---:|---:|---:|
| LightGCN | 0.0857 | 0.1935 | 0.2165 |
| SimGCL | 0.0983 | 0.2139 | 0.2348 |
| XSimGCL | 0.0997 | 0.2181 | 0.2382 |
| **SGC-Rec** | **0.1070** | **0.2223** | **0.2462** |

## Data

See [`data/README.md`](data/README.md) for the dataset statistics, the original sources and the scripts that rebuild the
pre-processed files from the raw data.
