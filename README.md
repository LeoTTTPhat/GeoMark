# GeoMark: reproduction files

GeoMark watermarks generators of mobility trajectories that emit hexagonal H3 cells. A keyed pseudorandom function marks
part of the (seed, cell) pairs as green and the generator slightly favours them. Each trip carries only a small excess
of green pairs. A released dataset, however, pools enough pairs for the key holder to detect the watermark with a
calibrated test. The code covers a small GPT trained from scratch and a LoRA-tuned Qwen2.5-1.5B generator, on the
GeoLife (Beijing) and Porto taxi datasets. It also covers trajectory attacks, among them noise, downsampling, retiming,
cropping, splicing, map matching, regeneration, watermark stealing and students trained on watermarked releases.

This folder holds the code, configurations, the road-network extracts, every summary and per-experiment result record,
and the figures.

## Layout

| Path | Contents |
|---|---|
| `src/geomark/` | Core package: H3 grid (`hexgrid.py`), keyed PRF (`prf.py`), watermark schemes and sampler (`scheme.py`, `sampler.py`), detection and pooling (`detect.py`, `sequential.py`, `stats.py`), attacks (`attacks.py`), map matching (`mapmatch.py`), post-hoc baseline (`posthoc.py`), utility metrics (`utility.py`) |
| `src/*.py` | Pipeline steps: preprocessing, training (`train_gpt.py`, `train_qwen_lora.py`), generation, evaluation, calibration, baselines, stealing, radioactivity, capacity, key variation, distribution shift, aggregation, statistics, figures |
| `src/rev1/`, `src/rev2/`, `src/rev3/`, `src/rev4/` | Follow-up experiments and model-free analyses, driven by `src/rev1_steps.py`, `src/rev2_steps.py` and `src/rev3_steps.py` |
| `src/pipeline.sh`, `src/run_*.sh`, `src/stage3_followup.sh` | Drivers; every step is checkpointed and resumable (done-markers in `<out_dir>/.done/`) |
| `src/configs/` | YAML configurations (`full.yaml` = main matrix, `rev*.yaml` follow-ups, `geomark_st.yaml` smoke test) |
| `src/tests/` | Unit tests (50) |
| `data/osm/` | OpenStreetMap drive networks of Porto and Beijing (GraphML, built with OSMnx by `src/fetch_osm.py`) |
| `data/derived/vocab_*.npy` | H3 cell vocabularies per dataset and resolution |
| `results/` | `summary.json` (all aggregated results), `stats_tests.json` (tests and intervals) and every per-experiment record (JSON, JSONL, CSV, NPZ) |
| `results/gen/gpt_porto_r9_s0/gm_kgw__k0__s0.pkl` | One generated release, used by the overview figure |
| `figures/` | All figures (PDF, and PNG for the icon figures) |
| `figkit/` | Script for the two icon figures (`fig_overview_art`, `fig_launder_art`) |
| `tools/memguard.py` | Pass-through stand-in for the memory guard that wrapped each step on our shared machine |
| `requirements.txt` | Exact Python package versions used (Python 3.11) |

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run every command from this folder. The drivers read two optional environment variables:
- `PY`: the Python interpreter (default `python`).
- `MEMGUARD`: a wrapper called as `$MEMGUARD --name NAME -- command` (default `tools/memguard.py`, which runs the command
  directly).

## 1. Check the results from the shipped records (CPU, about 1 minute)

```bash
python -m pytest -q src/tests
python src/make_figures.py
python src/rev1/make_figs_rev1.py
python src/plots.py --cfg src/configs/full.yaml
python figkit/p1_figures.py
```

All figures are drawn from `results/summary.json`, `results/stats_tests.json` and the per-experiment records.

`src/aggregate.py` and `src/stats_tests.py` recompute those two files. They need the generated and attacked releases
(about 11 GB), which section 2 produces. Running them on this folder alone would overwrite the two files with
incomplete versions.

## 2. Re-run the experiments (GPU)

**Data.** Download the two GPS datasets and check their SHA-256:

| File | Source | SHA-256 |
|---|---|---|
| `data/raw/geolife/geolife13.zip` | Microsoft Research GeoLife 1.3: https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip | `1107c5ac064d0a23c8d021a8736a77e53abc75b227062e6260342c6a8d86bdb6` |
| `data/raw/porto/porto_full.zip` | UCI 339, ECML/PKDD 2015 taxi trajectories (CC BY 4.0): https://archive.ics.uci.edu/static/public/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip | `a33e2a5e145607ae2bad0db5d21b7548c88b7e0f9db1ce15839f24c4c61f8c76` |

Unpack GeoLife to `data/raw/geolife/Geolife Trajectories 1.3/Data/` and the Porto archive to `data/raw/porto/train.csv`.

**Models.** The LLM generator starts from `Qwen/Qwen2.5-1.5B-Instruct`, and the smoke test uses
`Qwen/Qwen2.5-0.5B-Instruct`. Download both into the Hugging Face cache, because the drivers run with
`HF_HUB_OFFLINE=1`. The GPT generators, next-location models and student models are trained from scratch by the
pipeline.

**Runs.**

```bash
bash src/run_smoke.sh                  # tiny end-to-end run (results/smoke/)
bash src/run_full.sh                   # main matrix: prep, train, generate, attack, map-match, evaluate, calibrate,
                                       # baselines, utility, stealing, radioactivity, capacity, aggregate, plots
bash src/stage3_followup.sh            # key-randomised null for the radioactivity test
python src/rev1_steps.py --list        # follow-up steps; run them with:  bash src/run_rev1.sh <group> <step,step,...>
python src/rev2_steps.py --list        #                                  bash src/run_rev2.sh <group> <step,step,...>
python src/rev3_steps.py --list        #                                  bash src/run_rev3.sh <group> <step,step,...>
python src/rev4/calnull4.py            # calibrated null at release size (model-free)
```

`CFG=<config> TAG=<name> bash src/pipeline.sh <step ...>` runs single pipeline steps. Every step skips finished work,
so a driver can be stopped and restarted at any time. We ran everything on one Apple M3 Ultra (512 GB unified memory).

## Notes

- Not included, because the pipeline rebuilds them:
  - the raw GPS datasets (re-download them as above);
  - the preprocessed trip tables (`data/derived/geolife.pkl`, `porto.pkl`; about 390 MB, built by the `prep` step);
  - the generated and attacked releases (about 11 GB);
  - model weights (about 2.2 GB);
  - smoke-test outputs.
- The road networks in `data/osm/` are © OpenStreetMap contributors, available under the Open Database License (ODbL).
- The icon set in `figkit/icons/` is Phosphor Icons v2.1.1 (MIT licence, `figkit/icons/LICENSE-phosphor-MIT.txt`).
