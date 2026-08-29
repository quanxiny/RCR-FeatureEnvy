# RCR-FeatureEnvy

This repository contains the implementation and frozen results for
**Relation-Aware Contrastive Replay for Continual Feature-Envy Detection: A
Controlled Study**.

The study extends the CG-LSMN Method--Class detector in two directions:

1. static contrastive objectives, including standard supervised contrastive
   learning and local relation contrast (LRC/CL6); and
2. project-incremental learning with fine-tuning, EWC, replay, logit and
   representation distillation, and cross-graph relation preservation.

The repository reports corrected repeated-fold comparisons and does not claim
universal superiority when the confirmatory gates do not pass.

## Repository contents

- `src/losses/`: focal, supervised contrastive, local matching, and directed
  relation objectives.
- `src/continual/`: project stream construction, replay, EWC, and relation
  distillation.
- `src/models/`: CG-LSMN plus the projection head used by contrastive methods.
- `scripts/`: training, aggregation, experiment orchestration, and statistical
  summaries.
- `artifacts/continual/`: the fixed project/case-preserving streams for five
  folds and three stream seeds.
- `results/`: frozen run-level and summary CSV/JSON files used by the paper.
- `paper_figures/`: data and deterministic code for paper Figures 2--4.

Dataset caches and training checkpoints are omitted from Git history because
of their size. All reported numerical summaries and stream manifests are
included.

## Environment

The study was verified with Python 3.11, PyTorch 2.10.0, PyTorch Geometric
2.7.0, CUDA 12.8, NumPy 2.2.6, SciPy 1.17.1, and scikit-learn 1.9.0.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

Install PyTorch from the wheel index matching the local CUDA version when
necessary.

## Dataset setup

Copy or link the original CG-LSMN `Dataset_Method_Class` directory into
`data/`, or set:

```bash
export CGLSMN_DATA_ROOT=/path/to/CG-LSMN/Data
```

See `data/README.md` for the required files and checksums. Generate the
derived metadata once:

```bash
python scripts/build_metadata.py
python scripts/build_token_statistics.py --task 2
```

The precomputed continual streams in `artifacts/continual/` can then be reused
directly.

## Static contrastive experiment

Tune CL6 on a project-disjoint inner validation split:

```bash
python scripts/run_contrastive.py \
  --method CL6 --task 2 --fold 1 --seed 123 \
  --epochs 50 --selection-metric mcc --patience 5 \
  --skip-test --tag reproduction_cl6_tune \
  --device cuda
```

After reading the selected epoch from the tuning result, retrain on the full
outer training fold and evaluate the outer test fold once:

```bash
python scripts/run_contrastive.py \
  --method CL6 --task 2 --fold 1 --seed 123 \
  --full-train-epochs SELECTED_EPOCHS \
  --tag reproduction_cl6_full --device cuda
```

Standard supervised contrastive learning uses `--method CL1`; the focal
control uses `--method CL5` with all auxiliary coefficients set to zero. The
paper's complete static plan can be inspected without launching jobs:

```bash
python scripts/orchestrate_task2_detection_sci.py --plan-only
python scripts/orchestrate_task2_comparison_extension.py --plan-only
```

## Continual experiment

Run a single relation-aware replay unit:

```bash
python scripts/run_continual.py \
  --method replay_relation --base-objective cl6 \
  --task 2 --fold 1 --seed 42 --stage-epochs 8 \
  --buffer-ratio 0.10 \
  --stream artifacts/continual/task2_fold1_seed42.json \
  --device cuda
```

Available methods include `naive`, `ewc`, `replay`, `replay_kd`, and
`replay_relation`. The complete two-GPU workflows are:

```bash
python scripts/orchestrate_task2_continual_sci.py --devices cuda:0 cuda:1
python scripts/orchestrate_task2_ei_extension.py --devices cuda:0,cuda:1
```

Both workflows are resumable and retain one fixed outer-test evaluation per
run.

## Intermediate results

The following files reproduce the paper's numerical evidence without GPU
training:

- `results/workflow/task2_detection_sci_raw.csv` and
  `task2_detection_sci_summary.json`;
- `results/workflow/task2_comparison_extension_raw.csv` and its summary;
- `results/workflow/task2_continual_sci_raw.csv` and its summary;
- `results/workflow/task2_ei_extension_raw.csv` and its summary;
- `results/continual/stage_metrics.csv`, `forgetting_matrix.csv`, and
  `summary.csv`.

Regenerate the three quantitative paper figures with:

```bash
python paper_figures/plot_figures_2_4.py
```

The script reads only the frozen JSON files under `paper_figures/data/` and
writes PDF, SVG, and PNG outputs.

## Reproducibility scope

The graph cache is approximately 1.7 GB and the full checkpoint archive is
several gigabytes. They should be published as archival data or GitHub Release
assets rather than normal Git objects. Every retained result and stream file
is covered by `SHA256SUMS`.

## Citation

Please cite the accompanying paper and the original CG-LSMN paper. Software
citation metadata is provided in `CITATION.cff`.

## License

The implementation is released under the MIT License. Dataset contents,
third-party projects, and dependencies remain subject to their own licenses.
