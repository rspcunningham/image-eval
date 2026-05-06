# image-eval

## Workflow

Create or edit the ROI template from the raw/base image:

```bash
uv run python -m image_eval.initialize samples/raw.npy template.json --groups 4-7 --elements 1-6
```

Evaluate the raw image:

```bash
uv run image-eval --base-url samples/raw.npy --template template.json --subject-url samples/raw.npy --out outputs/raw
```

Evaluate the reconstruction:

```bash
uv run image-eval --base-url samples/raw.npy --template template.json --subject-url samples/reconstruction.npy --out outputs/reconstruction
```

Each evaluation writes `report.json`, `mtf.png`, `roi_fits/`, `nps.png`, `nps_spectra/`, `dqe.png`, and `registration/` by default. Use `--json` to print the JSON report to stdout, `--no-plots` to write only `report.json`, or `--plots mtf,nps` to choose plot classes.

## Comparison

After both evaluations complete, generate `mtf_comparison.png` from the `mtf.rows` entries in `outputs/raw/report.json` and `outputs/reconstruction/report.json` by plotting `x_mtf` and `y_mtf` against `frequency`.

Generate `nps_comparison.png` from the `nps.rows` entries in `outputs/raw/report.json` and `outputs/reconstruction/report.json` by plotting `black_nps` and `white_nps` against `frequency` with a log-scaled Y axis.

DQE values are available in `dqe.rows`; no DQE comparison plot is part of the default comparison workflow.
