# image-eval

## Workflow

Create or edit the ROI template from the raw/base image:

```bash
uv run python -m image_eval.initialize samples/raw.npy template.json --groups 4-7 --elements 1-6
```

Evaluate the raw image:

```bash
uv run image-eval samples/raw.npy template.json outputs/raw
```

Evaluate the reconstruction:

```bash
uv run image-eval samples/reconstruction.npy template.json outputs/reconstruction
```

Each evaluation writes `mtf.csv`, `mtf.png`, `roi_fits/`, `nps.csv`, `nps.png`, `nps_spectra/`, `dqe.csv`, `dqe.png`, and `registration/`.

## Comparison

After both evaluations complete, generate `mtf_comparison.png` from `outputs/raw/mtf.csv` and `outputs/reconstruction/mtf.csv` by plotting `average MTF` against `LP per MM`.

Generate `nps_comparison.png` from `outputs/raw/nps.csv` and `outputs/reconstruction/nps.csv` by plotting `average NPS` against `LP per MM` with a log-scaled Y axis.

Generate `dqe_comparison.png` from `outputs/raw/dqe.csv` and `outputs/reconstruction/dqe.csv` by plotting `DQE` against `LP per MM`.
