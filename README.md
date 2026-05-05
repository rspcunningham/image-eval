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

Each evaluation writes `mtf.csv`, `mtf.png`, `roi_fits/`, `nps.csv`, `nps.png`, `nps_spectra/`, and `registration/`.

## Comparison

Compare `outputs/raw/mtf.csv` and `outputs/reconstruction/mtf.csv` by matching rows on `LP per MM` and using the `average MTF` column. The plot `mtf_comparison.png` is generated from those two CSV files.

Compare `outputs/raw/nps.csv` and `outputs/reconstruction/nps.csv` by using the `frequency lp/mm`, `black NPS`, and `white NPS` columns. The plot `nps_comparison.png` is generated from those two CSV files with a log-scaled Y axis.
