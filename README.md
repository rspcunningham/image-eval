# image-eval

## Workflow

Create or edit an ROI template from a base image:

```bash
uv run python -m image_eval.initialize samples/raw.npy template.json --groups 4-7 --elements 1-6
```

Evaluate the mtf and nps:

```bash
uv run image-eval samples/reconstruction.npy template.json outputs/reconstruction
```

Each evaluation writes `mtf.csv`, `nps.csv`, and `registration/` JSON artifacts.
