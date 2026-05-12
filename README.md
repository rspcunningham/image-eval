# image-eval

## Install

From this repository:

```bash
uv tool install .
```

From GitHub: `uv tool install git+https://github.com/rspcunningham/image-eval.git`

The install builds the embedded ROI selector from Swift sources and currently targets Apple
Silicon macOS.

## Workflow

Create or edit an ROI template from a base image:

```bash
image-eval init samples/raw.npy template.json --groups 4-7 --elements 1-6
```

Evaluate the raw image:

```bash
image-eval eval --base-url samples/raw.npy --template template.json --subject-url samples/raw.npy --out outputs/raw
```

Evaluate the reconstruction:

```bash
image-eval eval --base-url samples/raw.npy --template template.json --subject-url samples/reconstruction.npy --out outputs/reconstruction
```

Each evaluation writes `report.json`. Use `--json` to print the JSON report to stdout.
