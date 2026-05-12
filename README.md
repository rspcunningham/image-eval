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
image-eval init base_image.npy template.json --groups 4-7 --elements 1-6
```

Evaluate an image against a base image and template:

```bash
image-eval eval --base-url base_image.npy --template template.json --subject-url subject_image.npy --json
```
