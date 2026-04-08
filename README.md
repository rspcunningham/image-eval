# mtf-calc

Interactive MTF extraction for USAF 1951 resolution target images.

This project loads a grayscale source image, finds a reference anchor square, lets you select normalization and bar ROIs, fits 1D bar profiles with an odd-harmonic model, and plots the resulting modulation transfer function (MTF) across the selected target elements.

## What It Does

- Loads image data from a NumPy array file.
- Detects a large dark reference square and refines its corners into an anchor ROI.
- Lets you interactively select:
  - one black normalization ROI
  - one white normalization ROI
  - one ROI per `(group, element, dim)` bar section you want to analyze
- Saves ROI selections to `roi_config.json` so they can be reused on later runs.
- Translates saved ROIs from the saved anchor position to the current image anchor.
- Fits each extracted profile with a harmonic bar model.
- Computes first-harmonic MTF values for X and Y orientations and displays a graph.

## Project Layout

- [`main.py`](main.py): initial interactive workflow that captures normalization and bar ROIs.
- [`auto.py`](auto.py): saved-config workflow that loads ROI selections from disk and previews the translated ROIs before fitting.
- [`src/mtf_calc/anchor.py`](src/mtf_calc/anchor.py): anchor-square detection and refinement.
- [`src/mtf_calc/select.py`](src/mtf_calc/select.py): ROI-selection wrapper.
- [`src/mtf_calc/profiles.py`](src/mtf_calc/profiles.py): profile extraction and harmonic fitting.
- [`src/mtf_calc/mtf.py`](src/mtf_calc/mtf.py): conversion from fit results to MTF points.
- [`src/mtf_calc/viz.py`](src/mtf_calc/viz.py): visualization host client for ROI selection and plotting.
- [`usaf1951-spec.md`](usaf1951-spec.md): reference notes on the USAF 1951 target geometry.

## Requirements

- Python 3.12+
- `uv` for dependency management and execution

Dependencies are declared in [`pyproject.toml`](pyproject.toml).

## Setup

Install the environment with:

```bash
uv sync
```

## Running

Run the initial interactive workflow with:

```bash
uv run python main.py
```

Run the saved-config workflow with:

```bash
uv run python auto.py
```

Both example scripts expose a few module-level constants near the top of the file:

- `SOURCE_PATH`: source `.npy` image to analyze.
- `ROI_CONFIG_PATH`: path used to save and reload ROI selections.
- `DEFAULT_SCALE_GROUPS`: USAF groups to analyze in `main.py`.
- `DEFAULT_HARMONICS`: odd-harmonic count used by the profile fit.

[`main.py`](main.py) also exposes `SHOW_ANCHOR_PREVIEW` for an optional anchor overlay before ROI selection.

## Typical Workflow

### First run: create ROI selections

```bash
uv run python main.py
```

You will be prompted to:

1. Confirm or inspect the detected anchor square.
2. Select a black normalization patch.
3. Select a white normalization patch.
4. Select one ROI for each configured group, element, and profile direction.

The selections are saved to `roi_config.json`.

### Later runs: reuse saved ROIs

```bash
uv run python auto.py
```

The saved ROI geometry will be loaded from `roi_config.json`, shifted using the newly detected anchor position, previewed in a read-only ROI review window, and then used for profile fitting and MTF computation.

## Input Expectations

- The source image is currently loaded with `numpy.load(...)`, so the expected input is a 2D NumPy array saved as `.npy`.
- The anchor finder expects a prominent dark square in the image.
- Normalization ROIs should be placed on clean dark and bright background patches with no bars crossing them.
- Bar ROIs should tightly cover the intended USAF element in the requested X or Y direction.

## Fitting Notes

Profile fitting in [`src/mtf_calc/profiles.py`](src/mtf_calc/profiles.py) does the following:

- averages the ROI across one axis to produce a 1D profile
- normalizes the profile using the black and white reference ROIs
- estimates an initial bar period
- performs a coarse search over period and phase
- refines the solution with `scipy.optimize.minimize`
- computes MTF from the first fitted odd harmonic

## Development

Type checking:

```bash
uv run basedpyright
```

Basic syntax check:

```bash
uv run python -m py_compile main.py auto.py src/mtf_calc/*.py
```

## Current Limitations

- The scripts are still configured by editing module-level constants; there is no CLI yet.
- Input loading is currently limited to NumPy `.npy` arrays.
- ROI selection is interactive and requires the visualization host UI.
- `roi_config.json` is local workflow state and is not tracked in git by default.
