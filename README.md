# image-eval

Image evaluation workflows for USAF-style MTF work.

The current implementation is focused on template initialization:

- `native/ROISelector`: macOS AppKit ROI selection, writes a template JSON file
- `src/image_eval/cli.py`: tiny Python entry point that launches the Swift CLI
- `template_schema.md`: shared template JSON schema
- `samples/`: sample NumPy arrays for local testing

The registration and calculation phases are expected to be Python-heavy, but
they have not been implemented yet.

## Setup

Install the Python environment:

```bash
uv sync
```

The Swift ROI selector is built on demand by SwiftPM when the CLI runs.

## Template Initialization

Run the native macOS ROI selector through the project CLI:

```bash
uv run image-eval samples/reconstruction.npy template.json \
  --groups 3-7 \
  --elements 1-6
```

This is equivalent to running the Swift executable directly:

```bash
uv run -- swift run --package-path native/ROISelector ROISelector \
  samples/reconstruction.npy template.json \
  --groups 3-7 \
  --elements 1-6
```

If the output template already exists, its saved `bar_rois` list and ROI
rectangles are reused. If the output template does not exist yet, `--groups`
and `--elements` are required to create the initial bar ROI list.

`--groups` and `--elements` accept comma-separated values and ranges, such as
`4,5,7`, `4-7`, or `-1..3`. Orientations are always `X,Y`.

For negative group ranges, use the equals form, for example `--groups=-1..3`,
so the CLI does not interpret the value as another option.

The ROI selector uses template schema version 2. See `template_schema.md`.

The AppKit canvas uses:

- current ROI: shown in the canvas HUD and selected in the sidebar
- saved ROIs: shown as muted context rectangles
- left drag: draw a rectangle
- drag inside active rectangle: move it
- drag an active edge or corner: resize it
- scroll, pinch, `+`, or `-`: zoom
- right/middle drag or space-drag: pan
- `Enter`: advance to the next ROI
- `Delete`: clear the active ROI
