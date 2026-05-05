# image-eval

Small, explicit image evaluation workflows for USAF-style MTF work.

The project is split into a native ROI authoring tool and headless analysis
scripts:

- `native/ROISelector`: macOS AppKit ROI selection, writes a template JSON file
- `register.py`: future headless registration, writes `registration.json`
- `evaluate.py`: future headless calculations and plotting

## ROI Selection

The repo includes sample arrays:

- `samples/reconstruction.npy`: original float32 reconstruction sample

Run the native macOS ROI selector through the project CLI:

```bash
uv run image-eval initialize samples/reconstruction.npy template.json \
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
