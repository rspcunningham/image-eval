# image-eval

Small, explicit image evaluation workflows for USAF-style MTF work.

The project is intentionally split into three top-level scripts:

- `initialize.py`: interactive OpenCV ROI selection, writes `template.json`
- `register.py`: future headless registration, writes `registration.json`
- `evaluate.py`: future headless calculations and plotting

## Initialize

The repo includes two sample arrays:

- `samples/reconstruction.npy`: original float32 reconstruction sample
- `samples/raw_object_intensity.npy`: `abs(raw_object) ** 2` from the complex64 sample

Edit the constants at the top of `initialize.py` if you want a different source image, then run:

```bash
uv run python initialize.py
```

The OpenCV picker uses:

- left drag: draw a rectangle
- drag inside rectangle: move it
- drag an edge or corner: resize it
- mouse wheel: zoom
- middle drag: pan
- `f`: fit image
- `r`: reset current rectangle
- `Enter`: confirm
- `Esc` or `q`: cancel
