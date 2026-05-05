# Template Schema

This document describes the template JSON schema.

Templates must match this schema exactly. Unknown fields are invalid, and
nullable fields must be present with a JSON `null` value when unset.

The template stores the image-space regions needed for MTF evaluation:

- two normalization ROIs
- one ROI for each selected bar group, element, and orientation

ROI coordinates are stored as integer half-open pixel bounds:

```json
{
  "x0": 120,
  "y0": 80,
  "x1": 180,
  "y1": 112
}
```

These bounds map directly to NumPy slicing:

```python
roi = image[y0:y1, x0:x1]
```

`x0` and `y0` are inclusive. `x1` and `y1` are exclusive. Therefore:

```text
width = x1 - x0
height = y1 - y0
```

All ROI bounds must be integers. Empty or unselected ROIs are represented as
`null`.

## Top-Level Fields

`base_image_path`
: Path to the base image used by the ROI selector when this template was
  authored or last saved. The Swift app writes this as an absolute path.

`source_image`
: Metadata for the source image this template was authored against.

`normalization_rois`
: Object containing `black` and `white` normalization ROI bounds. Each value is
  either an ROI bounds object or `null`.

`bar_rois`
: Array of bar ROI records. The array itself is the source of truth for which
  group, element, and orientation combinations are part of the template.

## Source Image

```json
{
  "path": "/path/to/reconstruction.npy",
  "width": 1024,
  "height": 1024
}
```

`path` is a convenience hint for reopening the image. `width` and `height` are
required because ROI bounds are only meaningful against the image dimensions
used when the template was authored.

## Bar ROI

Each bar ROI has:

`group`
: Integer USAF group.

`element`
: Integer USAF element.

`orientation`
: `"X"` or `"Y"`.

`rect`
: Required ROI bounds object, or `null` if unset.

## Example

```json
{
  "base_image_path": "/path/to/reconstruction.npy",
  "source_image": {
    "path": "/path/to/reconstruction.npy",
    "width": 1024,
    "height": 1024
  },
  "normalization_rois": {
    "black": {
      "x0": 20,
      "y0": 30,
      "x1": 80,
      "y1": 90
    },
    "white": null
  },
  "bar_rois": [
    {
      "group": 4,
      "element": 1,
      "orientation": "X",
      "rect": {
        "x0": 220,
        "y0": 310,
        "x1": 280,
        "y1": 340
      }
    },
    {
      "group": 4,
      "element": 1,
      "orientation": "Y",
      "rect": null
    }
  ]
}
```
