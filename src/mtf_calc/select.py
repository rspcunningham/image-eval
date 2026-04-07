from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mtf_calc.models import Roi
from mtf_calc.viz import select_roi as _select_roi


def select_roi(raw_image: NDArray[np.float32], size_ref: Roi | None = None) -> Roi:
    return _select_roi(raw_image, size_ref=size_ref)
