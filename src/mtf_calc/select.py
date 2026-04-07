import numpy as np
from numpy.typing import NDArray

from mtf_calc.models import Roi


def select_roi(raw_image: NDArray[np.float32], size_ref: Roi | None = None) -> Roi:
    del raw_image, size_ref
    raise NotImplementedError
