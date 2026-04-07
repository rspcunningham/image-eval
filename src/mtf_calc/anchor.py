import numpy as np
from numpy.typing import NDArray

from mtf_calc.models import Anchor


def find_anchor(raw_image: NDArray[np.float32]) -> Anchor:
    del raw_image
    raise NotImplementedError
