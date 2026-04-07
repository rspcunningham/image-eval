import numpy as np
from numpy.typing import NDArray

from mtf_calc.models import Anchor, MtfResult


def show_anchor(raw_image: NDArray[np.float32], anchor: Anchor) -> None:
    del raw_image, anchor
    raise NotImplementedError


def show_mtf_graph(mtf_result: MtfResult) -> None:
    del mtf_result
    raise NotImplementedError
