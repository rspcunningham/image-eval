from collections.abc import Mapping
import numpy as np
from numpy.typing import NDArray

from mtf_calc.models import Dim, FitResult, NormRegion, Profile, Roi


def extract(
    raw_image: NDArray[np.float32],
    *,
    bar_roi: Roi,
    norm_rois: Mapping[NormRegion, Roi],
    dim: Dim,
) -> Profile:
    del raw_image, bar_roi, norm_rois, dim
    raise NotImplementedError


def fit(
    profile: Profile,
    *,
    norm_rois: Mapping[NormRegion, Roi],
    n_harmonics: int,
) -> FitResult:
    del profile, norm_rois, n_harmonics
    raise NotImplementedError
