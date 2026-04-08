import mtf_calc
import numpy as np
from numpy.typing import NDArray

from mtf_calc.models import BarSection, Dim, FitResult, NormRegion, Roi

SOURCE_PATH = "example-data.npy"
ROI_CONFIG_PATH = "roi_config.json"
PROFILE_DIMS: tuple[Dim, ...] = ("X", "Y")
ELEMENTS_PER_GROUP = range(1, 7)
DEFAULT_HARMONICS = 3


def fit_profiles_from_rois(
    raw_image: NDArray[np.float32],
    *,
    scale_groups: list[int],
    bar_rois: dict[BarSection, Roi],
    norm_rois: dict[NormRegion, Roi],
    n_harmonics: int = DEFAULT_HARMONICS,
) -> dict[BarSection, FitResult]:
    fit_results: dict[BarSection, FitResult] = {}

    for group in scale_groups:
        for element in ELEMENTS_PER_GROUP:
            for dim in PROFILE_DIMS:
                section = BarSection(group, element, dim)
                bar_roi = bar_rois.get(section)
                if bar_roi is None:
                    continue

                profile = mtf_calc.profiles.extract(
                    raw_image,
                    bar_roi=bar_roi,
                    norm_rois=norm_rois,
                    dim=dim,
                )
                fit_results[section] = mtf_calc.profiles.fit(
                    profile,
                    norm_rois=norm_rois,
                    n_harmonics=n_harmonics,
                )

    return fit_results


def main() -> None:
    try:
        raw_image = mtf_calc.io.load_source(SOURCE_PATH)
        anchor = mtf_calc.anchor.find_anchor(raw_image)
        roi_config = mtf_calc.io.load_roi_config(ROI_CONFIG_PATH)
        bar_rois, norm_rois = mtf_calc.io.translate_rois_from_anchor(roi_config, anchor)
        mtf_calc.viz.show_rois(
            raw_image,
            anchor=anchor,
            norm_rois=norm_rois,
            bar_rois=bar_rois,
        )

        fit_results = fit_profiles_from_rois(
            raw_image,
            scale_groups=list(roi_config.scale_groups),
            bar_rois=bar_rois,
            norm_rois=norm_rois,
        )
        mtf_result = mtf_calc.mtf.compute(fit_results)
        mtf_calc.viz.show_mtf_graph(mtf_result)
    finally:
        mtf_calc.viz.close()


if __name__ == "__main__":
    main()
