import mtf_calc
import numpy as np
from numpy.typing import NDArray

from mtf_calc.models import BarSection, Dim, FitResult, NormRegion, Roi, RoiConfig, ScaleGroup

SOURCE_PATH = "example-data.npy"
ROI_CONFIG_PATH = "roi_config.json"
DEFAULT_SCALE_GROUPS: tuple[ScaleGroup, ...] = (4,)
PROFILE_DIMS: tuple[Dim, ...] = ("X", "Y")
ELEMENTS_PER_GROUP = range(1, 7)
DEFAULT_HARMONICS = 3
SHOW_ANCHOR_PREVIEW = False


def select_normalization_rois(raw_image: NDArray[np.float32]) -> dict[NormRegion, Roi]:
    black_roi = mtf_calc.select.select_roi(
        raw_image,
        prompt="Select the black normalization ROI from a dark background patch with no bars crossing it.",
    )
    white_roi = mtf_calc.select.select_roi(
        raw_image,
        size_ref=black_roi,
        prompt="Select the white normalization ROI from a bright background patch. Match the black ROI region type and size.",
    )
    return {
        0: black_roi,
        1: white_roi,
    }


def select_bar_rois(
    raw_image: NDArray[np.float32],
    *,
    scale_groups: list[ScaleGroup],
) -> dict[BarSection, Roi]:
    bar_rois: dict[BarSection, Roi] = {}

    for group in scale_groups:
        for element in ELEMENTS_PER_GROUP:
            for dim in PROFILE_DIMS:
                section = BarSection(group, element, dim)
                bar_rois[section] = mtf_calc.select.select_roi(
                    raw_image,
                    prompt=(
                        f"Select the ROI for Group {group}, Element {element}, "
                        f"the {dim}-directed profile."
                    ),
                )

    return bar_rois


def fit_profiles_from_rois(
    raw_image: NDArray[np.float32],
    *,
    scale_groups: list[ScaleGroup],
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
        if SHOW_ANCHOR_PREVIEW:
            mtf_calc.viz.show_anchor(raw_image, anchor)

        norm_rois = select_normalization_rois(raw_image)
        bar_rois = select_bar_rois(raw_image, scale_groups=list(DEFAULT_SCALE_GROUPS))
        roi_config = RoiConfig(
            anchor=anchor,
            scale_groups=list(DEFAULT_SCALE_GROUPS),
            bar_rois=bar_rois,
            norm_rois=norm_rois,
        )
        mtf_calc.io.save_roi_config(roi_config, ROI_CONFIG_PATH)

        fit_results = fit_profiles_from_rois(
            raw_image,
            scale_groups=roi_config.scale_groups,
            bar_rois=bar_rois,
            norm_rois=norm_rois,
        )
        mtf_result = mtf_calc.mtf.compute(fit_results)
        mtf_calc.viz.show_mtf_graph(mtf_result)
    finally:
        mtf_calc.viz.close()


if __name__ == "__main__":
    main()
