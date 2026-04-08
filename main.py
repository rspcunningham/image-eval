import mtf_calc
import numpy as np
from numpy.typing import NDArray

from mtf_calc.models import Anchor, BarSection, Dim, FitResult, NormRegion, Roi, RoiConfig, ScaleGroup

SOURCE_PATH = "example-data.npy"
ROI_CONFIG_PATH = "roi_config.json"
DEFAULT_SCALE_GROUPS: tuple[ScaleGroup, ...] = (4, )
PROFILE_DIMS: tuple[Dim, ...] = ("X", "Y")
ELEMENTS_PER_GROUP = range(1, 7)
DEFAULT_HARMONICS = 3
REUSE_SAVED_ROI_CONFIG = False
SHOW_ANCHOR_PREVIEW = False


def load_raw_image(path: str = SOURCE_PATH) -> NDArray[np.float32]:
    return mtf_calc.io.load_source(path)


def find_image_anchor(raw_image: NDArray[np.float32]) -> Anchor:
    return mtf_calc.anchor.find_anchor(raw_image)


def maybe_show_anchor_preview(raw_image: NDArray[np.float32], anchor: Anchor) -> None:
    if SHOW_ANCHOR_PREVIEW:
        mtf_calc.viz.show_anchor(raw_image, anchor)


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


def build_roi_config(
    raw_image: NDArray[np.float32],
    *,
    anchor: Anchor,
    scale_groups: list[ScaleGroup],
) -> RoiConfig:
    norm_rois = select_normalization_rois(raw_image)
    bar_rois = select_bar_rois(raw_image, scale_groups=scale_groups)
    roi_config = RoiConfig(
        anchor=anchor,
        scale_groups=scale_groups,
        bar_rois=bar_rois,
        norm_rois=norm_rois,
    )
    mtf_calc.io.save_roi_config(roi_config, ROI_CONFIG_PATH)
    return roi_config


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


def load_analysis_rois(
    *,
    raw_image: NDArray[np.float32],
    anchor: Anchor,
    reuse_saved_roi_config: bool,
    scale_groups: list[ScaleGroup],
) -> tuple[RoiConfig, dict[BarSection, Roi], dict[NormRegion, Roi]]:
    if reuse_saved_roi_config:
        roi_config = mtf_calc.io.load_roi_config(ROI_CONFIG_PATH)
        bar_rois, norm_rois = mtf_calc.io.translate_rois_from_anchor(roi_config, anchor)
        return roi_config, bar_rois, norm_rois

    roi_config = build_roi_config(
        raw_image,
        anchor=anchor,
        scale_groups=scale_groups,
    )
    return roi_config, roi_config.bar_rois, roi_config.norm_rois


def main(*, reuse_saved_roi_config: bool = REUSE_SAVED_ROI_CONFIG) -> None:
    raw_image = load_raw_image()
    anchor = find_image_anchor(raw_image)
    maybe_show_anchor_preview(raw_image, anchor)

    roi_config, bar_rois, norm_rois = load_analysis_rois(
        raw_image=raw_image,
        anchor=anchor,
        reuse_saved_roi_config=reuse_saved_roi_config,
        scale_groups=list(DEFAULT_SCALE_GROUPS),
    )
    fit_results = fit_profiles_from_rois(
        raw_image,
        scale_groups=roi_config.scale_groups,
        bar_rois=bar_rois,
        norm_rois=norm_rois,
    )
    mtf_result = mtf_calc.mtf.compute(fit_results)
    mtf_calc.viz.show_mtf_graph(mtf_result)


if __name__ == "__main__":
    main()
