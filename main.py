import mtf_calc

from mtf_calc.models import (
    BarSection,
    Dim,
    FitResult,
    NormRegion,
    RoiConfig,
    Roi,
    ScaleGroup,
)

SOURCE_PATH = "example-data.npy"
ROI_CONFIG_PATH = "roi_config.json"
MTF_RESULT_PATH = "mtf_result.csv"
MTF_PLOT_PATH = "mtf_plot.png"
DEFAULT_SCALE_GROUPS: tuple[ScaleGroup, ...] = (4, 5, 6, 7)
PROFILE_DIMS: tuple[Dim, ...] = ("X", "Y")
ELEMENTS_PER_GROUP = range(1, 7)
DEFAULT_HARMONICS = 3
SHOW_ANCHOR_PREVIEW = False

scale_groups: list[ScaleGroup] = list(DEFAULT_SCALE_GROUPS)
sections = [
    BarSection(group, element, dim)
    for group in scale_groups
    for element in ELEMENTS_PER_GROUP
    for dim in PROFILE_DIMS
]

raw_image = mtf_calc.io.load_source(SOURCE_PATH)
anchor = mtf_calc.anchor.find_anchor(raw_image)
if SHOW_ANCHOR_PREVIEW:
    mtf_calc.viz.show_anchor(raw_image, anchor)

black_roi = mtf_calc.select.select_roi(
    raw_image,
    prompt="Select the black normalization ROI from a dark background patch with no bars crossing it.",
)
white_roi = mtf_calc.select.select_roi(
    raw_image,
    size_ref=black_roi,
    prompt="Select the white normalization ROI from a bright background patch. Match the black ROI region type and size.",
)
norm_rois: dict[NormRegion, Roi] = {
    0: black_roi,
    1: white_roi,
}

bar_rois: dict[BarSection, Roi] = {
    section: mtf_calc.select.select_roi(
        raw_image,
        prompt=(
            f"Select the ROI for Group {section.group}, Element {section.element}, "
            f"the {section.dim}-directed profile."
        ),
    )
    for section in sections
}

mtf_calc.io.save_roi_config(
    RoiConfig(
        anchor=anchor,
        scale_groups=scale_groups,
        bar_rois=bar_rois,
        norm_rois=norm_rois,
    ),
    ROI_CONFIG_PATH,
)

fit_results: dict[BarSection, FitResult] = {}
for section in sections:
    profile = mtf_calc.profiles.extract(
        raw_image,
        bar_roi=bar_rois[section],
        norm_rois=norm_rois,
        dim=section.dim,
    )
    fit_results[section] = mtf_calc.profiles.fit(
        profile,
        norm_rois=norm_rois,
        n_harmonics=DEFAULT_HARMONICS,
    )

mtf_result = mtf_calc.mtf.compute(fit_results)
mtf_calc.viz.close()
mtf_calc.io.save_mtf_result_csv(mtf_result, MTF_RESULT_PATH)
mtf_calc.viz.show_mtf_graph(mtf_result, output_path=MTF_PLOT_PATH)
