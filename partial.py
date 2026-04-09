import mtf_calc

from mtf_calc.models import BarSection, Dim, FitResult, NormRegion, Roi, RoiConfig, ScaleGroup

SOURCE_PATH = "example-data.npy"
ROI_CONFIG_PATH = "roi_config.json"
MTF_RESULT_PATH = "mtf_result.csv"
MTF_PLOT_PATH = "mtf_plot.png"
DEFAULT_SCALE_GROUPS: tuple[ScaleGroup, ...] = (4, 5, 6, 7)
PROFILE_DIMS: tuple[Dim, ...] = ("X", "Y")
ELEMENTS_PER_GROUP = range(1, 7)
DEFAULT_HARMONICS = 3

scale_groups: list[ScaleGroup] = list(DEFAULT_SCALE_GROUPS)
sections = [
    BarSection(group, element, dim)
    for group in scale_groups
    for element in ELEMENTS_PER_GROUP
    for dim in PROFILE_DIMS
]

raw_image = mtf_calc.io.load_source(SOURCE_PATH)
anchor = mtf_calc.anchor.find_anchor(raw_image)
roi_config = mtf_calc.io.load_roi_config(ROI_CONFIG_PATH)
saved_bar_rois, saved_norm_rois = mtf_calc.io.translate_rois_from_anchor(
    roi_config,
    anchor,
)

norm_rois: dict[NormRegion, Roi] = dict(saved_norm_rois)
black_roi = norm_rois.get(0)
if black_roi is None:
    black_roi = mtf_calc.select.select_roi(
        raw_image,
        prompt="Select the black normalization ROI from a dark background patch with no bars crossing it.",
    )
    norm_rois[0] = black_roi
if 1 not in norm_rois:
    norm_rois[1] = mtf_calc.select.select_roi(
        raw_image,
        size_ref=black_roi,
        prompt="Select the white normalization ROI from a bright background patch. Match the black ROI region type and size.",
    )

bar_rois: dict[BarSection, Roi] = dict(saved_bar_rois)
for section in sections:
    if section in bar_rois:
        continue
    bar_rois[section] = mtf_calc.select.select_roi(
        raw_image,
        prompt=(
            f"Select the ROI for Group {section.group}, Element {section.element}, "
            f"the {section.dim}-directed profile."
        ),
    )

if (
    norm_rois != saved_norm_rois
    or bar_rois != saved_bar_rois
    or roi_config.scale_groups != scale_groups
):
    mtf_calc.io.save_roi_config(
        RoiConfig(
            anchor=anchor,
            scale_groups=scale_groups,
            bar_rois=bar_rois,
            norm_rois=norm_rois,
        ),
        ROI_CONFIG_PATH,
    )

mtf_calc.viz.show_rois(
    raw_image,
    anchor=anchor,
    norm_rois=norm_rois,
    bar_rois=bar_rois,
)

fit_results: dict[BarSection, FitResult] = {}
for section in sections:
    bar_roi = bar_rois.get(section)
    if bar_roi is None:
        continue
    profile = mtf_calc.profiles.extract(
        raw_image,
        bar_roi=bar_roi,
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
