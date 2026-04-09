import mtf_calc

from mtf_calc.models import BarSection, Dim, FitResult, ScaleGroup

SOURCE_PATH = "example-data.npy"
ROI_CONFIG_PATH = "roi_config.json"
MTF_RESULT_PATH = "mtf_result.csv"
MTF_PLOT_PATH = "mtf_plot.png"
PROFILE_DIMS: tuple[Dim, ...] = ("X", "Y")
ELEMENTS_PER_GROUP = range(1, 7)
DEFAULT_HARMONICS = 3

raw_image = mtf_calc.io.load_source(SOURCE_PATH)
anchor = mtf_calc.anchor.find_anchor(raw_image)
roi_config = mtf_calc.io.load_roi_config(ROI_CONFIG_PATH)
bar_rois, norm_rois = mtf_calc.io.translate_rois_from_anchor(roi_config, anchor)

scale_groups: list[ScaleGroup] = list(roi_config.scale_groups)
sections = [
    BarSection(group, element, dim)
    for group in scale_groups
    for element in ELEMENTS_PER_GROUP
    for dim in PROFILE_DIMS
]

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
