import mtf_calc

from mtf_calc.models import BarSection, FitResult, NormRegion, Roi, RoiConfig, ScaleGroup

# step 0: load data
raw_image = mtf_calc.io.load_source("path/to/image.npy")


# step 1: find anchor point
anchor = mtf_calc.anchor.find_anchor(raw_image)
mtf_calc.viz.show_anchor(raw_image, anchor)  # optional visual check


# step 2: identify scale groups
# these may be known ahead of time, selected by a user, or chosen by an agent
scale_groups: list[ScaleGroup] = [4, 5, 6, 7]

# step 3: normalization ROIs
black_roi = mtf_calc.select.select_roi(raw_image)
white_roi = mtf_calc.select.select_roi(raw_image, size_ref=black_roi)

norm_rois: dict[NormRegion, Roi] = {
    "black": black_roi,
    "white": white_roi,
}

# step 4: bar ROIs and curve fitting
bar_rois: dict[BarSection, Roi] = {}
results: dict[BarSection, FitResult] = {}

for group in scale_groups:
    for element in range(1, 7):

        n_harmonics = 6

        for dim in ("X", "Y"):

            bar_roi = mtf_calc.select.select_roi(raw_image)

            bar_rois[BarSection(group, element, dim)] = bar_roi

            profile = mtf_calc.profiles.extract(raw_image, bar_roi=bar_roi, norm_rois=norm_rois, dim=dim)


            results[BarSection(group, element, dim)] = mtf_calc.profiles.fit(
                profile,
                norm_rois=norm_rois,
                n_harmonics=n_harmonics,
            )


# step 5: save the reusable selection config
roi_config = RoiConfig(
    anchor=anchor,
    scale_groups=scale_groups,
    bar_rois=bar_rois,
    norm_rois=norm_rois,
)
mtf_calc.io.save_roi_config(roi_config, "roi_config.json")


# step 6: final MTF result and visualization
mtf_result = mtf_calc.mtf.compute(results)
mtf_calc.viz.show_mtf_graph(mtf_result)
