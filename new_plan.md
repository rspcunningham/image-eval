```python
import mtf_calc
from mtf_calc.models import BarSection


# step 0: load data
raw_image = mtf_calc.io.load_source("path/to/image.npy")


# step 1: find anchor point
anchor = mtf_calc.anchor.find_anchor(raw_image)
mtf_calc.viz.show_anchor(raw_image, anchor)  # optional visual check


# step 2: identify scale groups
# these may be known ahead of time, selected by a user, or chosen by an agent
scale_groups = [4, 5, 6, 7]


# step 3: bar ROIs
bar_rois = {}

for group in scale_groups:
    for element in range(1, 7):
        # X
        bar_rois[BarSection(group, element, "X")] = mtf_calc.select.select_roi(raw_image)
        # Y
        bar_rois[BarSection(group, element, "Y")] = mtf_calc.select.select_roi(raw_image)


# step 4: normalization ROIs
black_roi = mtf_calc.select.select_roi(raw_image)
white_roi = mtf_calc.select.select_roi(raw_image, size_ref=black_roi)

norm_rois = {
    "black": black_roi,
    "white": white_roi,
}


# save the reusable selection config
roi_config = mtf_calc.config.RoiConfig(
    anchor=anchor,
    scale_groups=scale_groups,
    bar_rois=bar_rois,
    norm_rois=norm_rois,
)
mtf_calc.io.save_roi_config(roi_config, "roi_config.json")


# step 5: extract profiles and normalization
profiles = mtf_calc.profiles.extract(
    raw_image,
    bar_rois=bar_rois,
    norm_rois=norm_rois,
)


# step 6: curve fitting
# if crop parameters are still needed, treat them as explicit fit options
n_harmonics = 5
results = {}

for section, profile in profiles.bar_profiles.items():
    results[section.key()] = mtf_calc.fit.analyze_profile(
        profile,
        normalization=profiles.normalization,
        n_harmonics=n_harmonics,
    )


# step 7: final MTF result and visualization
mtf_result = mtf_calc.mtf.compute(results)
mtf_calc.viz.show_mtf_graph(mtf_result)
```
