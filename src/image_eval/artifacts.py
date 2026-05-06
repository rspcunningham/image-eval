from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from image_eval.dqe_plot import save_dqe_curve_plot
from image_eval.evaluation import EvaluationResult, evaluation_result_to_dict
from image_eval.mtf_plot import save_bar_roi_fit_plot, save_mtf_curve_plot
from image_eval.mtf_results import roi_mtf_value
from image_eval.nps_plot import save_nps_curve_plot, save_nps_spectrum_plot
from image_eval.registration_artifacts import (
    RegistrationArtifactPaths,
    save_registration_artifact_plots,
)


DEFAULT_PLOTS = frozenset({"mtf", "nps", "dqe", "registration", "roi-fits", "nps-spectra"})
PLOT_ALIASES = {
    "roi_fits": "roi-fits",
    "nps_spectra": "nps-spectra",
}


class EvaluationArtifactPaths(NamedTuple):
    output_dir: Path
    report_json_path: Path | None
    plot_paths: dict[str, Path | list[Path] | RegistrationArtifactPaths]


def write_evaluation_artifacts(
    result: EvaluationResult,
    output_dir: Path,
    *,
    plots: set[str] | frozenset[str] = DEFAULT_PLOTS,
    write_json: bool = True,
) -> EvaluationArtifactPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_plots = normalize_plot_kinds(plots)
    plot_paths: dict[str, Path | list[Path] | RegistrationArtifactPaths] = {}

    report_json_path = output_dir / "report.json" if write_json else None
    report_dict = evaluation_result_to_dict(result)
    if report_json_path is not None:
        with report_json_path.open("w") as file:
            json.dump(report_dict, file, indent=2)
            file.write("\n")

    if "mtf" in normalized_plots:
        path = output_dir / "mtf.png"
        save_mtf_curve_plot(result.mtf_report.results, path)
        plot_paths["mtf"] = path

    if "nps" in normalized_plots:
        path = output_dir / "nps.png"
        save_nps_curve_plot(
            result.nps_report.results,
            path,
            frequency_unit=result.nps_report.frequency_calibration.unit,
        )
        plot_paths["nps"] = path

    if "dqe" in normalized_plots:
        path = output_dir / "dqe.png"
        save_dqe_curve_plot(result.dqe_report.results, path)
        plot_paths["dqe"] = path

    if "roi-fits" in normalized_plots:
        plot_paths["roi-fits"] = _write_roi_fit_plots(result, output_dir / "roi_fits")

    if "nps-spectra" in normalized_plots:
        plot_paths["nps-spectra"] = _write_nps_spectrum_plots(
            result,
            output_dir / "nps_spectra",
        )

    if "registration" in normalized_plots:
        plot_paths["registration"] = _write_registration_artifacts(
            result,
            output_dir / "registration",
        )

    return EvaluationArtifactPaths(
        output_dir=output_dir,
        report_json_path=report_json_path,
        plot_paths=plot_paths,
    )


def normalize_plot_kinds(plots: set[str] | frozenset[str]) -> set[str]:
    normalized = {PLOT_ALIASES.get(plot, plot) for plot in plots}
    unknown = normalized - DEFAULT_PLOTS
    if unknown:
        raise ValueError(f"unknown plot kind(s): {', '.join(sorted(unknown))}")
    return normalized


def _write_roi_fit_plots(result: EvaluationResult, roi_fit_dir: Path) -> list[Path]:
    roi_fit_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, fitted_profile in enumerate(result.mtf_report.fitted_profiles, start=1):
        roi = fitted_profile.roi_profile
        path = roi_fit_dir / (
            f"{index:03d}_g{roi.group}_e{roi.element}_{roi.orientation.lower()}_fit.png"
        )
        save_bar_roi_fit_plot(
            fitted_profile,
            path,
            mtf_value=roi_mtf_value(fitted_profile),
        )
        paths.append(path)
    return paths


def _write_nps_spectrum_plots(result: EvaluationResult, spectrum_dir: Path) -> list[Path]:
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for spectrum in result.nps_report.spectra:
        path = spectrum_dir / f"{spectrum.roi_name}_2d.png"
        save_nps_spectrum_plot(
            spectrum,
            path,
            frequency_unit=result.nps_report.frequency_calibration.unit,
        )
        paths.append(path)
    return paths


def _write_registration_artifacts(
    result: EvaluationResult,
    registration_dir: Path,
) -> RegistrationArtifactPaths:
    registration_dir.mkdir(parents=True, exist_ok=True)
    registration_json_path = registration_dir / "registration.json"
    registered_template_path = registration_dir / "registered_template.json"

    with registration_json_path.open("w") as file:
        json.dump(evaluation_result_to_dict(result)["registration"], file, indent=2)
        file.write("\n")
    with registered_template_path.open("w") as file:
        json.dump(evaluation_result_to_dict(result)["registered_template"], file, indent=2)
        file.write("\n")

    roi_overlay_path, image_overlay_path = save_registration_artifact_plots(
        result.base_image,
        result.subject_image,
        result.registered_template,
        result.registration["transform_subject_to_base"],
        registration_dir,
    )

    return RegistrationArtifactPaths(
        registration_dir=registration_dir,
        registration_json_path=registration_json_path,
        registered_template_path=registered_template_path,
        roi_overlay_path=roi_overlay_path,
        image_overlay_path=image_overlay_path,
    )
