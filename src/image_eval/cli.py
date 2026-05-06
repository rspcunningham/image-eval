from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from image_eval.artifacts import DEFAULT_PLOTS, normalize_plot_kinds, write_evaluation_artifacts
from image_eval.evaluation import evaluate_image, evaluation_result_to_dict
from image_eval.sources import load_image_source, load_template_source


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="image-eval",
        description=(
            "Evaluate a subject .npy image from an explicit base image and ROI template, "
            "returning JSON data and optional plots."
        ),
    )
    parser.add_argument("--base-url", required=True, help="Base/reference .npy path or URL.")
    parser.add_argument("--template", required=True, help="ROI template JSON path or URL.")
    parser.add_argument("--subject-url", required=True, help="Subject .npy path or URL.")
    parser.add_argument("--out", type=Path, help="Directory for report.json and artifacts.")
    parser.add_argument("--json", action="store_true", help="Print the JSON report to stdout.")
    parser.add_argument("--no-plots", action="store_true", help="Do not write plot artifacts.")
    parser.add_argument(
        "--plots",
        help=(
            "Comma-separated plot kinds to write: "
            "mtf,nps,dqe,registration,roi-fits,nps-spectra."
        ),
    )
    args = parser.parse_args(argv)

    try:
        if args.no_plots and args.plots:
            raise ValueError("--no-plots cannot be combined with --plots")
        plot_kinds = _plot_kinds_from_args(args.plots, no_plots=args.no_plots)

        base_image = load_image_source(args.base_url)
        template = load_template_source(args.template)
        subject_image = load_image_source(args.subject_url)
        result = evaluate_image(
            base_image=base_image,
            template=template,
            subject_image=subject_image,
        )

        if args.out is not None:
            write_evaluation_artifacts(
                result,
                args.out,
                plots=plot_kinds,
                write_json=True,
            )

        if args.json or args.out is None:
            json.dump(evaluation_result_to_dict(result), sys.stdout, indent=2)
            sys.stdout.write("\n")
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"image-eval: error: {error}", file=sys.stderr)
        return 1

    return 0


def _plot_kinds_from_args(plots: str | None, *, no_plots: bool) -> set[str] | frozenset[str]:
    if no_plots:
        return set()
    if plots is None:
        return DEFAULT_PLOTS
    return normalize_plot_kinds({plot.strip() for plot in plots.split(",") if plot.strip()})


if __name__ == "__main__":
    raise SystemExit(main())
