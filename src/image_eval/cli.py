from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Sequence

from image_eval.initialize import add_init_arguments, run_init


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-eval",
        description="Create ROI templates and evaluate .npy images.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create or edit an ROI template.")
    add_init_arguments(init_parser)
    init_parser.set_defaults(handler=run_init)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a subject image.")
    _add_eval_arguments(eval_parser)
    eval_parser.set_defaults(handler=_run_eval)

    return parser


def _add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Evaluate a subject .npy image from an explicit base image and ROI template, "
        "returning JSON data."
    )
    parser.add_argument("--base-url", required=True, help="Base/reference .npy path or URL.")
    parser.add_argument("--template", required=True, help="ROI template JSON path or URL.")
    parser.add_argument("--subject-url", required=True, help="Subject .npy path or URL.")
    parser.add_argument(
        "--norm",
        default="mean",
        type=_norm_argument,
        help=(
            "Normalization method: 'mean' or a quantile cutoff fraction. "
            "For example, 0.05 uses the 5th percentile black level and 95th "
            "percentile white level."
        ),
    )
    parser.add_argument(
        "--detail",
        action="append",
        choices=("fits", "all"),
        help="Include additional detail sections in the JSON report.",
    )


def _run_eval(args: argparse.Namespace) -> int:
    from image_eval.evaluation import evaluate_image, evaluation_result_to_dict
    from image_eval.sources import load_image_source, load_template_source

    try:
        base_image = load_image_source(args.base_url)
        template = load_template_source(args.template)
        subject_image = load_image_source(args.subject_url)
        result = evaluate_image(
            base_image=base_image,
            template=template,
            subject_image=subject_image,
            norm=args.norm,
        )

        json.dump(evaluation_result_to_dict(result, details=args.detail), sys.stdout, indent=2)
        sys.stdout.write("\n")
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"image-eval: error: {error}", file=sys.stderr)
        return 1

    return 0


def _norm_argument(value: str) -> str | float:
    if value == "mean":
        return value

    try:
        cutoff = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--norm must be 'mean' or a numeric cutoff fraction"
        ) from error

    if not math.isfinite(cutoff) or cutoff < 0 or cutoff >= 0.5:
        raise argparse.ArgumentTypeError(
            "--norm numeric cutoff must be >= 0 and < 0.5"
        )
    return cutoff


if __name__ == "__main__":
    raise SystemExit(main())
