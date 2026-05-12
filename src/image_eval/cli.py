from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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
    parser.add_argument("--out", type=Path, help="Directory for report.json.")
    parser.add_argument("--json", action="store_true", help="Print the JSON report to stdout.")


def _run_eval(args: argparse.Namespace) -> int:
    from image_eval.artifacts import write_evaluation_artifacts
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
        )

        if args.out is not None:
            write_evaluation_artifacts(result, args.out, write_json=True)

        if args.json or args.out is None:
            json.dump(evaluation_result_to_dict(result), sys.stdout, indent=2)
            sys.stdout.write("\n")
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"image-eval: error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
