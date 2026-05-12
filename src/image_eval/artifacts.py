from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from image_eval.evaluation import EvaluationResult, evaluation_result_to_dict


class EvaluationArtifactPaths(NamedTuple):
    output_dir: Path
    report_json_path: Path | None


def write_evaluation_artifacts(
    result: EvaluationResult,
    output_dir: Path,
    *,
    write_json: bool = True,
) -> EvaluationArtifactPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = output_dir / "report.json" if write_json else None

    if report_json_path is not None:
        with report_json_path.open("w") as file:
            json.dump(evaluation_result_to_dict(result), file, indent=2)
            file.write("\n")

    return EvaluationArtifactPaths(
        output_dir=output_dir,
        report_json_path=report_json_path,
    )
