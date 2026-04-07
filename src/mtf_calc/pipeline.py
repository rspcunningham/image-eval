from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel


class StageStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    SKIPPED = "skipped"


class StageMode(str, Enum):
    CANVAS = "canvas"
    FRAME = "frame"


@dataclass(frozen=True, slots=True)
class StageDefinition:
    name: str
    title: str
    mode: StageMode


STAGES: tuple[StageDefinition, ...] = (
    StageDefinition("view", "Image View", StageMode.CANVAS),
    StageDefinition("anchor", "Anchor Detection", StageMode.CANVAS),
    StageDefinition("scale", "Scale Identification", StageMode.CANVAS),
    StageDefinition("bar_rois", "Bar ROI Selection", StageMode.CANVAS),
    StageDefinition("norm_rois", "Normalization ROIs", StageMode.CANVAS),
    StageDefinition("stage_6", "Stage 6", StageMode.FRAME),
    StageDefinition("stage_7", "Stage 7", StageMode.FRAME),
)


class StageInfo(BaseModel):
    index: int
    name: str
    title: str
    status: StageStatus
    mode: StageMode


class PipelineState(BaseModel):
    stages: list[StageInfo]
    current: int | None
    has_source: bool
    is_complete: bool


def stage_names() -> list[str]:
    return [stage.name for stage in STAGES]


def stage_index(name: str) -> int:
    for index, stage in enumerate(STAGES):
        if stage.name == name:
            return index
    raise KeyError(name)


def stage_title(name: str) -> str:
    return STAGES[stage_index(name)].title


def stage_mode(name: str) -> StageMode:
    return STAGES[stage_index(name)].mode


def initial_stage_statuses(*, has_source: bool) -> dict[str, StageStatus]:
    statuses = {stage.name: StageStatus.PENDING for stage in STAGES}
    if has_source:
        statuses[STAGES[0].name] = StageStatus.ACTIVE
    return statuses


def current_stage_name(statuses: dict[str, StageStatus], *, has_source: bool) -> str | None:
    for stage in STAGES:
        if statuses.get(stage.name) == StageStatus.ACTIVE:
            return stage.name
    if not has_source:
        return STAGES[0].name
    return None


def build_pipeline_state(statuses: dict[str, StageStatus], *, has_source: bool) -> PipelineState:
    current_name = current_stage_name(statuses, has_source=has_source)
    current = stage_index(current_name) if current_name is not None else None
    stages = [
        StageInfo(
            index=index,
            name=stage.name,
            title=stage.title,
            status=statuses.get(stage.name, StageStatus.PENDING),
            mode=stage.mode,
        )
        for index, stage in enumerate(STAGES)
    ]

    is_complete = has_source and all(stage.status == StageStatus.COMPLETE for stage in stages)
    return PipelineState(
        stages=stages,
        current=current,
        has_source=has_source,
        is_complete=is_complete,
    )
