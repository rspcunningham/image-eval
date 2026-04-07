from __future__ import annotations

from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from mtf_calc.pipeline import PipelineState, StageMode
from mtf_calc.workflow.find_anchors import find_anchor
from mtf_calc.workspace import (
    InvalidStagePayloadError,
    NoSourceLoadedError,
    ReusableConfigNotFoundError,
    StageAdvanceError,
    StageResultNotFoundError,
    WorkspaceStore,
)


class WorkflowEnvelope(BaseModel):
    type: str
    stage: str | None = None
    payload: dict[str, Any] | None = None


class SessionReadyPayload(BaseModel):
    pipeline: PipelineState
    source_summary: dict[str, Any] | None
    current_stage_result: dict[str, Any] | None


class StageChangedPayload(BaseModel):
    pipeline: PipelineState
    stage_mode: StageMode | None
    stage_result: dict[str, Any] | None


class StageStatusPayload(BaseModel):
    status: str
    detail: str | None = None


class StageProgressPayload(BaseModel):
    label: str
    fraction: float | None = None


class ConfigAppliedPayload(BaseModel):
    pipeline: PipelineState
    anchor: dict[str, Any]
    scale: dict[str, Any]
    bar_rois: dict[str, Any]
    norm_rois: dict[str, Any]
    stage_6: dict[str, Any]
    translation: dict[str, float]


class DocumentResetPayload(BaseModel):
    ok: bool


class WorkflowSession:
    def __init__(self, workspace: WorkspaceStore) -> None:
        self._workspace = workspace

    async def handle_socket(self, websocket: WebSocket) -> None:
        await websocket.accept()

        try:
            while True:
                message = await websocket.receive_json()
                envelope = WorkflowEnvelope.model_validate(message)
                await self._dispatch(websocket, envelope)
        except WebSocketDisconnect:
            return

    async def _dispatch(self, websocket: WebSocket, envelope: WorkflowEnvelope) -> None:
        if envelope.type == "bootstrap":
            await self._send_session_ready(websocket)
            return

        if envelope.type == "advance_stage":
            await self._advance_stage(websocket)
            return

        if envelope.type == "retreat_stage":
            await self._retreat_stage(websocket)
            return

        if envelope.type == "run_anchor":
            await self._run_anchor(websocket)
            return

        if envelope.type == "submit_scale_groups":
            await self._submit_scale_groups(websocket, envelope)
            return

        if envelope.type == "set_bar_roi":
            await self._set_bar_roi(websocket, envelope)
            return

        if envelope.type == "clear_bar_roi":
            await self._clear_bar_roi(websocket, envelope)
            return

        if envelope.type == "set_norm_roi":
            await self._set_norm_roi(websocket, envelope)
            return

        if envelope.type == "clear_norm_roi":
            await self._clear_norm_roi(websocket, envelope)
            return

        if envelope.type == "set_stage_6_crop":
            await self._set_stage_6_crop(websocket, envelope)
            return

        if envelope.type == "run_stage_6_fit":
            await self._run_stage_6_fit(websocket, envelope)
            return

        if envelope.type == "auto_complete_stage":
            await self._send(
                websocket,
                "stage_status",
                stage=envelope.stage or self._workspace.current_stage_name(),
                payload=StageStatusPayload(
                    status="idle",
                    detail="Autocomplete is wired but not implemented yet.",
                ).model_dump(),
            )
            return

        if envelope.type == "apply_reusable_config":
            await self._apply_reusable_config(websocket)
            return

        if envelope.type == "rerun_stage" and envelope.stage == "anchor":
            await self._run_anchor(websocket)
            return

        if envelope.type == "reset_document":
            self._workspace.reset()
            await self._send(websocket, "document_reset", payload=DocumentResetPayload(ok=True).model_dump())
            return

        await self._send_error(websocket, envelope.stage, "Unknown workflow command.")

    async def _send_session_ready(self, websocket: WebSocket) -> None:
        pipeline = self._workspace.pipeline_state()
        source_summary = self._workspace.source_summary()
        stage_name = self._workspace.current_stage_name()
        stage_result = self._optional_stage_result(stage_name) if stage_name else None
        payload = SessionReadyPayload(
            pipeline=pipeline,
            source_summary=source_summary,
            current_stage_result=stage_result,
        )
        await self._send(websocket, "session_ready", stage=stage_name, payload=payload.model_dump(mode="json"))

    async def _advance_stage(self, websocket: WebSocket) -> None:
        try:
            pipeline = self._workspace.advance_stage()
        except (NoSourceLoadedError, StageAdvanceError) as error:
            await self._send_error(websocket, self._workspace.current_stage_name(), str(error))
            return

        stage_name = self._workspace.current_stage_name()
        payload = StageChangedPayload(
            pipeline=pipeline,
            stage_mode=self._stage_mode(stage_name),
            stage_result=self._optional_stage_result(stage_name) if stage_name else None,
        )
        await self._send(websocket, "stage_changed", stage=stage_name, payload=payload.model_dump(mode="json"))

    async def _run_anchor(self, websocket: WebSocket) -> None:
        stage_name = "anchor"
        try:
            data = self._workspace.source_array()
        except NoSourceLoadedError as error:
            await self._send_error(websocket, stage_name, str(error))
            return

        await self._send(
            websocket,
            "stage_status",
            stage=stage_name,
            payload=StageStatusPayload(status="running", detail="Detecting anchor").model_dump(),
        )
        await self._send(
            websocket,
            "stage_progress",
            stage=stage_name,
            payload=StageProgressPayload(label="Finding reference square", fraction=0.35).model_dump(),
        )

        try:
            result = find_anchor(data)
        except Exception as error:
            await self._send_error(websocket, stage_name, str(error))
            await self._send(
                websocket,
                "stage_status",
                stage=stage_name,
                payload=StageStatusPayload(status="error", detail=str(error)).model_dump(),
            )
            return

        self._workspace.set_stage_result(stage_name, result.model_dump())
        await self._send(
            websocket,
            "stage_progress",
            stage=stage_name,
            payload=StageProgressPayload(label="Anchor ready", fraction=1.0).model_dump(),
        )
        await self._send(websocket, "stage_result", stage=stage_name, payload=result.model_dump())
        await self._send(
            websocket,
            "stage_status",
            stage=stage_name,
            payload=StageStatusPayload(status="ready", detail="Detection complete").model_dump(),
        )

    async def _retreat_stage(self, websocket: WebSocket) -> None:
        try:
            pipeline = self._workspace.retreat_stage()
        except (NoSourceLoadedError, StageAdvanceError) as error:
            await self._send_error(websocket, self._workspace.current_stage_name(), str(error))
            return

        stage_name = self._workspace.current_stage_name()
        payload = StageChangedPayload(
            pipeline=pipeline,
            stage_mode=self._stage_mode(stage_name),
            stage_result=self._optional_stage_result(stage_name) if stage_name else None,
        )
        await self._send(websocket, "stage_changed", stage=stage_name, payload=payload.model_dump(mode="json"))

    async def _submit_scale_groups(self, websocket: WebSocket, envelope: WorkflowEnvelope) -> None:
        try:
            payload = self._workspace.set_scale_groups(list((envelope.payload or {}).get("groups", [])))
        except (InvalidStagePayloadError, NoSourceLoadedError) as error:
            await self._send_error(websocket, "scale", str(error))
            await self._send(
                websocket,
                "stage_status",
                stage="scale",
                payload=StageStatusPayload(status="error", detail=str(error)).model_dump(),
            )
            return

        await self._send(websocket, "stage_result", stage="scale", payload=payload)
        await self._send(
            websocket,
            "stage_status",
            stage="scale",
            payload=StageStatusPayload(status="ready", detail="Scale groups saved").model_dump(),
        )

    async def _set_bar_roi(self, websocket: WebSocket, envelope: WorkflowEnvelope) -> None:
        try:
            rect_payload = (envelope.payload or {}).get("rect", {})
            rect = (
                int(rect_payload["row"]),
                int(rect_payload["col"]),
                int(rect_payload["height"]),
                int(rect_payload["width"]),
            )
            payload = self._workspace.set_bar_roi(str((envelope.payload or {}).get("key", "")), rect)
        except (KeyError, TypeError, ValueError, InvalidStagePayloadError, NoSourceLoadedError) as error:
            await self._send_error(websocket, "bar_rois", str(error))
            await self._send(
                websocket,
                "stage_status",
                stage="bar_rois",
                payload=StageStatusPayload(status="error", detail=str(error)).model_dump(),
            )
            return

        await self._send(websocket, "stage_result", stage="bar_rois", payload=payload)
        await self._send(
            websocket,
            "stage_status",
            stage="bar_rois",
            payload=StageStatusPayload(status="ready", detail="ROI saved").model_dump(),
        )

    async def _clear_bar_roi(self, websocket: WebSocket, envelope: WorkflowEnvelope) -> None:
        try:
            payload = self._workspace.clear_bar_roi(str((envelope.payload or {}).get("key", "")))
        except (InvalidStagePayloadError, NoSourceLoadedError) as error:
            await self._send_error(websocket, "bar_rois", str(error))
            await self._send(
                websocket,
                "stage_status",
                stage="bar_rois",
                payload=StageStatusPayload(status="error", detail=str(error)).model_dump(),
            )
            return

        await self._send(websocket, "stage_result", stage="bar_rois", payload=payload)
        await self._send(
            websocket,
            "stage_status",
            stage="bar_rois",
            payload=StageStatusPayload(status="ready", detail="ROI cleared").model_dump(),
        )

    async def _set_norm_roi(self, websocket: WebSocket, envelope: WorkflowEnvelope) -> None:
        try:
            rect_payload = (envelope.payload or {}).get("rect", {})
            rect = (
                int(rect_payload["row"]),
                int(rect_payload["col"]),
                int(rect_payload["height"]),
                int(rect_payload["width"]),
            )
            payload = self._workspace.set_norm_roi(str((envelope.payload or {}).get("key", "")), rect)
        except (KeyError, TypeError, ValueError, InvalidStagePayloadError, NoSourceLoadedError) as error:
            await self._send_error(websocket, "norm_rois", str(error))
            await self._send(
                websocket,
                "stage_status",
                stage="norm_rois",
                payload=StageStatusPayload(status="error", detail=str(error)).model_dump(),
            )
            return

        await self._send(websocket, "stage_result", stage="norm_rois", payload=payload)
        await self._send(
            websocket,
            "stage_status",
            stage="norm_rois",
            payload=StageStatusPayload(status="ready", detail="Normalization ROI saved").model_dump(),
        )

    async def _apply_reusable_config(self, websocket: WebSocket) -> None:
        try:
            result = self._workspace.apply_reusable_config()
        except (NoSourceLoadedError, StageResultNotFoundError, ReusableConfigNotFoundError, InvalidStagePayloadError) as error:
            await self._send_error(websocket, "anchor", str(error))
            await self._send(
                websocket,
                "stage_status",
                stage="anchor",
                payload=StageStatusPayload(status="error", detail=str(error)).model_dump(),
            )
            return

        payload = ConfigAppliedPayload(
            pipeline=PipelineState.model_validate(result["pipeline"]),
            anchor=result["anchor"],
            scale=result["scale"],
            bar_rois=result["bar_rois"],
            norm_rois=result["norm_rois"],
            stage_6=result["stage_6"],
            translation=result["translation"],
        )
        await self._send(websocket, "config_applied", stage="stage_6", payload=payload.model_dump(mode="json"))
        await self._send(
            websocket,
            "stage_status",
            stage="stage_6",
            payload=StageStatusPayload(
                status="ready",
                detail=(
                    "Existing config applied"
                    f" (dx {payload.translation['x']:.1f}, dy {payload.translation['y']:.1f})"
                ),
            ).model_dump(),
        )

    async def _set_stage_6_crop(self, websocket: WebSocket, envelope: WorkflowEnvelope) -> None:
        try:
            payload = self._workspace.set_stage_6_crop(
                str((envelope.payload or {}).get("key", "")),
                left=int((envelope.payload or {}).get("left", 0)),
                right=int((envelope.payload or {}).get("right", 0)),
            )
        except (TypeError, ValueError, InvalidStagePayloadError, NoSourceLoadedError) as error:
            await self._send_error(websocket, "stage_6", str(error))
            await self._send(
                websocket,
                "stage_status",
                stage="stage_6",
                payload=StageStatusPayload(status="error", detail=str(error)).model_dump(),
            )
            return

        await self._send(websocket, "stage_result", stage="stage_6", payload=payload)
        await self._send(
            websocket,
            "stage_status",
            stage="stage_6",
            payload=StageStatusPayload(status="ready", detail="Fit window updated").model_dump(),
        )

    async def _run_stage_6_fit(self, websocket: WebSocket, envelope: WorkflowEnvelope) -> None:
        try:
            payload = self._workspace.run_stage_6_fit(
                str((envelope.payload or {}).get("key", "")),
                harmonic_count=int((envelope.payload or {}).get("harmonicCount", 5)),
            )
        except (TypeError, ValueError, InvalidStagePayloadError, NoSourceLoadedError) as error:
            await self._send_error(websocket, "stage_6", str(error))
            await self._send(
                websocket,
                "stage_status",
                stage="stage_6",
                payload=StageStatusPayload(status="error", detail=str(error)).model_dump(),
            )
            return

        await self._send(websocket, "stage_result", stage="stage_6", payload=payload)
        await self._send(
            websocket,
            "stage_status",
            stage="stage_6",
            payload=StageStatusPayload(status="ready", detail="Fit updated").model_dump(),
        )

    async def _clear_norm_roi(self, websocket: WebSocket, envelope: WorkflowEnvelope) -> None:
        try:
            payload = self._workspace.clear_norm_roi(str((envelope.payload or {}).get("key", "")))
        except (InvalidStagePayloadError, NoSourceLoadedError) as error:
            await self._send_error(websocket, "norm_rois", str(error))
            await self._send(
                websocket,
                "stage_status",
                stage="norm_rois",
                payload=StageStatusPayload(status="error", detail=str(error)).model_dump(),
            )
            return

        await self._send(websocket, "stage_result", stage="norm_rois", payload=payload)
        await self._send(
            websocket,
            "stage_status",
            stage="norm_rois",
            payload=StageStatusPayload(status="ready", detail="Normalization ROI cleared").model_dump(),
        )

    def _optional_stage_result(self, stage_name: str | None) -> dict[str, Any] | None:
        if stage_name is None:
            return None
        return self._workspace.ensure_stage_result(stage_name)

    def _stage_mode(self, stage_name: str | None) -> StageMode | None:
        if stage_name is None:
            return None
        pipeline = self._workspace.pipeline_state()
        for stage in pipeline.stages:
            if stage.name == stage_name:
                return stage.mode
        return None

    async def _send(self, websocket: WebSocket, event_type: str, *, stage: str | None = None, payload: dict[str, Any] | None = None) -> None:
        await websocket.send_json({"type": event_type, "stage": stage, "payload": payload})

    async def _send_error(self, websocket: WebSocket, stage: str | None, detail: str) -> None:
        await self._send(websocket, "stage_error", stage=stage, payload={"detail": detail})
