from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from mtf_calc.pages import (
    render_complete_page,
    render_load_page,
    render_workflow_page,
)
from mtf_calc.workflow.find_anchors import find_anchor
from mtf_calc.workflow_session import WorkflowSession
from mtf_calc.workspace import (
    InvalidSourceError,
    NoSourceLoadedError,
    StageAdvanceError,
    StageResultNotFoundError,
    WorkspaceStore,
)


def create_router(workspace: WorkspaceStore) -> APIRouter:
    router = APIRouter()
    workflow_session = WorkflowSession(workspace)

    @router.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        pipeline = workspace.pipeline_state()
        if not pipeline.has_source:
            return HTMLResponse(content=render_load_page(pipeline))
        if pipeline.is_complete:
            return HTMLResponse(content=render_complete_page(pipeline))

        source = workspace.source_summary()
        if source is None:
            return HTMLResponse(content=render_load_page(pipeline))

        anchor_result = _load_optional_stage_result(workspace, "anchor")
        return HTMLResponse(
            content=render_workflow_page(
                pipeline=pipeline,
                source=source,
                anchor_result=anchor_result,
                reusable_config=workspace.reusable_config_summary(),
            )
        )

    @router.websocket("/ws/workflow")
    async def workflow_socket(websocket: WebSocket) -> None:
        await workflow_session.handle_socket(websocket)

    @router.post("/actions/load")
    async def load_source(request: Request, name: str = "uploaded.npy") -> Response:
        body = await request.body()
        try:
            workspace.load_source(payload=body, file_name=name)
        except InvalidSourceError as error:
            return Response(content=str(error), status_code=400)
        return RedirectResponse(url="/", status_code=303)

    @router.post("/actions/new")
    def reset_document() -> Response:
        workspace.reset()
        return RedirectResponse(url="/", status_code=303)

    @router.post("/actions/next")
    def advance_stage() -> Response:
        try:
            workspace.advance_stage()
        except (NoSourceLoadedError, StageAdvanceError) as error:
            return Response(content=str(error), status_code=400)
        return RedirectResponse(url="/", status_code=303)

    @router.post("/actions/stage/anchor/run")
    def run_anchor_stage() -> JSONResponse:
        try:
            data = workspace.source_array()
        except NoSourceLoadedError as error:
            return JSONResponse(content={"error": str(error)}, status_code=404)

        try:
            result = find_anchor(data)
        except Exception as error:
            return JSONResponse(content={"error": str(error)}, status_code=400)

        payload = result.model_dump()
        workspace.set_stage_result("anchor", payload)
        return JSONResponse(content=payload)

    @router.get("/api/pipeline")
    def get_pipeline_state() -> JSONResponse:
        return JSONResponse(content=workspace.pipeline_state().model_dump())

    @router.get("/api/source-summary")
    def get_source_summary() -> JSONResponse:
        summary = workspace.source_summary()
        if summary is None:
            return JSONResponse(content={"error": "No source loaded"}, status_code=404)
        return JSONResponse(content=summary)

    @router.get("/api/source")
    def get_source_buffer() -> Response:
        try:
            summary = workspace.source_summary()
            if summary is None:
                raise NoSourceLoadedError("No source loaded")
            content = workspace.source_buffer()
        except NoSourceLoadedError as error:
            return Response(content=str(error), status_code=404)

        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "X-Source-Rows": str(summary["rows"]),
                "X-Source-Cols": str(summary["cols"]),
                "X-Source-Dtype": str(summary["dtype"]),
            },
        )

    @router.get("/api/stage/anchor")
    def get_anchor_result() -> JSONResponse:
        try:
            return JSONResponse(content=workspace.get_stage_result("anchor"))
        except StageResultNotFoundError as error:
            return JSONResponse(content={"error": str(error)}, status_code=404)

    return router


def _load_optional_stage_result(workspace: WorkspaceStore, stage_name: str) -> dict[str, Any] | None:
    try:
        return workspace.get_stage_result(stage_name)
    except StageResultNotFoundError:
        return None
