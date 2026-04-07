from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from mtf_calc.routes import create_router
from mtf_calc.workspace import WorkspaceStore


PACKAGE_DIR = Path(__file__).resolve().parent
UI_DIR = PACKAGE_DIR / "ui"


def create_app(workspace_root: Path | None = None) -> FastAPI:
    workspace = WorkspaceStore(workspace_root)

    app = FastAPI(title="MTF Calculator", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
    app.include_router(create_router(workspace))
    return app
