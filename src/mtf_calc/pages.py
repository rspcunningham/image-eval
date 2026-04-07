from __future__ import annotations

import json
from html import escape
from typing import Any

from mtf_calc.pipeline import PipelineState, StageInfo


def _stage_header(stages: list[StageInfo], current: int | None) -> str:
    items = []
    for stage in stages:
        cls = f"stage-item stage-{stage.status.value}"
        if current is not None and stage.index == current:
            cls += " stage-current"
        items.append(
            f'<div class="{cls}" data-index="{stage.index}">'
            f'<span class="stage-num">{stage.index}</span>'
            f'<span class="stage-title">{escape(stage.title)}</span>'
            "</div>"
        )
    return f'<div id="stage-header">{"".join(items)}</div>'


def _base_page(
    *,
    pipeline: PipelineState,
    page: str,
    body_html: str,
    title: str = "MTF Calculator",
    config: dict[str, Any] | None = None,
) -> str:
    pipeline_json = json.dumps(pipeline.model_dump(), separators=(",", ":"))
    config_block = ""
    if config is not None:
        config_block = (
            '<script type="application/json" id="source-config">'
            f"{json.dumps(config, separators=(',', ':'))}"
            "</script>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{escape(title)}</title>
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body data-page="{escape(page)}">
  {_stage_header(pipeline.stages, pipeline.current)}
  {body_html}
  <script type="application/json" id="pipeline-state">{pipeline_json}</script>
  {config_block}
  <script type="module" src="/static/app.js"></script>
</body>
</html>
"""


def render_load_page(pipeline: PipelineState) -> str:
    body = """
<div id="landing">
  <div class="brand">
    <div class="brand-name">Parasight</div>
    <div class="brand-sub">MTF Calculator</div>
  </div>

  <div id="landing-columns">
    <div class="landing-col">
      <div id="drop-zone">
        <div class="drop-icon"></div>
        <p>Drop <strong>.npy</strong> file or <span class="browse" id="browse-btn">browse</span></p>
      </div>
      <div class="landing-status" id="landing-status"></div>
    </div>
  </div>

  <input type="file" id="file-input" accept=".npy">
</div>
"""
    return _base_page(pipeline=pipeline, page="load", body_html=body)


def render_workflow_page(
    *,
    pipeline: PipelineState,
    source: dict[str, Any],
    anchor_result: dict[str, Any] | None,
    reusable_config: dict[str, Any] | None,
) -> str:
    stage_name = pipeline.stages[pipeline.current].name if pipeline.current is not None else "view"
    stage_ui = _stage_ui(stage_name, anchor_result)
    return _render_source_page(
        pipeline=pipeline,
        source=source,
        page="workflow",
        stage_name=stage_name,
        stage_heading=stage_ui["heading"],
        stage_prompt=stage_ui["prompt"],
        stage_detail=stage_ui["detail"],
        show_rerun=stage_ui["show_rerun"],
        next_disabled=stage_ui["next_disabled"],
        anchor_result=anchor_result,
        reusable_config=reusable_config,
    )


def render_complete_page(pipeline: PipelineState) -> str:
    body = """
<div class="stage-shell">
  <div class="stage-panel">
    <div class="stage-heading">Pipeline Complete</div>
    <div class="stage-body">
      <p class="stage-placeholder">All current workflow stages are complete.</p>
    </div>
    <div class="stage-actions">
      <form method="post" action="/actions/new" class="tb-form">
        <button class="tb-btn" type="submit">New</button>
      </form>
    </div>
  </div>
</div>
"""
    return _base_page(pipeline=pipeline, page="complete", body_html=body)


def _render_source_page(
    *,
    pipeline: PipelineState,
    source: dict[str, Any],
    page: str,
    stage_name: str,
    stage_heading: str,
    stage_prompt: str,
    stage_detail: str,
    show_rerun: bool,
    next_disabled: bool,
    anchor_result: dict[str, Any] | None = None,
    reusable_config: dict[str, Any] | None = None,
) -> str:
    config = {
        "page": page,
        "websocketPath": "/ws/workflow",
        "fileName": source["fileName"],
        "sourceLabel": source["sourceLabel"],
        "byteLength": source["byteLength"],
        "rows": source["rows"],
        "cols": source["cols"],
        "dtype": source["dtype"],
        "dataMin": source["dataMin"],
        "dataMax": source["dataMax"],
        "histogram": source["histogram"],
        "stageName": stage_name,
        "anchorResult": anchor_result,
        "reusableConfig": reusable_config or {"count": 0},
    }

    next_disabled_attr = " disabled" if next_disabled else ""
    rerun_hidden_attr = "" if show_rerun else ' style="display:none;"'
    reusable_count = int((reusable_config or {}).get("count", 0))
    reusable_label = (
        "1 existing config available"
        if reusable_count == 1
        else f"{reusable_count} existing configs available"
    )
    reusable_hidden_attr = "" if reusable_count > 0 and stage_name == "anchor" else ' style="display:none;"'
    body = f"""
<div id="viewer">
  <div id="toolbar">
    <span id="file-name">{escape(source["fileName"])}</span>
    <span id="array-info">{source["rows"]}x{source["cols"]}</span>
    <span class="tb-sep"></span>
    <span class="tb-value">{escape(stage_heading)}</span>

    <div class="tb-spacer"></div>

    <div class="tb-group">
      <button class="tb-btn" id="btn-prev" type="button" title="Previous stage">Prev</button>
      <button class="tb-btn" id="btn-fit" type="button" title="Fit to view">Fit</button>
      <button class="tb-btn" id="btn-1x" type="button" title="1:1 pixels">1:1</button>
      <button class="tb-btn" id="btn-toggle-histogram" type="button" title="Show or hide the window/level histogram">Window</button>
      <button class="tb-btn" id="btn-open" type="button" title="Open new file">Open</button>
      <button class="tb-btn" id="btn-autocomplete" type="button" title="Ask the AI agent to complete this stage">Autocomplete</button>
      <button class="tb-btn" id="btn-rerun-anchor" type="button"{rerun_hidden_attr}>Re-run</button>
      <button class="tb-btn" id="btn-new" type="button" title="Reset the current document">New</button>
      <button class="tb-btn tb-btn-primary" id="btn-next" type="button"{next_disabled_attr}>Next</button>
    </div>
  </div>

  <div id="histogram-panel" hidden>
    <div id="histogram-container">
      <canvas id="histogram-canvas"></canvas>
      <button id="hist-auto-btn" type="button">Auto</button>
      <button id="hist-reset-btn" type="button">Reset</button>
      <span class="hist-label" id="hist-label-min"></span>
      <span class="hist-label" id="hist-label-max"></span>
      <div id="hist-wl-info"></div>
    </div>
  </div>

  <div id="stage-bar">
    <div class="stage-bar-copy">
      <span id="stage-prompt">{escape(stage_prompt)}</span>
      <span id="stage-detail">{escape(stage_detail)}</span>
    </div>
    <button class="tb-btn" id="btn-apply-config" type="button"{reusable_hidden_attr}>{escape(reusable_label)}</button>
    <div class="stage-bar-status" id="stage-run-state"></div>
  </div>

  <div id="workspace-center">
    <aside id="stage-tools" hidden>
      <div class="stage-tools-card" id="scale-tool-panel" hidden>
        <div class="stage-tools-header">
          <span class="stage-tools-title">Scale Groups</span>
          <span class="stage-tools-detail">Select any group numbers from -2 to 7, then continue.</span>
        </div>
        <div class="scale-chip-grid" id="scale-chip-grid"></div>
        <div class="stage-tools-actions">
          <div class="stage-tools-selection" id="scale-selection-summary">No groups selected</div>
          <button class="tb-btn tb-btn-primary" id="btn-sidebar-next" type="button">Next</button>
        </div>
      </div>
      <div class="stage-tools-card" id="bar-roi-panel" hidden>
        <div class="stage-tools-header">
          <span class="stage-tools-title">Bar ROIs</span>
          <span class="stage-tools-detail">Draw the highlighted ROI on the image. The active slot advances through the ordered list.</span>
        </div>
        <div class="stage-tools-progress" id="bar-roi-progress">0 / 0 complete</div>
        <div class="bar-roi-list" id="bar-roi-list"></div>
        <div class="stage-tools-actions">
          <div class="stage-tools-selection" id="bar-roi-selection-summary">Select an ROI slot on the left, then draw a rectangle.</div>
          <button class="tb-btn" id="btn-clear-roi" type="button">Clear Active ROI</button>
          <button class="tb-btn tb-btn-primary" id="btn-bar-roi-next" type="button">Next</button>
        </div>
      </div>
      <div class="stage-tools-card" id="norm-roi-panel" hidden>
        <div class="stage-tools-header">
          <span class="stage-tools-title">Normalization ROIs</span>
          <span class="stage-tools-detail">Select one black ROI for the image, then place one white ROI with the same dimensions.</span>
        </div>
        <div class="stage-tools-progress" id="norm-roi-progress">0 / 0 complete</div>
        <div class="bar-roi-list" id="norm-roi-list"></div>
        <div class="stage-tools-actions">
          <div class="stage-tools-selection" id="norm-roi-selection-summary">Select a normalization ROI slot, then draw a rectangle.</div>
          <button class="tb-btn" id="btn-clear-norm-roi" type="button">Clear Active ROI</button>
          <button class="tb-btn tb-btn-primary" id="btn-norm-roi-next" type="button">Next</button>
        </div>
      </div>
    </aside>

    <div id="canvas-container">
      <canvas id="main-canvas"></canvas>
      <canvas id="overlay-canvas"></canvas>
      <div id="inspector" style="display:none;"></div>
      <div id="zoom-indicator">1.00x</div>
    </div>

    <div id="frame-panel" hidden>
      <div class="frame-panel-card">
        <div class="frame-panel-step" id="frame-panel-step">{escape(stage_heading)}</div>
        <div class="frame-panel-title" id="frame-panel-title">{escape(stage_prompt)}</div>
        <div class="frame-panel-copy" id="frame-panel-copy">{escape(stage_detail)}</div>
        <div class="stage6-panel" id="stage6-panel" hidden>
          <div class="stage6-meta-grid">
            <div class="stage6-meta-card">
              <span class="stage6-meta-label">Black ROI Mean</span>
              <span class="stage6-meta-value" id="stage6-black-mean">-</span>
            </div>
            <div class="stage6-meta-card">
              <span class="stage6-meta-label">White ROI Mean</span>
              <span class="stage6-meta-value" id="stage6-white-mean">-</span>
            </div>
            <div class="stage6-meta-card">
              <span class="stage6-meta-label">Contrast</span>
              <span class="stage6-meta-value" id="stage6-contrast">-</span>
            </div>
          </div>
          <div class="stage6-layout">
            <div class="stage6-profile-list" id="stage6-profile-list"></div>
            <div class="stage6-profile-card">
              <div class="stage6-profile-head">
                <div class="stage6-profile-head-row">
                  <div class="stage6-profile-title" id="stage6-profile-title">Select a profile</div>
                  <div class="stage6-mode-switch" id="stage6-mode-switch">
                    <button class="tb-btn" id="btn-stage6-raw" type="button">Raw</button>
                    <button class="tb-btn" id="btn-stage6-normalized" type="button">Normalized</button>
                  </div>
                </div>
                <div class="stage6-profile-detail" id="stage6-profile-detail">Stage 6 will display the averaged line profile for each bar ROI.</div>
                <div class="stage6-fit-controls">
                  <button class="tb-btn" id="btn-stage6-save-window" type="button">Save Window</button>
                  <label class="stage6-fit-label" for="stage6-harmonic-count">Terms</label>
                  <input class="stage6-fit-input" id="stage6-harmonic-count" type="number" min="1" max="15" step="1" value="5">
                  <button class="tb-btn tb-btn-primary" id="btn-stage6-run-fit" type="button">Run Fit</button>
                </div>
              </div>
              <svg class="stage6-plot" id="stage6-profile-plot" viewBox="0 0 640 280" hidden>
                <g class="stage6-plot-grid" id="stage6-profile-grid"></g>
                <g class="stage6-plot-axis">
                  <line class="stage6-plot-axis-line" x1="68" y1="18" x2="68" y2="238"></line>
                  <line class="stage6-plot-axis-line" x1="68" y1="238" x2="622" y2="238"></line>
                </g>
                <g class="stage6-plot-crop" id="stage6-profile-crop">
                  <rect class="stage6-plot-mask" id="stage6-crop-mask-left" x="68" y="18" width="0" height="220"></rect>
                  <rect class="stage6-plot-mask" id="stage6-crop-mask-right" x="622" y="18" width="0" height="220"></rect>
                  <line class="stage6-plot-crop-line" id="stage6-crop-left-line" x1="68" y1="18" x2="68" y2="238"></line>
                  <line class="stage6-plot-crop-line" id="stage6-crop-right-line" x1="622" y1="18" x2="622" y2="238"></line>
                  <circle class="stage6-plot-crop-handle" id="stage6-crop-left-handle" cx="68" cy="128" r="6"></circle>
                  <circle class="stage6-plot-crop-handle" id="stage6-crop-right-handle" cx="622" cy="128" r="6"></circle>
                </g>
                <g class="stage6-plot-yticks" id="stage6-profile-yticks"></g>
                <text class="stage6-plot-ytitle" x="18" y="128" transform="rotate(-90 18 128)">Pixel Value</text>
                <polyline class="stage6-plot-line" id="stage6-profile-polyline" points=""></polyline>
                <polyline class="stage6-plot-fit-line" id="stage6-profile-fit-polyline" points=""></polyline>
                <g class="stage6-plot-hover" id="stage6-profile-hover" hidden>
                  <line class="stage6-plot-crosshair" id="stage6-profile-vline" x1="0" y1="18" x2="0" y2="238"></line>
                  <line class="stage6-plot-crosshair" id="stage6-profile-hline" x1="68" y1="0" x2="622" y2="0"></line>
                  <circle class="stage6-plot-marker" id="stage6-profile-marker" cx="0" cy="0" r="4"></circle>
                </g>
                <rect class="stage6-plot-hitbox" id="stage6-profile-hitbox" x="68" y="18" width="554" height="220"></rect>
              </svg>
              <div class="stage6-hover-readout" id="stage6-hover-readout" hidden>
                <span class="stage6-hover-chip" id="stage6-hover-sample">Sample -</span>
                <span class="stage6-hover-chip" id="stage6-hover-value">Value -</span>
              </div>
              <pre class="stage6-fit-coeffs" id="stage6-fit-coeffs">No fit yet.</pre>
              <div class="stage6-profile-empty" id="stage6-profile-empty">No profile available.</div>
            </div>
          </div>
        </div>
        <div class="stage7-panel" id="stage7-panel" hidden>
          <div class="stage7-summary" id="stage7-summary">Stage 7 will aggregate saved Stage 6 fits into one MTF curve per axis.</div>
          <div class="stage7-legend">
            <div class="stage7-legend-chip"><span class="stage7-legend-swatch stage7-legend-swatch-x"></span><span id="stage7-x-count">X: 0 points</span></div>
            <div class="stage7-legend-chip"><span class="stage7-legend-swatch stage7-legend-swatch-y"></span><span id="stage7-y-count">Y: 0 points</span></div>
          </div>
          <svg class="stage7-plot" id="stage7-plot" viewBox="0 0 720 340" hidden>
            <g class="stage7-plot-grid" id="stage7-grid"></g>
            <g class="stage7-plot-axis">
              <line class="stage7-plot-axis-line" x1="72" y1="22" x2="72" y2="292"></line>
              <line class="stage7-plot-axis-line" x1="72" y1="292" x2="696" y2="292"></line>
            </g>
            <g class="stage7-plot-yticks" id="stage7-yticks"></g>
            <g class="stage7-plot-xticks" id="stage7-xticks"></g>
            <text class="stage7-plot-ytitle" x="20" y="156" transform="rotate(-90 20 156)">MTF</text>
            <text class="stage7-plot-xtitle" x="384" y="328">Spatial Frequency (lp/mm)</text>
            <polyline class="stage7-plot-line stage7-plot-line-x" id="stage7-line-x" points=""></polyline>
            <polyline class="stage7-plot-line stage7-plot-line-y" id="stage7-line-y" points=""></polyline>
            <g id="stage7-points-x"></g>
            <g id="stage7-points-y"></g>
          </svg>
          <div class="stage7-empty" id="stage7-empty">Run Stage 6 fits to populate the aggregate MTF curves.</div>
        </div>
      </div>
    </div>
  </div>

  <input type="file" id="file-input" accept=".npy">
</div>
"""
    return _base_page(pipeline=pipeline, page=page, body_html=body, config=config)


def _stage_ui(stage_name: str, anchor_result: dict[str, Any] | None) -> dict[str, Any]:
    generic_detail = "This stage will use the same source canvas as its substrate. Tooling for it has not been implemented yet."
    stage_map: dict[str, dict[str, Any]] = {
        "view": {
            "heading": "Step 0: Inspect Source Image",
            "prompt": "Review the loaded source image before starting measurement.",
            "detail": "Use pan, zoom, and window/level controls to inspect the chart.",
            "show_rerun": False,
            "next_disabled": False,
        },
        "anchor": {
            "heading": "Step 1: Anchor Detection",
            "prompt": "Confirm the detected reference square before continuing.",
            "detail": (
                "Detecting the reference square automatically."
                if anchor_result is None
                else (
                    f"Center ({anchor_result['center_x']:.1f}, {anchor_result['center_y']:.1f})"
                    f" · edge {anchor_result['edge_length_px']:.1f}px"
                    f" · angle {anchor_result['angle_deg']:.2f}°"
                )
            ),
            "show_rerun": True,
            "next_disabled": anchor_result is None,
        },
        "scale": {
            "heading": "Step 2: Scale Identification",
            "prompt": "Identify the scale group numbers on the source chart.",
            "detail": "Deliverable: any selected integers between -2 and 7.",
            "show_rerun": False,
            "next_disabled": False,
        },
        "bar_rois": {
            "heading": "Step 3: Bar ROI Selection",
            "prompt": "For each selected scale group, draw the ordered bar ROIs on the source image.",
            "detail": "Order: for each group, 1X, 1Y, 2X, 2Y, 3X, 3Y, 4X, 4Y, 5X, 5Y, 6X, 6Y.",
            "show_rerun": False,
            "next_disabled": False,
        },
        "norm_rois": {
            "heading": "Step 4: Normalization ROIs",
            "prompt": "Draw one black ROI and one white ROI on the source image.",
            "detail": "The white ROI must match the black ROI dimensions.",
            "show_rerun": False,
            "next_disabled": True,
        },
        "stage_6": {
            "heading": "Step 5: Line Profiles",
            "prompt": "Inspect the averaged line profile for each bar ROI.",
            "detail": "X ROIs are averaged across Y. Y ROIs are averaged across X. Normalization ROIs are reduced to scalar means.",
            "show_rerun": False,
            "next_disabled": False,
        },
        "stage_7": {
            "heading": "Step 6: MTF Curves",
            "prompt": "Compare the saved MTF response for X and Y ROIs on one chart.",
            "detail": "Each point comes from the saved first-harmonic MTF of one Stage 6 fit, grouped by ROI axis.",
            "show_rerun": False,
            "next_disabled": False,
        },
    }
    return stage_map.get(stage_name, stage_map["view"])
