import {
  applyCanvasTransform,
  fitDisplay,
  oneToOneDisplay,
  percentileFromHistogram,
  renderHistogram,
  renderSourceImage,
  screenToPixel,
  setCanvasTransform,
} from "./render.js";
import { createROIController } from "./roi.js";
import { clamp, fmtNum } from "./utils.js";

const $ = (id) => document.getElementById(id);
const page = document.body.dataset.page;
const STAGE_UI = {
  view: {
    heading: "Step 0: Inspect Source Image",
    prompt: "Review the loaded source image before starting measurement.",
    detail: "Use pan, zoom, and window/level controls to inspect the chart.",
    showRerun: false,
    nextDisabled: false,
  },
  anchor: {
    heading: "Step 1: Anchor Detection",
    prompt: "Confirm the detected reference square before continuing.",
    detail: "Detecting the reference square automatically.",
    showRerun: true,
    nextDisabled: true,
  },
  scale: {
    heading: "Step 2: Scale Identification",
    prompt: "Identify the scale group numbers on the source chart.",
    detail: "Deliverable: any selected integers between -2 and 7.",
    showRerun: false,
    nextDisabled: true,
  },
  bar_rois: {
    heading: "Step 3: Bar ROI Selection",
    prompt: "For each selected scale group, draw the ordered bar ROIs on the source image.",
    detail: "Order: for each group, 1X, 1Y, 2X, 2Y, 3X, 3Y, 4X, 4Y, 5X, 5Y, 6X, 6Y.",
    showRerun: false,
    nextDisabled: true,
  },
  norm_rois: {
    heading: "Step 4: Normalization ROIs",
    prompt: "Draw one black ROI and one white ROI on the source image.",
    detail: "The white ROI must match the black ROI dimensions.",
    showRerun: false,
    nextDisabled: true,
  },
  stage_6: {
    heading: "Step 5: Line Profiles",
    prompt: "Inspect the averaged line profile for each bar ROI.",
    detail: "X ROIs are averaged across Y. Y ROIs are averaged across X. Normalization ROIs are reduced to scalar means.",
    showRerun: false,
    nextDisabled: false,
  },
  stage_7: {
    heading: "Step 6: MTF Curves",
    prompt: "Compare the saved MTF response for X and Y ROIs on one chart.",
    detail: "Each point comes from the saved first-harmonic MTF of one Stage 6 fit, grouped by ROI axis.",
    showRerun: false,
    nextDisabled: false,
  },
};

if (page === "load") {
  initLoadPage();
} else if (page === "workflow") {
  initWorkflowPage();
}

function initLoadPage() {
  const dropZone = $("drop-zone");
  const browseButton = $("browse-btn");
  const fileInput = $("file-input");
  const status = $("landing-status");

  const setStatus = (message, tone = "") => {
    status.textContent = message;
    status.className = tone ? `landing-status ${tone}` : "landing-status";
  };

  const beginUpload = async (file) => {
    setStatus(`Uploading ${file.name}...`);

    try {
      await uploadSourceFile(file);
      window.location.assign("/");
    } catch (error) {
      setStatus(error.message, "error");
    } finally {
      fileInput.value = "";
    }
  };

  browseButton.addEventListener("click", (event) => {
    event.stopPropagation();
    fileInput.click();
  });

  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (event) => {
    if (event.target.files && event.target.files[0]) {
      beginUpload(event.target.files[0]);
    }
  });

  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
    if (event.dataTransfer.files && event.dataTransfer.files[0]) {
      beginUpload(event.dataTransfer.files[0]);
    }
  });

  document.body.addEventListener("dragover", (event) => event.preventDefault());
  document.body.addEventListener("drop", (event) => event.preventDefault());
}

async function initWorkflowPage() {
  const configNode = $("source-config");
  if (!configNode) {
    return;
  }

  const config = JSON.parse(configNode.textContent);
  const fileInput = $("file-input");
  const histogramCanvas = $("histogram-canvas");
  const histogramPanel = $("histogram-panel");
  const histogramContainer = $("histogram-container");
  const histLabelMin = $("hist-label-min");
  const histLabelMax = $("hist-label-max");
  const histWlInfo = $("hist-wl-info");
  const canvasContainer = $("canvas-container");
  const mainCanvas = $("main-canvas");
  const overlayCanvas = $("overlay-canvas");
  const framePanel = $("frame-panel");
  const framePanelStep = $("frame-panel-step");
  const framePanelTitle = $("frame-panel-title");
  const framePanelCopy = $("frame-panel-copy");
  const stage6Panel = $("stage6-panel");
  const stage6BlackMean = $("stage6-black-mean");
  const stage6WhiteMean = $("stage6-white-mean");
  const stage6Contrast = $("stage6-contrast");
  const stage6ProfileList = $("stage6-profile-list");
  const stage6ProfileTitle = $("stage6-profile-title");
  const stage6ProfileDetail = $("stage6-profile-detail");
  const stage6RawButton = $("btn-stage6-raw");
  const stage6NormalizedButton = $("btn-stage6-normalized");
  const stage6SaveWindowButton = $("btn-stage6-save-window");
  const stage6RunFitButton = $("btn-stage6-run-fit");
  const stage6HarmonicCountInput = $("stage6-harmonic-count");
  const stage6ProfilePlot = $("stage6-profile-plot");
  const stage6ProfileGrid = $("stage6-profile-grid");
  const stage6ProfileCrop = $("stage6-profile-crop");
  const stage6CropMaskLeft = $("stage6-crop-mask-left");
  const stage6CropMaskRight = $("stage6-crop-mask-right");
  const stage6CropLeftLine = $("stage6-crop-left-line");
  const stage6CropRightLine = $("stage6-crop-right-line");
  const stage6CropLeftHandle = $("stage6-crop-left-handle");
  const stage6CropRightHandle = $("stage6-crop-right-handle");
  const stage6ProfileYTicks = $("stage6-profile-yticks");
  const stage6ProfilePolyline = $("stage6-profile-polyline");
  const stage6ProfileFitPolyline = $("stage6-profile-fit-polyline");
  const stage6ProfileHover = $("stage6-profile-hover");
  const stage6ProfileVLine = $("stage6-profile-vline");
  const stage6ProfileHLine = $("stage6-profile-hline");
  const stage6ProfileMarker = $("stage6-profile-marker");
  const stage6ProfileHitbox = $("stage6-profile-hitbox");
  const stage6HoverReadout = $("stage6-hover-readout");
  const stage6HoverSample = $("stage6-hover-sample");
  const stage6HoverValue = $("stage6-hover-value");
  const stage6FitCoeffs = $("stage6-fit-coeffs");
  const stage6ProfileEmpty = $("stage6-profile-empty");
  const stage7Panel = $("stage7-panel");
  const stage7Summary = $("stage7-summary");
  const stage7XCount = $("stage7-x-count");
  const stage7YCount = $("stage7-y-count");
  const stage7Plot = $("stage7-plot");
  const stage7Grid = $("stage7-grid");
  const stage7YTicks = $("stage7-yticks");
  const stage7XTicks = $("stage7-xticks");
  const stage7LineX = $("stage7-line-x");
  const stage7LineY = $("stage7-line-y");
  const stage7PointsX = $("stage7-points-x");
  const stage7PointsY = $("stage7-points-y");
  const stage7Empty = $("stage7-empty");
  const mainCtx = mainCanvas.getContext("2d");
  const overlayCtx = overlayCanvas.getContext("2d");
  const inspector = $("inspector");
  const zoomIndicator = $("zoom-indicator");
  const runState = $("stage-run-state");
  const stageDetail = $("stage-detail");
  const stagePrompt = $("stage-prompt");
  const stageTools = $("stage-tools");
  const scaleToolPanel = $("scale-tool-panel");
  const scaleChipGrid = $("scale-chip-grid");
  const scaleSelectionSummary = $("scale-selection-summary");
  const sidebarNextButton = $("btn-sidebar-next");
  const barRoiPanel = $("bar-roi-panel");
  const barRoiProgress = $("bar-roi-progress");
  const barRoiList = $("bar-roi-list");
  const barRoiSelectionSummary = $("bar-roi-selection-summary");
  const clearRoiButton = $("btn-clear-roi");
  const barRoiNextButton = $("btn-bar-roi-next");
  const normRoiPanel = $("norm-roi-panel");
  const normRoiProgress = $("norm-roi-progress");
  const normRoiList = $("norm-roi-list");
  const normRoiSelectionSummary = $("norm-roi-selection-summary");
  const clearNormRoiButton = $("btn-clear-norm-roi");
  const normRoiNextButton = $("btn-norm-roi-next");
  const autocompleteButton = $("btn-autocomplete");
  const applyConfigButton = $("btn-apply-config");
  const histogramToggleButton = $("btn-toggle-histogram");
  const prevButton = $("btn-prev");
  const newButton = $("btn-new");
  const nextButton = $("btn-next");
  const toolbarStageHeading = document.querySelector("#toolbar .tb-value");
  const rerunButton = $("btn-rerun-anchor");
  const stageHeader = $("stage-header");
  const body = document.body;

  const state = {
    socket: null,
    socketReady: false,
    data: null,
    sourceKey: null,
    sourceSummary: null,
    pipeline: null,
    stageName: null,
    stageMode: "canvas",
    stageResults: {},
    reusableConfig: config.reusableConfig || { count: 0 },
    draftScaleGroups: [],
    pendingScaleAdvance: false,
    activeBarRoiKey: null,
    activeNormRoiKey: null,
    activeStage6ProfileKey: null,
    stage6DisplayMode: "raw",
    stage6HarmonicCount: 5,
    stage6DraftCrops: {},
    display: {
      window: 1,
      level: 0.5,
      zoom: 1,
      panX: 0,
      panY: 0,
      histogramOpen: false,
    },
    interaction: {
      histDragMode: null,
      histDragStartX: 0,
      histDragStartLevel: 0,
      histDragStartWindow: 0,
      isPanning: false,
      panStartX: 0,
      panStartY: 0,
      stage6HoverIndex: null,
      stage6HoverActive: false,
      stage6CropDrag: null,
      stage6CropPreview: null,
    },
  };
  let roiController = null;

  const currentStageInfo = () => {
    if (!state.pipeline || state.pipeline.current === null) {
      return null;
    }
    return state.pipeline.stages[state.pipeline.current];
  };

  const currentAnchorResult = () => state.stageResults.anchor || null;
  const currentScaleResult = () => state.stageResults.scale || null;
  const currentBarRoiResult = () => state.stageResults.bar_rois || null;
  const currentNormRoiResult = () => state.stageResults.norm_rois || null;
  const currentStage6Result = () => state.stageResults.stage_6 || null;
  const currentStage6Profile = () => (
    (currentStage6Result()?.profiles || []).find((profile) => profile.key === state.activeStage6ProfileKey) || null
  );
  const currentStage7Result = () => state.stageResults.stage_7 || null;

  const sameCrop = (a, b) => (
    Number(a?.left || 0) === Number(b?.left || 0) && Number(a?.right || 0) === Number(b?.right || 0)
  );

  const stage6EffectiveCrop = (profile) => {
    if (!profile) {
      return { left: 0, right: 0 };
    }
    if (state.interaction.stage6CropPreview?.key === profile.key) {
      return state.interaction.stage6CropPreview;
    }
    return state.stage6DraftCrops[profile.key] || profile.crop || { left: 0, right: 0 };
  };

  const setRunStatus = (message, tone = "") => {
    if (!runState) {
      return;
    }
    runState.textContent = message;
    runState.className = tone ? `stage-bar-status ${tone}` : "stage-bar-status";
  };

  const syncTransforms = () => {
    applyCanvasTransform({ canvas: mainCanvas, display: state.display, zoomIndicator });
    setCanvasTransform(overlayCanvas, state.display);
    if (roiController) {
      roiController.syncTransform();
    }
  };

  const renderAnchorOverlay = () => {
    const summary = state.sourceSummary;
    if (!summary) {
      return;
    }

    overlayCanvas.width = summary.cols;
    overlayCanvas.height = summary.rows;
    overlayCtx.clearRect(0, 0, summary.cols, summary.rows);

    const anchorResult = currentAnchorResult();
    if ((state.stageName === "bar_rois" || state.stageName === "norm_rois") && roiController) {
      roiController.render();
      return;
    }
    if (state.stageName !== "anchor" || !anchorResult) {
      syncTransforms();
      return;
    }

    const { corners, center_x: centerX, center_y: centerY } = anchorResult;
    overlayCtx.strokeStyle = "rgba(255, 159, 149, 0.95)";
    overlayCtx.fillStyle = "rgba(255, 159, 149, 0.12)";
    overlayCtx.lineWidth = 3;
    overlayCtx.beginPath();
    corners.forEach((point, index) => {
      if (index === 0) {
        overlayCtx.moveTo(point.x, point.y);
      } else {
        overlayCtx.lineTo(point.x, point.y);
      }
    });
    overlayCtx.closePath();
    overlayCtx.fill();
    overlayCtx.stroke();

    overlayCtx.strokeStyle = "rgba(61, 216, 197, 0.95)";
    overlayCtx.lineWidth = 2;
    overlayCtx.beginPath();
    overlayCtx.moveTo(centerX - 10, centerY);
    overlayCtx.lineTo(centerX + 10, centerY);
    overlayCtx.moveTo(centerX, centerY - 10);
    overlayCtx.lineTo(centerX, centerY + 10);
    overlayCtx.stroke();

    syncTransforms();
  };

  const renderAll = () => {
    const summary = state.sourceSummary;
    if (!summary || !state.data) {
      return;
    }

    if (state.display.histogramOpen && histogramPanel && histogramContainer && !histogramPanel.hidden) {
      renderHistogram({
        canvas: histogramCanvas,
        container: histogramContainer,
        counts: summary.histogram,
        display: state.display,
        dataMin: summary.dataMin,
        dataMax: summary.dataMax,
        labels: { min: histLabelMin, max: histLabelMax },
        info: histWlInfo,
        fmtNum,
      });
    }

    renderSourceImage({
      canvas: mainCanvas,
      context: mainCtx,
      data: state.data,
      rows: summary.rows,
      cols: summary.cols,
      display: state.display,
    });

    renderAnchorOverlay();
  };

  const fitToView = () => {
    const summary = state.sourceSummary;
    if (!summary) {
      return;
    }
    const next = fitDisplay(canvasContainer, summary.rows, summary.cols);
    state.display.zoom = next.zoom;
    state.display.panX = next.panX;
    state.display.panY = next.panY;
    syncTransforms();
  };

  const setOneToOne = () => {
    const summary = state.sourceSummary;
    if (!summary) {
      return;
    }
    const next = oneToOneDisplay(canvasContainer, summary.rows, summary.cols);
    state.display.zoom = next.zoom;
    state.display.panX = next.panX;
    state.display.panY = next.panY;
    syncTransforms();
  };

  const updateInspector = (clientX, clientY) => {
    const summary = state.sourceSummary;
    if (!summary || !state.data || state.stageMode !== "canvas") {
      inspector.style.display = "none";
      return;
    }

    const point = screenToPixel(canvasContainer, state.display, clientX, clientY);
    const inside = point.x >= 0 && point.x < summary.cols && point.y >= 0 && point.y < summary.rows;
    if (!inside) {
      inspector.style.display = "none";
      return;
    }

    const value = state.data[point.y * summary.cols + point.x];
    inspector.style.display = "block";
    inspector.innerHTML =
      `<span class="coord">[${point.y}, ${point.x}]</span>` +
      `<br><span class="inspector-label">value</span> ` +
      `<span class="inspector-value">${fmtNum(value)}</span>`;
  };

  const resetWindowLevel = () => {
    const summary = state.sourceSummary;
    if (!summary) {
      return;
    }
    state.display.window = Math.max(summary.dataMax - summary.dataMin, 0.001);
    state.display.level = (summary.dataMin + summary.dataMax) / 2;
    renderAll();
  };

  const autoWindowLevel = () => {
    const summary = state.sourceSummary;
    if (!summary) {
      return;
    }
    const low = percentileFromHistogram(summary.histogram, 0.02);
    const high = percentileFromHistogram(summary.histogram, 0.98);
    state.display.window = Math.max(high - low, 0.001);
    state.display.level = (low + high) / 2;
    renderAll();
  };

  const updateHeader = () => {
    if (!state.pipeline || !stageHeader) {
      return;
    }

    for (const item of stageHeader.querySelectorAll(".stage-item")) {
      const stageIndex = Number(item.dataset.index);
      const stage = state.pipeline.stages[stageIndex];
      item.className = `stage-item stage-${stage.status}`;
      if (state.pipeline.current === stageIndex) {
        item.classList.add("stage-current");
      }
    }
  };

  const applyLayoutMode = () => {
    const showCanvas = state.stageMode === "canvas";
    canvasContainer.hidden = !showCanvas;
    framePanel.hidden = showCanvas;
    if (stageTools) {
      stageTools.hidden = !showCanvas;
    }
    if (histogramPanel) {
      histogramPanel.hidden = !showCanvas || !state.display.histogramOpen;
    }
    if (histogramToggleButton) {
      histogramToggleButton.style.display = showCanvas ? "" : "none";
      histogramToggleButton.dataset.open = String(showCanvas && state.display.histogramOpen);
    }
    if (!showCanvas) {
      inspector.style.display = "none";
    }
    body.dataset.stageMode = state.stageMode;
    body.dataset.stageName = state.stageName || "";
  };

  const toggleHistogramPanel = () => {
    state.display.histogramOpen = !state.display.histogramOpen;
    applyLayoutMode();
    if (state.display.histogramOpen && state.stageMode === "canvas") {
      requestAnimationFrame(() => renderAll());
    }
  };

  const updateStageChrome = () => {
    const ui = STAGE_UI[state.stageName] || STAGE_UI.view;
    const stageInfo = currentStageInfo();
    if (toolbarStageHeading) {
      toolbarStageHeading.textContent = ui.heading;
    }
    if (stagePrompt) {
      stagePrompt.textContent = ui.prompt;
    }

    const anchorResult = currentAnchorResult();
    if (state.stageName === "anchor" && anchorResult) {
      stageDetail.textContent =
        `Center (${anchorResult.center_x.toFixed(1)}, ${anchorResult.center_y.toFixed(1)})` +
        ` · edge ${anchorResult.edge_length_px.toFixed(1)}px` +
        ` · angle ${anchorResult.angle_deg.toFixed(2)}°`;
    } else {
      stageDetail.textContent = ui.detail;
    }

    if (framePanelStep) {
      framePanelStep.textContent = ui.heading;
    }
    if (framePanelTitle) {
      framePanelTitle.textContent = ui.prompt;
    }
    if (framePanelCopy) {
      framePanelCopy.textContent = state.stageName === "anchor" && anchorResult ? stageDetail.textContent : ui.detail;
    }

    if (rerunButton) {
      rerunButton.style.display = ui.showRerun ? "" : "none";
    }
    if (applyConfigButton) {
      const count = Number(state.reusableConfig?.count || 0);
      applyConfigButton.style.display = state.stageName === "anchor" && count > 0 ? "" : "none";
      applyConfigButton.textContent = count === 1 ? "1 existing config available" : `${count} existing configs available`;
      applyConfigButton.disabled = !anchorResult || count < 1;
    }
    if (prevButton) {
      const hasPrev = Boolean(stageInfo && stageInfo.index > 0);
      prevButton.style.display = hasPrev ? "" : "none";
      prevButton.disabled = !hasPrev;
    }
    if (nextButton) {
      if (state.stageName === "anchor") {
        nextButton.hidden = false;
        nextButton.disabled = !anchorResult;
      } else if (state.stageName === "scale") {
        nextButton.hidden = true;
        nextButton.disabled = true;
      } else if (state.stageName === "bar_rois") {
        nextButton.hidden = true;
        nextButton.disabled = true;
      } else if (state.stageName === "norm_rois") {
        nextButton.hidden = true;
        nextButton.disabled = true;
      } else {
        nextButton.hidden = false;
        nextButton.disabled = ui.nextDisabled;
      }
    }
  };

  const findNextIncompleteBarRoi = (preferredKey = null) => {
    const slots = currentBarRoiResult()?.slots || [];
    if (preferredKey) {
      const startIndex = Math.max(0, slots.findIndex((slot) => slot.key === preferredKey));
      for (let index = startIndex; index < slots.length; index += 1) {
        if (!slots[index].rect) {
          return slots[index].key;
        }
      }
    }
    const next = slots.find((slot) => !slot.rect);
    return next ? next.key : (slots[0]?.key || null);
  };

  const findNextIncompleteNormRoi = (preferredKey = null) => {
    const slots = currentNormRoiResult()?.slots || [];
    if (preferredKey) {
      const startIndex = Math.max(0, slots.findIndex((slot) => slot.key === preferredKey));
      for (let index = startIndex; index < slots.length; index += 1) {
        if (!slots[index].rect) {
          return slots[index].key;
        }
      }
    }
    const next = slots.find((slot) => !slot.rect);
    return next ? next.key : (slots[0]?.key || null);
  };

  const selectStage6ProfileKey = (preferredKey = null) => {
    const profiles = currentStage6Result()?.profiles || [];
    if (preferredKey && profiles.some((profile) => profile.key === preferredKey)) {
      return preferredKey;
    }
    return profiles[0]?.key || null;
  };

  const renderScaleTools = () => {
    if (!stageTools || !scaleToolPanel || !scaleChipGrid || !scaleSelectionSummary || !sidebarNextButton) {
      return;
    }

    const showScaleTools = state.stageName === "scale";
    const showBarRoiTools = state.stageName === "bar_rois";
    const showNormRoiTools = state.stageName === "norm_rois";
    stageTools.hidden = !(showScaleTools || showBarRoiTools || showNormRoiTools);
    scaleToolPanel.hidden = !showScaleTools;
    if (barRoiPanel) {
      barRoiPanel.hidden = !showBarRoiTools;
    }
    if (normRoiPanel) {
      normRoiPanel.hidden = !showNormRoiTools;
    }
    if (nextButton) {
      nextButton.hidden = showScaleTools || showBarRoiTools || showNormRoiTools;
    }
    if (!showScaleTools) {
      return;
    }

    if (!scaleChipGrid.dataset.initialized) {
      for (let value = -2; value <= 7; value += 1) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "scale-chip";
        button.dataset.value = String(value);
        button.textContent = String(value);
        button.addEventListener("click", () => {
          const numeric = Number(button.dataset.value);
          if (state.draftScaleGroups.includes(numeric)) {
            state.draftScaleGroups = state.draftScaleGroups.filter((entry) => entry !== numeric);
          } else {
            state.draftScaleGroups = [...state.draftScaleGroups, numeric].sort((a, b) => a - b);
          }
          renderScaleTools();
        });
        scaleChipGrid.appendChild(button);
      }
      scaleChipGrid.dataset.initialized = "true";
    }

    for (const chip of scaleChipGrid.querySelectorAll(".scale-chip")) {
      const numeric = Number(chip.dataset.value);
      chip.dataset.selected = String(state.draftScaleGroups.includes(numeric));
    }

    if (state.draftScaleGroups.length === 0) {
      scaleSelectionSummary.textContent = "No groups selected";
    } else {
      scaleSelectionSummary.textContent = `Selected: ${state.draftScaleGroups.join(", ")}`;
    }
    sidebarNextButton.disabled = state.draftScaleGroups.length === 0;
  };

  const renderBarRoiTools = () => {
    if (!stageTools || !barRoiPanel || !barRoiList || !barRoiProgress || !barRoiSelectionSummary || !clearRoiButton) {
      return;
    }

    const showBarTools = state.stageName === "bar_rois";
    barRoiPanel.hidden = !showBarTools;
    if (!showBarTools) {
      return;
    }

    const slots = currentBarRoiResult()?.slots || [];
    if (!state.activeBarRoiKey) {
      state.activeBarRoiKey = findNextIncompleteBarRoi();
    }

    const completeCount = slots.filter((slot) => slot.rect).length;
    barRoiProgress.textContent = `${completeCount} / ${slots.length} complete`;
    if (barRoiNextButton) {
      barRoiNextButton.disabled = slots.length === 0 || completeCount !== slots.length;
    }
    barRoiList.innerHTML = "";

    for (const slot of slots) {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "bar-roi-item";
      item.dataset.active = String(slot.key === state.activeBarRoiKey);
      item.dataset.complete = String(Boolean(slot.rect));
      item.innerHTML =
        `<span class="bar-roi-item-label">${slot.label}</span>` +
        `<span class="bar-roi-item-status">${slot.rect ? "Done" : "Pending"}</span>`;
      item.addEventListener("click", () => {
        state.activeBarRoiKey = slot.key;
        renderBarRoiTools();
        if (roiController) {
          roiController.render();
        }
      });
      barRoiList.appendChild(item);
    }

    const activeSlot = slots.find((slot) => slot.key === state.activeBarRoiKey) || null;
    if (!activeSlot) {
      barRoiSelectionSummary.textContent = "No ROI slots available.";
      clearRoiButton.disabled = true;
    } else if (activeSlot.rect) {
      barRoiSelectionSummary.textContent = `Active: ${activeSlot.label}. Draw again to replace, or clear it.`;
      clearRoiButton.disabled = false;
    } else {
      barRoiSelectionSummary.textContent = `Active: ${activeSlot.label}. Draw a rectangle on the image.`;
      clearRoiButton.disabled = true;
    }
  };

  const renderNormRoiTools = () => {
    if (!stageTools || !normRoiPanel || !normRoiList || !normRoiProgress || !normRoiSelectionSummary || !clearNormRoiButton) {
      return;
    }

    const showNormTools = state.stageName === "norm_rois";
    normRoiPanel.hidden = !showNormTools;
    if (!showNormTools) {
      return;
    }

    const slots = currentNormRoiResult()?.slots || [];
    if (!state.activeNormRoiKey) {
      state.activeNormRoiKey = findNextIncompleteNormRoi();
    }

    const completeCount = slots.filter((slot) => slot.rect).length;
    normRoiProgress.textContent = `${completeCount} / ${slots.length} complete`;
    if (normRoiNextButton) {
      normRoiNextButton.disabled = slots.length === 0 || completeCount !== slots.length;
    }
    normRoiList.innerHTML = "";

    for (const slot of slots) {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "bar-roi-item";
      item.dataset.active = String(slot.key === state.activeNormRoiKey);
      item.dataset.complete = String(Boolean(slot.rect));
      item.innerHTML =
        `<span class="bar-roi-item-label">${slot.label}</span>` +
        `<span class="bar-roi-item-status">${slot.rect ? "Done" : "Pending"}</span>`;
      item.addEventListener("click", () => {
        state.activeNormRoiKey = slot.key;
        renderNormRoiTools();
        if (roiController) {
          roiController.render();
        }
      });
      normRoiList.appendChild(item);
    }

    const activeSlot = slots.find((slot) => slot.key === state.activeNormRoiKey) || null;
    if (!activeSlot) {
      normRoiSelectionSummary.textContent = "No normalization ROI slots available.";
      clearNormRoiButton.disabled = true;
      return;
    }

    const blackSlot = slots.find((slot) => slot.tone === "black") || null;
    if (activeSlot.tone === "white" && (!blackSlot || !blackSlot.rect)) {
      normRoiSelectionSummary.textContent = `Active: ${activeSlot.label}. Select the black ROI first.`;
      clearNormRoiButton.disabled = true;
    } else if (activeSlot.tone === "white" && blackSlot?.rect) {
      normRoiSelectionSummary.textContent =
        `Active: ${activeSlot.label}. Draw a ${blackSlot.rect.width}×${blackSlot.rect.height} ROI to match the black selection.`;
      clearNormRoiButton.disabled = !activeSlot.rect;
    } else if (activeSlot.rect) {
      normRoiSelectionSummary.textContent = `Active: ${activeSlot.label}. Draw again to replace, or clear it.`;
      clearNormRoiButton.disabled = false;
    } else {
      normRoiSelectionSummary.textContent = `Active: ${activeSlot.label}. Draw a rectangle on the image.`;
      clearNormRoiButton.disabled = true;
    }
  };

  const renderStage6Panel = () => {
    if (
      !stage6Panel
      || !stage6BlackMean
      || !stage6WhiteMean
      || !stage6Contrast
      || !stage6ProfileList
      || !stage6ProfileTitle
      || !stage6ProfileDetail
      || !stage6RawButton
      || !stage6NormalizedButton
      || !stage6SaveWindowButton
      || !stage6RunFitButton
      || !stage6HarmonicCountInput
      || !stage6ProfilePlot
      || !stage6ProfileGrid
      || !stage6ProfileCrop
      || !stage6CropMaskLeft
      || !stage6CropMaskRight
      || !stage6CropLeftLine
      || !stage6CropRightLine
      || !stage6CropLeftHandle
      || !stage6CropRightHandle
      || !stage6ProfileYTicks
      || !stage6ProfilePolyline
      || !stage6ProfileFitPolyline
      || !stage6ProfileHover
      || !stage6ProfileVLine
      || !stage6ProfileHLine
      || !stage6ProfileMarker
      || !stage6ProfileHitbox
      || !stage6HoverReadout
      || !stage6HoverSample
      || !stage6HoverValue
      || !stage6FitCoeffs
      || !stage6ProfileEmpty
    ) {
      return;
    }

    const showStage6 = state.stageName === "stage_6";
    stage6Panel.hidden = !showStage6;
    if (!showStage6) {
      return;
    }

    const result = currentStage6Result();
    const normalization = result?.normalization || {};
    const normalizedAvailable = Boolean(normalization.normalized);
    stage6BlackMean.textContent = Number.isFinite(normalization.blackMean) ? fmtNum(normalization.blackMean) : "-";
    stage6WhiteMean.textContent = Number.isFinite(normalization.whiteMean) ? fmtNum(normalization.whiteMean) : "-";
    stage6Contrast.textContent = Number.isFinite(normalization.contrast) ? fmtNum(normalization.contrast) : "-";
    if (state.stage6DisplayMode === "normalized" && !normalizedAvailable) {
      state.stage6DisplayMode = "raw";
    }
    stage6RawButton.dataset.active = String(state.stage6DisplayMode === "raw");
    stage6NormalizedButton.dataset.active = String(state.stage6DisplayMode === "normalized");
    stage6NormalizedButton.disabled = !normalizedAvailable;
    stage6HarmonicCountInput.value = String(state.stage6HarmonicCount);

    const profiles = result?.profiles || [];
    state.activeStage6ProfileKey = selectStage6ProfileKey(state.activeStage6ProfileKey);
    stage6ProfileList.innerHTML = "";

    for (const profile of profiles) {
      const draft = state.stage6DraftCrops[profile.key];
      const dirty = Boolean(draft && !sameCrop(draft, profile.crop || { left: 0, right: 0 }));
      const item = document.createElement("button");
      item.type = "button";
      item.className = "bar-roi-item";
      item.dataset.active = String(profile.key === state.activeStage6ProfileKey);
      item.dataset.complete = String(!dirty);
      item.innerHTML =
        `<span class="bar-roi-item-label">${profile.label}</span>` +
        `<span class="bar-roi-item-status">${dirty ? "Window draft" : `${profile.profileLength} samples`}</span>`;
      item.addEventListener("click", () => {
        state.activeStage6ProfileKey = profile.key;
        state.interaction.stage6HoverIndex = null;
        state.interaction.stage6HoverActive = false;
        state.interaction.stage6CropPreview = null;
        state.interaction.stage6CropDrag = null;
        renderStage6Panel();
      });
      stage6ProfileList.appendChild(item);
    }

    const activeProfile = profiles.find((profile) => profile.key === state.activeStage6ProfileKey) || null;
    const activeValues = activeProfile
      ? (state.stage6DisplayMode === "normalized" ? activeProfile.normalizedProfile : activeProfile.rawProfile)
      : null;
    if (!activeProfile || !Array.isArray(activeValues) || activeValues.length === 0) {
      stage6ProfileTitle.textContent = "Select a profile";
      stage6ProfileDetail.textContent = "Stage 6 will display the averaged line profile for each bar ROI.";
      stage6ProfilePlot.hidden = true;
      stage6HoverReadout.hidden = true;
      stage6ProfileHover.hidden = true;
      stage6ProfileEmpty.hidden = false;
      stage6ProfileEmpty.textContent = profiles.length === 0 ? "No line profiles are available." : "No profile available.";
      stage6ProfileGrid.innerHTML = "";
      stage6ProfileYTicks.innerHTML = "";
      stage6ProfilePolyline.setAttribute("points", "");
      stage6ProfileFitPolyline.setAttribute("points", "");
      stage6FitCoeffs.textContent = "No fit yet.";
      stage6SaveWindowButton.disabled = true;
      stage6RunFitButton.disabled = true;
      stage6ProfileCrop.hidden = true;
      state.interaction.stage6HoverIndex = null;
      return;
    }

    const values = activeValues.map((value) => Number(value));
    const valueMin = Math.min(...values);
    const valueMax = Math.max(...values);
    const span = Math.max(valueMax - valueMin, 1e-6);
    const left = 68;
    const right = 18;
    const top = 18;
    const bottom = 42;
    const width = 640 - left - right;
    const height = 280 - top - bottom;
    const pointXs = values.map((value, index) => (
      left + (values.length === 1 ? 0.5 : index / (values.length - 1)) * width
    ));
    const pointYs = values.map((value) => top + (1 - ((value - valueMin) / span)) * height);
    const points = values.map((value, index) => {
      const x = pointXs[index];
      const y = pointYs[index];
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
    const fitValues = activeProfile.fit
      ? (state.stage6DisplayMode === "normalized" ? activeProfile.fit.normalizedFitProfile : activeProfile.fit.rawFitProfile)
      : null;
    const fitPoints = Array.isArray(fitValues) && fitValues.length === values.length
      ? fitValues.map((value, index) => {
        const x = pointXs[index];
        const y = top + (1 - ((Number(value) - valueMin) / span)) * height;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      }).join(" ")
      : "";
    const tickCount = 5;
    stage6ProfileGrid.innerHTML = "";
    stage6ProfileYTicks.innerHTML = "";
    for (let tickIndex = 0; tickIndex < tickCount; tickIndex += 1) {
      const t = tickCount === 1 ? 0 : tickIndex / (tickCount - 1);
      const y = top + t * height;
      const value = valueMax - t * span;

      const gridLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
      gridLine.setAttribute("class", "stage6-plot-grid-line");
      gridLine.setAttribute("x1", String(left));
      gridLine.setAttribute("y1", y.toFixed(2));
      gridLine.setAttribute("x2", String(left + width));
      gridLine.setAttribute("y2", y.toFixed(2));
      stage6ProfileGrid.appendChild(gridLine);

      const tickLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
      tickLine.setAttribute("class", "stage6-plot-tick");
      tickLine.setAttribute("x1", String(left - 6));
      tickLine.setAttribute("y1", y.toFixed(2));
      tickLine.setAttribute("x2", String(left));
      tickLine.setAttribute("y2", y.toFixed(2));
      stage6ProfileYTicks.appendChild(tickLine);

      const tickLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      tickLabel.setAttribute("class", "stage6-plot-tick-label");
      tickLabel.setAttribute("x", String(left - 10));
      tickLabel.setAttribute("y", (y + 3).toFixed(2));
      tickLabel.setAttribute("text-anchor", "end");
      tickLabel.textContent = fmtNum(value);
      stage6ProfileYTicks.appendChild(tickLabel);
    }

    const crop = stage6EffectiveCrop(activeProfile);
    const draftDirty = !sameCrop(crop, activeProfile.crop || { left: 0, right: 0 });
    const leftIndex = clamp(Number(crop.left || 0), 0, Math.max(values.length - 2, 0));
    const rightIndex = clamp(values.length - 1 - Number(crop.right || 0), 1, values.length - 1);
    stage6ProfileTitle.textContent = activeProfile.label;
    stage6ProfileDetail.textContent =
      `${state.stage6DisplayMode === "normalized" ? "Normalized" : "Raw"} ${activeProfile.profileAxis} profile` +
      ` · averaged across ${activeProfile.averagedAxis}` +
      ` · ${activeProfile.sampleCount} pixels per sample` +
      ` · crop p=${leftIndex}, q=${Number(crop.right || 0)}` +
      ` · range ${fmtNum(valueMin)} .. ${fmtNum(valueMax)}` +
      (activeProfile.fit ? ` · fit rmse ${fmtNum(activeProfile.fit.rmse)}` : " · fit unavailable");
    stage6ProfilePolyline.setAttribute("points", points);
    stage6ProfileFitPolyline.setAttribute("points", fitPoints);
    stage6ProfileHitbox.setAttribute("x", String(left));
    stage6ProfileHitbox.setAttribute("y", String(top));
    stage6ProfileHitbox.setAttribute("width", String(width));
    stage6ProfileHitbox.setAttribute("height", String(height));
    stage6SaveWindowButton.disabled = !draftDirty;
    stage6RunFitButton.disabled = draftDirty;
    if (!activeProfile.fit) {
      stage6FitCoeffs.textContent = "No fit yet.";
    } else {
      stage6FitCoeffs.textContent =
        `terms=${activeProfile.fit.harmonicCount}  rmse=${fmtNum(activeProfile.fit.rmse)}  period=${fmtNum(activeProfile.fit.periodSamples)}  mtf1=${fmtNum(activeProfile.fit.mtf?.firstHarmonicMtf)}\n` +
        `phase=${fmtNum(activeProfile.fit.phaseRad)}  slope=${fmtNum(activeProfile.fit.slope)}  intercept=${fmtNum(activeProfile.fit.intercept)}\n` +
        `a=${activeProfile.fit.harmonicAmplitudes.map((value) => fmtNum(value)).join(", ")}`;
    }
    const cropLeftX = pointXs[leftIndex];
    const cropRightX = pointXs[rightIndex];
    stage6CropMaskLeft.setAttribute("x", String(left));
    stage6CropMaskLeft.setAttribute("y", String(top));
    stage6CropMaskLeft.setAttribute("width", Math.max(0, cropLeftX - left).toFixed(2));
    stage6CropMaskLeft.setAttribute("height", String(height));
    stage6CropMaskRight.setAttribute("x", cropRightX.toFixed(2));
    stage6CropMaskRight.setAttribute("y", String(top));
    stage6CropMaskRight.setAttribute("width", Math.max(0, left + width - cropRightX).toFixed(2));
    stage6CropMaskRight.setAttribute("height", String(height));
    stage6CropLeftLine.setAttribute("x1", cropLeftX.toFixed(2));
    stage6CropLeftLine.setAttribute("y1", String(top));
    stage6CropLeftLine.setAttribute("x2", cropLeftX.toFixed(2));
    stage6CropLeftLine.setAttribute("y2", String(top + height));
    stage6CropRightLine.setAttribute("x1", cropRightX.toFixed(2));
    stage6CropRightLine.setAttribute("y1", String(top));
    stage6CropRightLine.setAttribute("x2", cropRightX.toFixed(2));
    stage6CropRightLine.setAttribute("y2", String(top + height));
    stage6CropLeftHandle.setAttribute("cx", cropLeftX.toFixed(2));
    stage6CropLeftHandle.setAttribute("cy", (top + 14).toFixed(2));
    stage6CropRightHandle.setAttribute("cx", cropRightX.toFixed(2));
    stage6CropRightHandle.setAttribute("cy", (top + 14).toFixed(2));
    stage6ProfileCrop.hidden = false;
    stage6ProfilePlot.hidden = false;
    stage6ProfileEmpty.hidden = true;

    if (!state.interaction.stage6HoverActive || state.interaction.stage6HoverIndex == null) {
      stage6HoverReadout.hidden = true;
      stage6ProfileHover.hidden = true;
      return;
    }
    const hoverIndex = clamp(state.interaction.stage6HoverIndex, 0, values.length - 1);
    const hoverX = pointXs[hoverIndex];
    const hoverY = pointYs[hoverIndex];
    stage6ProfileVLine.setAttribute("x1", hoverX.toFixed(2));
    stage6ProfileVLine.setAttribute("y1", String(top));
    stage6ProfileVLine.setAttribute("x2", hoverX.toFixed(2));
    stage6ProfileVLine.setAttribute("y2", String(top + height));
    stage6ProfileHLine.setAttribute("x1", String(left));
    stage6ProfileHLine.setAttribute("y1", hoverY.toFixed(2));
    stage6ProfileHLine.setAttribute("x2", String(left + width));
    stage6ProfileHLine.setAttribute("y2", hoverY.toFixed(2));
    stage6ProfileMarker.setAttribute("cx", hoverX.toFixed(2));
    stage6ProfileMarker.setAttribute("cy", hoverY.toFixed(2));
    stage6HoverSample.textContent = `Sample ${hoverIndex + 1} / ${values.length}`;
    stage6HoverValue.textContent = `Value ${fmtNum(values[hoverIndex])}`;
    stage6HoverReadout.hidden = false;
    stage6ProfileHover.hidden = false;
    state.interaction.stage6HoverIndex = hoverIndex;
  };

  const renderStage7Panel = () => {
    if (
      !stage7Panel
      || !stage7Summary
      || !stage7XCount
      || !stage7YCount
      || !stage7Plot
      || !stage7Grid
      || !stage7YTicks
      || !stage7XTicks
      || !stage7LineX
      || !stage7LineY
      || !stage7PointsX
      || !stage7PointsY
      || !stage7Empty
    ) {
      return;
    }

    const showStage7 = state.stageName === "stage_7";
    stage7Panel.hidden = !showStage7;
    if (!showStage7) {
      return;
    }

    const result = currentStage7Result();
    const xCurve = result?.curves?.X || [];
    const yCurve = result?.curves?.Y || [];
    const allPoints = [...xCurve, ...yCurve];

    stage7XCount.textContent = `X: ${xCurve.length} point${xCurve.length === 1 ? "" : "s"}`;
    stage7YCount.textContent = `Y: ${yCurve.length} point${yCurve.length === 1 ? "" : "s"}`;
    stage7Summary.textContent = result
      ? `${result.summary?.fittedProfiles || 0} / ${result.summary?.totalProfiles || 0} Stage 6 profiles have saved fits.`
      : "Stage 7 will aggregate saved Stage 6 fits into one MTF curve per axis.";

    if (allPoints.length === 0) {
      stage7Plot.hidden = true;
      stage7Empty.hidden = false;
      stage7Grid.innerHTML = "";
      stage7YTicks.innerHTML = "";
      stage7XTicks.innerHTML = "";
      stage7LineX.setAttribute("points", "");
      stage7LineY.setAttribute("points", "");
      stage7PointsX.innerHTML = "";
      stage7PointsY.innerHTML = "";
      return;
    }

    const left = 72;
    const right = 24;
    const top = 22;
    const bottom = 48;
    const width = 720 - left - right;
    const height = 340 - top - bottom;
    const minFreq = Math.min(...allPoints.map((point) => Number(point.frequencyLpPerMm)));
    const maxFreq = Math.max(...allPoints.map((point) => Number(point.frequencyLpPerMm)));
    const freqSpan = Math.max(maxFreq - minFreq, 1e-6);
    const minMtf = Math.min(0, ...allPoints.map((point) => Number(point.mtf)));
    const maxMtf = Math.max(1, ...allPoints.map((point) => Number(point.mtf)));
    const mtfSpan = Math.max(maxMtf - minMtf, 1e-6);
    const xAt = (frequency) => left + (((Number(frequency) - minFreq) / freqSpan) * width);
    const yAt = (mtf) => top + ((1 - ((Number(mtf) - minMtf) / mtfSpan)) * height);
    const seriesPoints = (curve) => curve.map((point) => `${xAt(point.frequencyLpPerMm).toFixed(2)},${yAt(point.mtf).toFixed(2)}`).join(" ");

    stage7Grid.innerHTML = "";
    stage7YTicks.innerHTML = "";
    stage7XTicks.innerHTML = "";
    stage7PointsX.innerHTML = "";
    stage7PointsY.innerHTML = "";

    const yTickCount = 5;
    for (let tickIndex = 0; tickIndex < yTickCount; tickIndex += 1) {
      const t = yTickCount === 1 ? 0 : tickIndex / (yTickCount - 1);
      const y = top + (t * height);
      const value = maxMtf - (t * mtfSpan);

      const gridLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
      gridLine.setAttribute("class", "stage7-plot-grid-line");
      gridLine.setAttribute("x1", String(left));
      gridLine.setAttribute("y1", y.toFixed(2));
      gridLine.setAttribute("x2", String(left + width));
      gridLine.setAttribute("y2", y.toFixed(2));
      stage7Grid.appendChild(gridLine);

      const tickLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
      tickLine.setAttribute("class", "stage7-plot-tick");
      tickLine.setAttribute("x1", String(left - 6));
      tickLine.setAttribute("y1", y.toFixed(2));
      tickLine.setAttribute("x2", String(left));
      tickLine.setAttribute("y2", y.toFixed(2));
      stage7YTicks.appendChild(tickLine);

      const tickLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      tickLabel.setAttribute("class", "stage7-plot-tick-label");
      tickLabel.setAttribute("x", String(left - 10));
      tickLabel.setAttribute("y", (y + 3).toFixed(2));
      tickLabel.setAttribute("text-anchor", "end");
      tickLabel.textContent = fmtNum(value);
      stage7YTicks.appendChild(tickLabel);
    }

    const xTickCount = Math.min(6, Math.max(allPoints.length, 2));
    for (let tickIndex = 0; tickIndex < xTickCount; tickIndex += 1) {
      const t = xTickCount === 1 ? 0 : tickIndex / (xTickCount - 1);
      const x = left + (t * width);
      const value = minFreq + (t * freqSpan);

      const gridLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
      gridLine.setAttribute("class", "stage7-plot-grid-line");
      gridLine.setAttribute("x1", x.toFixed(2));
      gridLine.setAttribute("y1", String(top));
      gridLine.setAttribute("x2", x.toFixed(2));
      gridLine.setAttribute("y2", String(top + height));
      stage7Grid.appendChild(gridLine);

      const tickLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
      tickLine.setAttribute("class", "stage7-plot-tick");
      tickLine.setAttribute("x1", x.toFixed(2));
      tickLine.setAttribute("y1", String(top + height));
      tickLine.setAttribute("x2", x.toFixed(2));
      tickLine.setAttribute("y2", String(top + height + 6));
      stage7XTicks.appendChild(tickLine);

      const tickLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      tickLabel.setAttribute("class", "stage7-plot-tick-label");
      tickLabel.setAttribute("x", x.toFixed(2));
      tickLabel.setAttribute("y", String(top + height + 18));
      tickLabel.setAttribute("text-anchor", "middle");
      tickLabel.textContent = fmtNum(value);
      stage7XTicks.appendChild(tickLabel);
    }

    const appendPointMarkers = (node, curve, className) => {
      for (const point of curve) {
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("class", className);
        circle.setAttribute("cx", xAt(point.frequencyLpPerMm).toFixed(2));
        circle.setAttribute("cy", yAt(point.mtf).toFixed(2));
        circle.setAttribute("r", "4");
        node.appendChild(circle);
      }
    };

    stage7LineX.setAttribute("points", seriesPoints(xCurve));
    stage7LineY.setAttribute("points", seriesPoints(yCurve));
    appendPointMarkers(stage7PointsX, xCurve, "stage7-plot-point stage7-plot-point-x");
    appendPointMarkers(stage7PointsY, yCurve, "stage7-plot-point stage7-plot-point-y");
    stage7Plot.hidden = false;
    stage7Empty.hidden = true;
  };

  const applySessionPayload = async ({ pipeline, source_summary: sourceSummary, current_stage_result: currentStageResult }) => {
    state.pipeline = pipeline;
    state.sourceSummary = sourceSummary;
    const stageInfo = currentStageInfo();
    state.stageName = stageInfo ? stageInfo.name : null;
    state.stageMode = stageInfo ? stageInfo.mode : "canvas";

    if (currentStageResult && state.stageName) {
      state.stageResults[state.stageName] = currentStageResult;
    }
    if (state.stageName === "scale") {
      state.draftScaleGroups = [...(currentScaleResult()?.groups || [])];
    } else if (state.stageName === "bar_rois") {
      state.activeBarRoiKey = findNextIncompleteBarRoi();
    } else if (state.stageName === "norm_rois") {
      state.activeNormRoiKey = findNextIncompleteNormRoi();
    } else if (state.stageName === "stage_6") {
      state.activeStage6ProfileKey = selectStage6ProfileKey(state.activeStage6ProfileKey);
      const activeProfile = currentStage6Profile();
      if (activeProfile?.fit?.harmonicCount) {
        state.stage6HarmonicCount = activeProfile.fit.harmonicCount;
      }
      state.interaction.stage6HoverIndex = null;
      state.interaction.stage6HoverActive = false;
    }

    await ensureSourceBuffer();
    updateHeader();
    applyLayoutMode();
    updateStageChrome();
    renderScaleTools();
    renderBarRoiTools();
    renderNormRoiTools();
    renderStage6Panel();
    renderStage7Panel();
    renderAll();

    if (state.stageName === "anchor" && !currentAnchorResult()) {
      sendCommand("run_anchor");
    }
  };

  const applyStageChangedPayload = async ({ pipeline, stage_mode: stageMode, stage_result: stageResult }, stage) => {
    state.pipeline = pipeline;
    const stageInfo = currentStageInfo();
    state.stageName = stageInfo ? stageInfo.name : stage;
    state.stageMode = stageMode || (stageInfo ? stageInfo.mode : "canvas");
    if (stageResult && state.stageName) {
      state.stageResults[state.stageName] = stageResult;
    }
    if (state.stageName === "scale") {
      state.draftScaleGroups = [...(currentScaleResult()?.groups || [])];
    } else if (state.stageName === "bar_rois") {
      state.activeBarRoiKey = findNextIncompleteBarRoi(state.activeBarRoiKey);
    } else if (state.stageName === "norm_rois") {
      state.activeNormRoiKey = findNextIncompleteNormRoi(state.activeNormRoiKey);
    } else if (state.stageName === "stage_6") {
      state.activeStage6ProfileKey = selectStage6ProfileKey(state.activeStage6ProfileKey);
      const activeProfile = currentStage6Profile();
      if (activeProfile?.fit?.harmonicCount) {
        state.stage6HarmonicCount = activeProfile.fit.harmonicCount;
      }
      state.interaction.stage6HoverIndex = null;
      state.interaction.stage6HoverActive = false;
    }
    updateHeader();
    applyLayoutMode();
    updateStageChrome();
    renderScaleTools();
    renderBarRoiTools();
    renderNormRoiTools();
    renderStage6Panel();
    renderStage7Panel();
    renderAll();

    if (state.stageName === "anchor" && !currentAnchorResult()) {
      sendCommand("run_anchor");
    } else if (state.stageMode === "canvas") {
      setRunStatus("Ready", "ready");
    } else {
      setRunStatus("Frame stage ready", "ready");
    }
  };

  const ensureSourceBuffer = async () => {
    const summary = state.sourceSummary;
    if (!summary) {
      state.data = null;
      state.sourceKey = null;
      return;
    }

    const nextKey = `${summary.fileName}:${summary.byteLength}:${summary.rows}:${summary.cols}`;
    if (state.sourceKey === nextKey && state.data) {
      return;
    }

    const buffer = await fetchSourceBuffer();
    const data = new Float32Array(buffer);
    const expectedLength = summary.rows * summary.cols;
    if (data.length !== expectedLength) {
      throw new Error("Source buffer length does not match source metadata.");
    }

    state.data = data;
    state.sourceKey = nextKey;
    state.display.window = Math.max(summary.dataMax - summary.dataMin, 0.001);
    state.display.level = (summary.dataMax + summary.dataMin) / 2;
    requestAnimationFrame(() => fitToView());
  };

  const handleSocketMessage = async (event) => {
    const message = JSON.parse(event.data);
    const { type, stage, payload } = message;

    if (type === "session_ready") {
      await applySessionPayload(payload);
      setRunStatus("Session ready", "ready");
      return;
    }

    if (type === "stage_changed") {
      await applyStageChangedPayload(payload, stage);
      return;
    }

    if (type === "config_applied") {
      state.pipeline = payload.pipeline;
      state.stageName = "stage_6";
      state.stageMode = "frame";
      state.stageResults.anchor = payload.anchor;
      state.stageResults.scale = payload.scale;
      state.stageResults.bar_rois = payload.bar_rois;
      state.stageResults.norm_rois = payload.norm_rois;
      state.stageResults.stage_6 = payload.stage_6;
      state.activeStage6ProfileKey = selectStage6ProfileKey(state.activeStage6ProfileKey);
      const activeProfile = currentStage6Profile();
      if (activeProfile?.fit?.harmonicCount) {
        state.stage6HarmonicCount = activeProfile.fit.harmonicCount;
      }
      state.interaction.stage6HoverIndex = null;
      state.interaction.stage6HoverActive = false;
      updateHeader();
      applyLayoutMode();
      updateStageChrome();
      renderScaleTools();
      renderBarRoiTools();
      renderNormRoiTools();
      renderStage6Panel();
      renderStage7Panel();
      renderAll();
      setRunStatus(
        `Existing config applied (dx ${payload.translation.x.toFixed(1)}, dy ${payload.translation.y.toFixed(1)})`,
        "ready",
      );
      return;
    }

    if (type === "stage_result") {
      if (stage) {
        state.stageResults[stage] = payload;
      }
      if (stage === "scale") {
        state.draftScaleGroups = [...(payload.groups || [])];
        if (state.pendingScaleAdvance) {
          state.pendingScaleAdvance = false;
          sendCommand("advance_stage");
        }
      } else if (stage === "bar_rois") {
        state.activeBarRoiKey = findNextIncompleteBarRoi(state.activeBarRoiKey);
      } else if (stage === "norm_rois") {
        state.activeNormRoiKey = findNextIncompleteNormRoi(state.activeNormRoiKey);
      } else if (stage === "stage_6") {
        state.activeStage6ProfileKey = selectStage6ProfileKey(state.activeStage6ProfileKey);
        const activeProfile = currentStage6Profile();
        if (activeProfile?.fit?.harmonicCount) {
          state.stage6HarmonicCount = activeProfile.fit.harmonicCount;
        }
        if (state.activeStage6ProfileKey) {
          delete state.stage6DraftCrops[state.activeStage6ProfileKey];
        }
        state.interaction.stage6HoverIndex = null;
        state.interaction.stage6HoverActive = false;
        state.interaction.stage6CropPreview = null;
        state.interaction.stage6CropDrag = null;
      }
      updateStageChrome();
      renderScaleTools();
      renderBarRoiTools();
      renderNormRoiTools();
      renderStage6Panel();
      renderStage7Panel();
      renderAll();
      return;
    }

    if (type === "stage_status") {
      const tone = payload.status === "error" ? "error" : payload.status === "running" ? "busy" : "ready";
      setRunStatus(payload.detail || payload.status, tone);
      if (stage === "anchor" && payload.status === "ready") {
        updateStageChrome();
      }
      return;
    }

    if (type === "stage_progress") {
      const detail = payload.fraction != null
        ? `${payload.label} (${Math.round(payload.fraction * 100)}%)`
        : payload.label;
      setRunStatus(detail, "busy");
      return;
    }

    if (type === "stage_error") {
      setRunStatus(payload.detail || "Workflow error", "error");
      updateStageChrome();
      return;
    }

    if (type === "document_reset") {
      window.location.assign("/");
    }
  };

  const sendCommand = (type, stage = null, payload = null) => {
    if (!state.socketReady || !state.socket) {
      setRunStatus("Workflow session is not connected", "error");
      return;
    }
    state.socket.send(JSON.stringify({ type, stage, payload }));
  };

  const connectSocket = async () => {
    const socketUrl = buildWebSocketUrl(config.websocketPath);
    const socket = new WebSocket(socketUrl);
    state.socket = socket;

    await new Promise((resolve, reject) => {
      socket.addEventListener("open", resolve, { once: true });
      socket.addEventListener("error", () => reject(new Error("Could not open workflow session.")), { once: true });
    });

    state.socketReady = true;
    socket.addEventListener("message", (event) => {
      handleSocketMessage(event).catch((error) => setRunStatus(error.message, "error"));
    });
    socket.addEventListener("close", () => {
      state.socketReady = false;
      setRunStatus("Workflow session closed", "error");
    });
    sendCommand("bootstrap");
  };

  const commitBarRoi = async (key, rect) => {
    sendCommand("set_bar_roi", "bar_rois", { key, rect });
  };

  const clearBarRoi = async (key) => {
    sendCommand("clear_bar_roi", "bar_rois", { key });
  };

  const commitNormRoi = async (key, rect) => {
    sendCommand("set_norm_roi", "norm_rois", { key, rect });
  };

  const clearNormRoi = async (key) => {
    sendCommand("clear_norm_roi", "norm_rois", { key });
  };

  const commitStage6Crop = (profile) => {
    if (!profile) {
      return;
    }
    const draft = state.stage6DraftCrops[profile.key];
    if (!draft || sameCrop(draft, profile.crop || { left: 0, right: 0 })) {
      return;
    }
    sendCommand("set_stage_6_crop", "stage_6", { key: profile.key, left: draft.left, right: draft.right });
  };

  const runStage6Fit = (profile) => {
    if (!profile) {
      return;
    }
    sendCommand("run_stage_6_fit", "stage_6", {
      key: profile.key,
      harmonicCount: state.stage6HarmonicCount,
    });
  };

  const setStage6DisplayMode = (mode) => {
    if (mode !== "raw" && mode !== "normalized") {
      return;
    }
    if (mode === "normalized" && !currentStage6Result()?.normalization?.normalized) {
      return;
    }
    state.stage6DisplayMode = mode;
    state.interaction.stage6HoverIndex = null;
    state.interaction.stage6HoverActive = false;
    renderStage6Panel();
  };

  const setStage6HarmonicCount = (value) => {
    const next = clamp(Number(value) || 1, 1, 15);
    state.stage6HarmonicCount = Math.round(next);
    if (stage6HarmonicCountInput) {
      stage6HarmonicCountInput.value = String(state.stage6HarmonicCount);
    }
  };

  const handleStage6PointerDown = (event) => {
    if (state.stageName !== "stage_6") {
      return;
    }
    const activeProfile = currentStage6Profile();
    if (!activeProfile || !stage6ProfileHitbox) {
      return;
    }
    const crop = stage6EffectiveCrop(activeProfile);
    const rect = stage6ProfileHitbox.getBoundingClientRect();
    const x = clamp(event.clientX - rect.left, 0, rect.width);
    const length = Math.max(activeProfile.profileLength - 1, 1);
    const leftX = (crop.left / length) * rect.width;
    const rightX = ((length - crop.right) / length) * rect.width;
    state.interaction.stage6CropDrag = Math.abs(x - leftX) <= Math.abs(x - rightX) ? "left" : "right";
    state.interaction.stage6CropPreview = { key: activeProfile.key, left: crop.left, right: crop.right };
    state.interaction.stage6HoverActive = false;
    if (stage6ProfileHitbox?.setPointerCapture) {
      stage6ProfileHitbox.setPointerCapture(event.pointerId);
    }
    event.preventDefault();
  };

  const handleStage6PointerMove = (event) => {
    if (state.stageName !== "stage_6") {
      return;
    }
    const activeProfile = currentStage6Profile();
    if (!activeProfile || !Array.isArray(activeProfile.rawProfile) || activeProfile.rawProfile.length === 0 || !stage6ProfileHitbox) {
      return;
    }
    const rect = stage6ProfileHitbox.getBoundingClientRect();
    const x = clamp(event.clientX - rect.left, 0, rect.width);
    const fraction = rect.width <= 0 ? 0 : x / rect.width;
    const index = Math.round(fraction * Math.max(activeProfile.profileLength - 1, 0));
    if (state.interaction.stage6CropDrag) {
      const next = { ...(state.interaction.stage6CropPreview || stage6EffectiveCrop(activeProfile)), key: activeProfile.key };
      if (state.interaction.stage6CropDrag === "left") {
        next.left = clamp(index, 0, Math.max(activeProfile.profileLength - 2 - next.right, 0));
      } else {
        next.right = clamp(activeProfile.profileLength - 1 - index, 0, Math.max(activeProfile.profileLength - 2 - next.left, 0));
      }
      state.interaction.stage6CropPreview = next;
      state.stage6DraftCrops[activeProfile.key] = { left: next.left, right: next.right };
      renderStage6Panel();
      return;
    }
    state.interaction.stage6HoverIndex = index;
    state.interaction.stage6HoverActive = true;
    renderStage6Panel();
  };

  const handleStage6PointerLeave = () => {
    if (state.stageName !== "stage_6") {
      return;
    }
    if (state.interaction.stage6CropDrag) {
      return;
    }
    state.interaction.stage6HoverIndex = null;
    state.interaction.stage6HoverActive = false;
    renderStage6Panel();
  };

  const uploadReplacementSource = async (file) => {
    try {
      await uploadSourceFile(file);
      window.location.assign("/");
    } catch {
      setRunStatus("Source replacement failed", "error");
    } finally {
      fileInput.value = "";
    }
  };

  const handleHistogramPointerDown = (event) => {
    if (state.stageMode !== "canvas") {
      return;
    }

    const rect = histogramContainer.getBoundingClientRect();
    const fx = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    const winLo = state.display.level - state.display.window / 2;
    const winHi = state.display.level + state.display.window / 2;

    if (Math.abs(fx - winLo) < 0.02) {
      state.interaction.histDragMode = "left-edge";
    } else if (Math.abs(fx - winHi) < 0.02) {
      state.interaction.histDragMode = "right-edge";
    } else if (fx > winLo && fx < winHi) {
      state.interaction.histDragMode = "center";
    } else {
      state.display.level = fx;
      state.interaction.histDragMode = "center";
      renderAll();
    }

    state.interaction.histDragStartX = fx;
    state.interaction.histDragStartLevel = state.display.level;
    state.interaction.histDragStartWindow = state.display.window;
  };

  const handleHistogramPointerMove = (event) => {
    if (!state.interaction.histDragMode || state.stageMode !== "canvas") {
      return;
    }

    const rect = histogramContainer.getBoundingClientRect();
    const fx = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    const delta = fx - state.interaction.histDragStartX;

    if (state.interaction.histDragMode === "center") {
      state.display.level = clamp(state.interaction.histDragStartLevel + delta, 0, 1);
    } else if (state.interaction.histDragMode === "left-edge") {
      const right = state.interaction.histDragStartLevel + state.interaction.histDragStartWindow / 2;
      const left = clamp(
        state.interaction.histDragStartLevel - state.interaction.histDragStartWindow / 2 + delta,
        0,
        right - 0.001,
      );
      state.display.window = right - left;
      state.display.level = (left + right) / 2;
    } else if (state.interaction.histDragMode === "right-edge") {
      const left = state.interaction.histDragStartLevel - state.interaction.histDragStartWindow / 2;
      const right = clamp(
        state.interaction.histDragStartLevel + state.interaction.histDragStartWindow / 2 + delta,
        left + 0.001,
        1,
      );
      state.display.window = right - left;
      state.display.level = (left + right) / 2;
    }

    state.display.window = clamp(state.display.window, 0.001, 1);
    state.display.level = clamp(state.display.level, 0, 1);
    renderAll();
  };

  const handleMouseWheel = (event) => {
    if (!state.data || state.stageMode !== "canvas") {
      return;
    }

    event.preventDefault();
    const rect = canvasContainer.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    const oldZoom = state.display.zoom;
    const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;

    state.display.zoom = clamp(state.display.zoom * factor, 0.05, 128);
    state.display.panX = mouseX - (mouseX - state.display.panX) * (state.display.zoom / oldZoom);
    state.display.panY = mouseY - (mouseY - state.display.panY) * (state.display.zoom / oldZoom);
    syncTransforms();
  };

  const handleCanvasMouseDown = (event) => {
    if (event.button !== 0 || !event.shiftKey || state.stageMode !== "canvas") {
      return;
    }

    state.interaction.isPanning = true;
    state.interaction.panStartX = event.clientX - state.display.panX;
    state.interaction.panStartY = event.clientY - state.display.panY;
  };

  const handleWindowMouseMove = (event) => {
    if (state.interaction.isPanning) {
      state.display.panX = event.clientX - state.interaction.panStartX;
      state.display.panY = event.clientY - state.interaction.panStartY;
      syncTransforms();
    }

    updateInspector(event.clientX, event.clientY);
    handleHistogramPointerMove(event);
  };

  const finishStage6CropDrag = () => {
    if (!state.interaction.stage6CropDrag) {
      return;
    }
    state.interaction.stage6CropDrag = null;
    state.interaction.stage6CropPreview = null;
    const profile = currentStage6Profile();
    if (profile) {
      setRunStatus("Fit window adjusted. Use Save Window to persist it.", "ready");
      renderStage6Panel();
    }
  };

  const handleWindowMouseUp = () => {
    state.interaction.isPanning = false;
    state.interaction.histDragMode = null;
  };

  const handleStage6PointerUp = () => {
    finishStage6CropDrag();
  };

  $("btn-open").addEventListener("click", () => fileInput.click());
  prevButton.addEventListener("click", () => sendCommand("retreat_stage"));
  $("btn-fit").addEventListener("click", fitToView);
  $("btn-1x").addEventListener("click", setOneToOne);
  histogramToggleButton.addEventListener("click", toggleHistogramPanel);
  $("hist-auto-btn").addEventListener("click", autoWindowLevel);
  $("hist-reset-btn").addEventListener("click", resetWindowLevel);
  autocompleteButton.addEventListener("click", () => sendCommand("auto_complete_stage", state.stageName));
  applyConfigButton.addEventListener("click", () => {
    if (state.stageName === "anchor" && currentAnchorResult() && Number(state.reusableConfig?.count || 0) > 0) {
      sendCommand("apply_reusable_config", "anchor");
    }
  });
  rerunButton.addEventListener("click", () => sendCommand("rerun_stage", "anchor"));
  newButton.addEventListener("click", () => sendCommand("reset_document"));
  sidebarNextButton.addEventListener("click", () => {
    if (state.stageName === "scale") {
      state.pendingScaleAdvance = true;
      sendCommand("submit_scale_groups", "scale", { groups: state.draftScaleGroups });
      return;
    }
    sendCommand("advance_stage");
  });
  barRoiNextButton.addEventListener("click", () => {
    if (state.stageName === "bar_rois") {
      sendCommand("advance_stage");
    }
  });
  clearNormRoiButton.addEventListener("click", () => {
    if (state.stageName === "norm_rois" && state.activeNormRoiKey) {
      clearNormRoi(state.activeNormRoiKey);
    }
  });
  normRoiNextButton.addEventListener("click", () => {
    if (state.stageName === "norm_rois") {
      sendCommand("advance_stage");
    }
  });
  clearRoiButton.addEventListener("click", () => {
    if (state.stageName === "bar_rois" && state.activeBarRoiKey) {
      clearBarRoi(state.activeBarRoiKey);
    }
  });
  nextButton.addEventListener("click", () => {
    nextButton.disabled = true;
    sendCommand("advance_stage");
  });

  fileInput.addEventListener("change", (event) => {
    if (event.target.files && event.target.files[0]) {
      uploadReplacementSource(event.target.files[0]);
    }
  });

  histogramContainer.addEventListener("mousedown", handleHistogramPointerDown);
  canvasContainer.addEventListener("wheel", handleMouseWheel, { passive: false });
  canvasContainer.addEventListener("mousedown", handleCanvasMouseDown);
  if (stage6RawButton) {
    stage6RawButton.addEventListener("click", () => setStage6DisplayMode("raw"));
  }
  if (stage6NormalizedButton) {
    stage6NormalizedButton.addEventListener("click", () => setStage6DisplayMode("normalized"));
  }
  if (stage6SaveWindowButton) {
    stage6SaveWindowButton.addEventListener("click", () => commitStage6Crop(currentStage6Profile()));
  }
  if (stage6RunFitButton) {
    stage6RunFitButton.addEventListener("click", () => runStage6Fit(currentStage6Profile()));
  }
  if (stage6HarmonicCountInput) {
    stage6HarmonicCountInput.addEventListener("change", (event) => setStage6HarmonicCount(event.target.value));
  }
  if (stage6ProfileHitbox) {
    stage6ProfileHitbox.addEventListener("pointerdown", handleStage6PointerDown);
    stage6ProfileHitbox.addEventListener("pointermove", handleStage6PointerMove);
    stage6ProfileHitbox.addEventListener("pointerleave", handleStage6PointerLeave);
    stage6ProfileHitbox.addEventListener("pointerup", handleStage6PointerUp);
    stage6ProfileHitbox.addEventListener("pointercancel", handleStage6PointerUp);
  }
  window.addEventListener("mousemove", handleWindowMouseMove);
  window.addEventListener("mouseup", handleWindowMouseUp);
  window.addEventListener("pointerup", handleStage6PointerUp);

  window.addEventListener("keydown", (event) => {
    if (event.target && event.target.tagName === "INPUT") {
      return;
    }

    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "o") {
      event.preventDefault();
      fileInput.click();
    } else if (event.key === "f") {
      fitToView();
    } else if (event.key === "1") {
      setOneToOne();
    } else if (event.key === "a") {
      autoWindowLevel();
    } else if (event.key === "r" && state.stageName === "anchor") {
      sendCommand("rerun_stage", "anchor");
    }
  });

  window.addEventListener("resize", () => {
    if (state.data && state.stageMode === "canvas") {
      renderAll();
    }
  });

  try {
    roiController = createROIController({
      canvas: overlayCanvas,
      canvasContainer,
      display: state.display,
      getConfig: () => state.sourceSummary,
      getSequence: () => (
        state.stageName === "bar_rois"
          ? currentBarRoiResult()?.slots || []
          : state.stageName === "norm_rois"
            ? currentNormRoiResult()?.slots || []
            : []
      ),
      getActiveKey: () => (
        state.stageName === "bar_rois"
          ? state.activeBarRoiKey
          : state.stageName === "norm_rois"
            ? state.activeNormRoiKey
            : null
      ),
      setActiveKey: (key) => {
        if (state.stageName === "bar_rois") {
          state.activeBarRoiKey = key;
          renderBarRoiTools();
        } else if (state.stageName === "norm_rois") {
          state.activeNormRoiKey = key;
          renderNormRoiTools();
        }
      },
      onCommitRect: (key, rect) => (
        state.stageName === "bar_rois" ? commitBarRoi(key, rect) : commitNormRoi(key, rect)
      ),
      onClearRect: (key) => (
        state.stageName === "bar_rois" ? clearBarRoi(key) : clearNormRoi(key)
      ),
      adjustDraftRect: (slot, rect, sequence) => {
        if (slot.tone !== "white") {
          return rect;
        }
        const blackSlot = sequence.find((entry) => entry.tone === "black");
        if (!blackSlot?.rect) {
          return rect;
        }
        return {
          row: rect.row,
          col: rect.col,
          width: blackSlot.rect.width,
          height: blackSlot.rect.height,
        };
      },
    });
    await connectSocket();
  } catch (error) {
    setRunStatus(error.message, "error");
  }
}

function buildWebSocketUrl(path) {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}${path}`;
}

async function uploadSourceFile(file) {
  const query = new URLSearchParams({ name: file.name });
  const response = await fetch(`/actions/load?${query.toString()}`, {
    method: "POST",
    body: file,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Upload failed with ${response.status}.`);
  }
}

async function fetchSourceBuffer() {
  const response = await fetch("/api/source");
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Source request failed with ${response.status}.`);
  }

  return response.arrayBuffer();
}
