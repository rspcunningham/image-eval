const $ = (id) => document.getElementById(id);

const state = {
  config: null,
  image: null,
  mode: "draw",
  display: {
    zoom: 1,
    panX: 0,
    panY: 0,
  },
  spacePan: false,
  panning: false,
  drawing: false,
  panStartX: 0,
  panStartY: 0,
  drawStart: null,
  drawCurrent: null,
  selection: null,
};

const canvasContainer = $("canvas-container");
const imageCanvas = $("image-canvas");
const overlayCanvas = $("overlay-canvas");
const imageCtx = imageCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");
const zoomLabel = $("zoom-label");
const modeLabel = $("mode-label");
const toolTitle = $("tool-title");
const promptPanel = $("prompt-panel");
const promptCopy = $("prompt-copy");
const interactionCopy = $("interaction-copy");
const selectionMeta = $("selection-meta");
const acceptButton = $("btn-accept");
const drawButton = $("btn-draw");
const panButton = $("btn-pan");
const fitButton = $("btn-fit");
const oneToOneButton = $("btn-1x");
const cancelButton = $("btn-cancel");

async function init() {
  await waitForPywebview();

  if (typeof window.pywebview?.api?.get_request !== "function") {
    throw new Error("pywebview bridge did not expose get_request()");
  }

  state.config = await window.pywebview.api.get_request();
  if (typeof state.config.imageDataUrl === "string") {
    state.image = await loadImage(state.config.imageDataUrl);
    imageCanvas.width = state.config.cols;
    imageCanvas.height = state.config.rows;
    overlayCanvas.width = state.config.cols;
    overlayCanvas.height = state.config.rows;
    imageCtx.drawImage(state.image, 0, 0, state.config.cols, state.config.rows);
  }

  bindEvents();
  applyToolConfig();
  fitToView();
  renderAll();
}

function waitForPywebview() {
  return new Promise((resolve) => {
    const isReady = () => typeof window.pywebview?.api?.get_request === "function";

    if (isReady()) {
      resolve();
      return;
    }

    let attempts = 0;
    let intervalId = 0;
    const finishIfReady = () => {
      attempts += 1;
      if (isReady()) {
        window.clearInterval(intervalId);
        resolve();
      }
    };

    window.addEventListener("pywebviewready", () => {
      finishIfReady();
    });

    intervalId = window.setInterval(() => {
      finishIfReady();
      if (attempts >= 50) {
        window.clearInterval(intervalId);
        resolve();
      }
    }, 100);
  });
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load image"));
    image.src = src;
  });
}

function bindEvents() {
  $("btn-fit").addEventListener("click", fitToView);
  $("btn-1x").addEventListener("click", setOneToOne);
  $("btn-draw").addEventListener("click", () => setMode("draw"));
  $("btn-pan").addEventListener("click", () => setMode("pan"));
  cancelButton.addEventListener("click", cancelSelection);
  acceptButton.addEventListener("click", submitSelection);

  canvasContainer.addEventListener("wheel", handleWheel, { passive: false });
  canvasContainer.addEventListener("mousedown", handleMouseDown);
  canvasContainer.addEventListener("mousemove", handleMouseMove);
  window.addEventListener("mouseup", handleMouseUp);
  window.addEventListener("keydown", handleKeyDown);
  window.addEventListener("keyup", handleKeyUp);
  window.addEventListener("resize", fitToView);
}

function applyToolConfig() {
  const tool = state.config.tool ?? "select-roi";

  if (tool === "show-anchor") {
    state.mode = "pan";
    toolTitle.textContent = "Anchor Preview";
    promptPanel.hidden = false;
    promptCopy.textContent = state.config.prompt ?? "Inspect the detected anchor before continuing.";
    drawButton.hidden = true;
    fitButton.hidden = false;
    oneToOneButton.hidden = false;
    modeLabel.hidden = true;
    acceptButton.disabled = false;
    acceptButton.textContent = "Done";
    cancelButton.textContent = "Close";
    interactionCopy.textContent = "Inspect the detected anchor. Use the mouse wheel to zoom. Hold Shift and drag to pan. Press Enter, Escape, or Close when finished.";
    updateAnchorMeta();
    return;
  }

  if (tool === "show-rois") {
    state.mode = "pan";
    toolTitle.textContent = "Seeded ROI Review";
    promptPanel.hidden = false;
    promptCopy.textContent = "Inspect the translated ROIs loaded from roi_config.json before fitting starts.";
    drawButton.hidden = true;
    fitButton.hidden = false;
    oneToOneButton.hidden = false;
    modeLabel.hidden = true;
    acceptButton.disabled = false;
    acceptButton.textContent = "Continue";
    cancelButton.textContent = "Close";
    interactionCopy.textContent = "Inspect the translated normalization and bar ROIs. Use the mouse wheel to zoom. Hold Shift and drag to pan. Press Enter, Escape, or Close when finished.";
    updateRoisMeta();
    return;
  }

  toolTitle.textContent = "ROI Picker";
  promptPanel.hidden = false;
  promptCopy.textContent = resolvePromptCopy();
  drawButton.hidden = false;
  panButton.hidden = false;
  fitButton.hidden = false;
  oneToOneButton.hidden = false;
  modeLabel.hidden = false;
  zoomLabel.hidden = false;
  acceptButton.textContent = "Accept";
  cancelButton.textContent = "Cancel";
  setMode("draw");
  updateSelectionMeta();
}

function resolvePromptCopy() {
  if (typeof state.config.prompt === "string" && state.config.prompt.trim().length > 0) {
    return state.config.prompt.trim();
  }

  if (state.config.sizeRef) {
    return "Select the next ROI using the same size as the reference ROI.";
  }

  return "Select the ROI for the current analysis step.";
}

function setMode(mode) {
  state.mode = mode;
  drawButton.classList.toggle("toolbtn--active", mode === "draw");
  panButton.classList.toggle("toolbtn--active", mode === "pan");
  modeLabel.textContent = `Mode: ${mode === "draw" ? "Draw" : "Pan"}`;

  if (state.config.sizeRef) {
    interactionCopy.textContent = mode === "draw"
      ? "Click to place the fixed-size ROI. Use the mouse wheel to zoom. Hold Shift and drag to pan."
      : "Drag to pan. Switch back to Draw to place the fixed-size ROI.";
  } else {
    interactionCopy.textContent = mode === "draw"
      ? "Drag to draw. Use the mouse wheel to zoom. Hold Shift and drag to pan."
      : "Drag to pan. Switch back to Draw to place a new ROI.";
  }
}

function fitToView() {
  if (!state.config) {
    return;
  }

  const width = canvasContainer.clientWidth;
  const height = canvasContainer.clientHeight;
  const zoom = Math.max(0.05, Math.min((width / state.config.cols) * 0.96, (height / state.config.rows) * 0.96));
  state.display.zoom = zoom;
  state.display.panX = (width - state.config.cols * zoom) / 2;
  state.display.panY = (height - state.config.rows * zoom) / 2;
  renderAll();
}

function setOneToOne() {
  if (!state.config) {
    return;
  }

  state.display.zoom = 1;
  state.display.panX = (canvasContainer.clientWidth - state.config.cols) / 2;
  state.display.panY = (canvasContainer.clientHeight - state.config.rows) / 2;
  renderAll();
}

function handleWheel(event) {
  event.preventDefault();

  const rect = canvasContainer.getBoundingClientRect();
  const mouseX = event.clientX - rect.left;
  const mouseY = event.clientY - rect.top;
  const oldZoom = state.display.zoom;
  const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
  const nextZoom = clamp(state.display.zoom * factor, 0.05, 128);

  state.display.zoom = nextZoom;
  state.display.panX = mouseX - (mouseX - state.display.panX) * (nextZoom / oldZoom);
  state.display.panY = mouseY - (mouseY - state.display.panY) * (nextZoom / oldZoom);
  renderAll();
}

function handleMouseDown(event) {
  if (state.config.tool === "show-anchor") {
    if (event.button !== 0 && event.button !== 1) {
      return;
    }

    if (shouldPan(event)) {
      state.panning = true;
      state.panStartX = event.clientX - state.display.panX;
      state.panStartY = event.clientY - state.display.panY;
    }
    return;
  }

  if (event.button !== 0 && event.button !== 1) {
    return;
  }

  if (shouldPan(event)) {
    state.panning = true;
    state.panStartX = event.clientX - state.display.panX;
    state.panStartY = event.clientY - state.display.panY;
    return;
  }

  const pixel = screenToPixel(event.clientX, event.clientY);
  if (!isInside(pixel)) {
    return;
  }

  if (state.mode !== "draw") {
    return;
  }

  if (state.config.sizeRef) {
    state.selection = fixedBounds(pixel.x, pixel.y, state.config.sizeRef.width, state.config.sizeRef.height);
    updateSelectionMeta();
    renderAll();
    return;
  }

  state.drawing = true;
  state.drawStart = pixel;
  state.drawCurrent = pixel;
  renderAll();
}

function handleMouseMove(event) {
  if (state.panning) {
    state.display.panX = event.clientX - state.panStartX;
    state.display.panY = event.clientY - state.panStartY;
    renderAll();
    return;
  }

  const pixel = screenToPixel(event.clientX, event.clientY);
  if (!isInside(pixel)) {
    return;
  }

  if (state.config.tool === "show-anchor") {
    return;
  }

  if (state.config.sizeRef && state.mode === "draw") {
    state.drawCurrent = fixedBounds(pixel.x, pixel.y, state.config.sizeRef.width, state.config.sizeRef.height);
    renderAll();
    return;
  }

  if (!state.drawing || !state.drawStart) {
    return;
  }

  state.drawCurrent = pixel;
  renderAll();
}

function handleMouseUp() {
  state.panning = false;

  if (!state.drawing || !state.drawStart || !state.drawCurrent) {
    return;
  }

  state.selection = normalizeBounds(state.drawStart, state.drawCurrent);
  state.drawing = false;
  state.drawStart = null;
  state.drawCurrent = null;
  updateSelectionMeta();
  renderAll();
}

function handleKeyDown(event) {
  if (event.key === " ") {
    state.spacePan = true;
    event.preventDefault();
    return;
  }

  if (event.key === "Enter") {
    if (state.config.tool === "show-anchor" || state.config.tool === "show-rois") {
      completeView();
      return;
    }

    if (state.selection) {
      submitSelection();
    }
    return;
  }

  if (event.key === "Escape") {
    cancelSelection();
    return;
  }

  if (event.key.toLowerCase() === "f") {
    fitToView();
    return;
  }

  if (event.key === "1") {
    setOneToOne();
  }
}

function handleKeyUp(event) {
  if (event.key === " ") {
    state.spacePan = false;
  }
}

function shouldPan(event) {
  return event.button === 1 || event.shiftKey || state.mode === "pan" || state.spacePan;
}

function renderAll() {
  applyTransform(imageCanvas);
  applyTransform(overlayCanvas);
  zoomLabel.textContent = `${state.display.zoom.toFixed(2)}x`;
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  if (state.config.tool === "show-anchor" && state.config.anchor) {
    drawAnchorOverlay(state.config.anchor);
    acceptButton.disabled = false;
    return;
  }

  if (state.config.tool === "show-rois") {
    if (state.config.anchor) {
      drawAnchorOverlay(state.config.anchor);
    }
    drawSeededRois(state.config.normRois, "norm");
    drawSeededRois(state.config.barRois, "bar");
    acceptButton.disabled = false;
    return;
  }

  if (state.selection) {
    drawRect(state.selection, "rgba(105, 224, 208, 1)", "rgba(105, 224, 208, 0.18)", 2);
  }

  if (state.config.sizeRef && state.mode === "draw" && state.drawCurrent && !state.selection) {
    drawRect(state.drawCurrent, "rgba(255, 255, 255, 0.72)", "rgba(255, 255, 255, 0.08)", 1, [8, 6]);
  }

  if (!state.config.sizeRef && state.drawing && state.drawStart && state.drawCurrent) {
    drawRect(normalizeBounds(state.drawStart, state.drawCurrent), "rgba(255, 255, 255, 0.8)", "rgba(255, 255, 255, 0.06)", 1, [8, 6]);
  }

  acceptButton.disabled = !state.selection;
}

function applyTransform(canvas) {
  canvas.style.transform = `translate(${state.display.panX}px, ${state.display.panY}px) scale(${state.display.zoom})`;
}

function drawRect(rect, stroke, fill, lineWidth, dash = []) {
  overlayCtx.save();
  overlayCtx.strokeStyle = stroke;
  overlayCtx.fillStyle = fill;
  overlayCtx.lineWidth = lineWidth;
  overlayCtx.setLineDash(dash);
  overlayCtx.fillRect(rect.left, rect.top, rect.width, rect.height);
  overlayCtx.strokeRect(rect.left + 0.5, rect.top + 0.5, rect.width - 1, rect.height - 1);
  overlayCtx.restore();
}

function drawAnchorOverlay(anchor) {
  const roi = anchor.roi;
  const left = roi.left;
  const top = roi.top;
  const right = roi.right;
  const bottom = roi.bottom;
  const width = right - left;
  const height = bottom - top;

  drawRect(
    { left, top, right, bottom, width, height },
    "rgba(255, 111, 145, 0.95)",
    "rgba(255, 111, 145, 0.12)",
    2,
  );

  overlayCtx.save();
  overlayCtx.strokeStyle = "rgba(255, 111, 145, 0.95)";
  overlayCtx.lineWidth = 2;
  overlayCtx.beginPath();
  overlayCtx.moveTo(left, top);
  overlayCtx.lineTo(right, bottom);
  overlayCtx.moveTo(right, top);
  overlayCtx.lineTo(left, bottom);
  overlayCtx.stroke();

  overlayCtx.fillStyle = "rgba(105, 224, 208, 1)";
  overlayCtx.beginPath();
  overlayCtx.arc(anchor.centroid.x, anchor.centroid.y, 6, 0, Math.PI * 2);
  overlayCtx.fill();
  overlayCtx.restore();
}

function drawSeededRois(rois, kind) {
  if (!Array.isArray(rois)) {
    return;
  }

  for (const entry of rois) {
    const roi = entry?.roi;
    if (!roi) {
      continue;
    }

    const rect = {
      left: Number(roi.left),
      top: Number(roi.top),
      right: Number(roi.right),
      bottom: Number(roi.bottom),
      width: Number(roi.right) - Number(roi.left),
      height: Number(roi.bottom) - Number(roi.top),
    };

    const isNorm = kind === "norm";
    drawRect(
      rect,
      isNorm ? "rgba(255, 206, 84, 0.98)" : "rgba(105, 224, 208, 0.96)",
      isNorm ? "rgba(255, 206, 84, 0.16)" : "rgba(105, 224, 208, 0.10)",
      isNorm ? 2.5 : 1.75,
    );
    drawRoiLabel(rect, typeof entry.label === "string" ? entry.label : "ROI", isNorm);
  }
}

function drawRoiLabel(rect, label, isNorm) {
  overlayCtx.save();
  overlayCtx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  overlayCtx.textBaseline = "top";
  const textWidth = overlayCtx.measureText(label).width;
  const paddingX = 6;
  const labelHeight = 18;
  const labelWidth = textWidth + paddingX * 2;
  const left = clamp(rect.left, 0, Math.max(0, overlayCanvas.width - labelWidth));
  const top = rect.top > labelHeight + 4 ? rect.top - labelHeight - 4 : Math.min(rect.bottom + 4, Math.max(0, overlayCanvas.height - labelHeight));

  overlayCtx.fillStyle = isNorm ? "rgba(255, 206, 84, 0.92)" : "rgba(18, 28, 34, 0.88)";
  overlayCtx.strokeStyle = isNorm ? "rgba(255, 206, 84, 1)" : "rgba(105, 224, 208, 0.95)";
  overlayCtx.lineWidth = 1;
  overlayCtx.fillRect(left, top, labelWidth, labelHeight);
  overlayCtx.strokeRect(left + 0.5, top + 0.5, labelWidth - 1, labelHeight - 1);
  overlayCtx.fillStyle = isNorm ? "rgba(26, 18, 0, 0.95)" : "rgba(238, 248, 247, 0.98)";
  overlayCtx.fillText(label, left + paddingX, top + 3);
  overlayCtx.restore();
}

function screenToPixel(clientX, clientY) {
  const rect = canvasContainer.getBoundingClientRect();
  return {
    x: Math.floor((clientX - rect.left - state.display.panX) / state.display.zoom),
    y: Math.floor((clientY - rect.top - state.display.panY) / state.display.zoom),
  };
}

function isInside(pixel) {
  return pixel.x >= 0 && pixel.y >= 0 && pixel.x < state.config.cols && pixel.y < state.config.rows;
}

function normalizeBounds(a, b) {
  const left = Math.min(a.x, b.x);
  const top = Math.min(a.y, b.y);
  const right = Math.max(a.x, b.x);
  const bottom = Math.max(a.y, b.y);
  return {
    left,
    top,
    right,
    bottom,
    width: right - left,
    height: bottom - top,
  };
}

function fixedBounds(centerX, centerY, width, height) {
  const halfWidth = width / 2;
  const halfHeight = height / 2;
  const left = clamp(centerX - halfWidth, 0, state.config.cols - width);
  const top = clamp(centerY - halfHeight, 0, state.config.rows - height);
  const right = left + width;
  const bottom = top + height;
  return { left, top, right, bottom, width, height };
}

function updateSelectionMeta() {
  if (!state.selection) {
    selectionMeta.textContent = "No ROI selected.";
    return;
  }

  selectionMeta.innerHTML = [
    `left ${fmt(state.selection.left)}`,
    `top ${fmt(state.selection.top)}`,
    `width ${fmt(state.selection.width)}`,
    `height ${fmt(state.selection.height)}`,
  ].join("<br>");
}

function updateAnchorMeta() {
  if (!state.config.anchor) {
    selectionMeta.textContent = "No anchor data.";
    return;
  }

  const { roi, centroid } = state.config.anchor;
  selectionMeta.innerHTML = [
    `left ${fmt(roi.left)}`,
    `top ${fmt(roi.top)}`,
    `width ${fmt(roi.right - roi.left)}`,
    `height ${fmt(roi.bottom - roi.top)}`,
    `centroid x ${fmt(centroid.x)}`,
    `centroid y ${fmt(centroid.y)}`,
  ].join("<br>");
}

function updateRoisMeta() {
  const normCount = Array.isArray(state.config.normRois) ? state.config.normRois.length : 0;
  const barCount = Array.isArray(state.config.barRois) ? state.config.barRois.length : 0;
  const anchor = state.config.anchor;

  selectionMeta.innerHTML = [
    `${normCount} normalization ROI${normCount === 1 ? "" : "s"}`,
    `${barCount} bar ROI${barCount === 1 ? "" : "s"}`,
    anchor ? `anchor centroid x ${fmt(anchor.centroid.x)}` : null,
    anchor ? `anchor centroid y ${fmt(anchor.centroid.y)}` : null,
  ].filter(Boolean).join("<br>");
}

async function submitSelection() {
  if (state.config.tool === "show-anchor") {
    await completeView();
    return;
  }

  if (state.config.tool === "show-rois") {
    await completeView();
    return;
  }

  if (!state.selection) {
    return;
  }

  await window.pywebview.api.resolve({
    left: state.selection.left,
    top: state.selection.top,
    right: state.selection.right,
    bottom: state.selection.bottom,
  });
}

async function completeView() {
  await window.pywebview.api.resolve(null);
}

async function cancelSelection() {
  await window.pywebview.api.cancel();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function fmt(value) {
  return Number(value).toFixed(1);
}

init().catch(async (error) => {
  console.error(error);
  alert(error.message);
  if (typeof window.pywebview?.api?.cancel === "function") {
    await window.pywebview.api.cancel();
  }
});
