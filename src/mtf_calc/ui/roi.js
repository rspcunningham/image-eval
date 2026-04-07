import { screenToPixel, setCanvasTransform } from "./render.js";
import { clamp } from "./utils.js";

export function createROIController({
  canvas,
  canvasContainer,
  display,
  getConfig,
  getSequence,
  getActiveKey,
  setActiveKey,
  onCommitRect,
  onClearRect,
  adjustDraftRect,
}) {
  const ctx = canvas.getContext("2d");
  let drawing = false;
  let drawStart = null;
  let drawEnd = null;

  function roiColor(slot) {
    if (slot.tone === "black") {
      return "rgba(255,159,149,";
    }
    if (slot.tone === "white") {
      return "rgba(242,242,242,";
    }
    return slot.axis === "X" ? "rgba(61,216,197," : "rgba(107,138,253,";
  }

  function normalizeRect(a, b) {
    const row = Math.min(a.y, b.y);
    const col = Math.min(a.x, b.x);
    const height = Math.abs(b.y - a.y);
    const width = Math.abs(b.x - a.x);
    return { row, col, height, width };
  }

  function render() {
    const config = getConfig();
    if (!config) {
      return;
    }

    canvas.width = config.cols;
    canvas.height = config.rows;
    ctx.clearRect(0, 0, config.cols, config.rows);

    const sequence = getSequence();
    const activeKey = getActiveKey();

    for (const slot of sequence) {
      if (!slot.rect) {
        continue;
      }
      const { row, col, height, width } = slot.rect;
      const colorBase = roiColor(slot);
      ctx.strokeStyle = colorBase + "1)";
      ctx.lineWidth = slot.key === activeKey ? 3 : 2;
      ctx.strokeRect(col + 0.5, row + 0.5, width - 1, height - 1);
      ctx.fillStyle = colorBase + (slot.key === activeKey ? "0.12)" : "0.08)");
      ctx.fillRect(col, row, width, height);
    }

    if (drawing && drawStart && drawEnd) {
      const activeSlot = sequence.find((slot) => slot.key === activeKey) || null;
      const rect = activeSlot ? buildRectForSlot(activeSlot, normalizeRect(drawStart, drawEnd), sequence) : normalizeRect(drawStart, drawEnd);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(rect.col + 0.5, rect.row + 0.5, rect.width - 1, rect.height - 1);
      ctx.setLineDash([]);
    }

    syncTransform();
  }

  function syncTransform() {
    setCanvasTransform(canvas, display);
  }

  function buildRectForSlot(slot, rect, sequence) {
    if (!adjustDraftRect) {
      return rect;
    }
    return adjustDraftRect(slot, rect, sequence) || rect;
  }

  function findNextIncomplete(fromKey) {
    const sequence = getSequence();
    const startIndex = Math.max(0, sequence.findIndex((slot) => slot.key === fromKey));
    for (let index = startIndex; index < sequence.length; index += 1) {
      if (!sequence[index].rect) {
        return sequence[index].key;
      }
    }
    return fromKey;
  }

  function handleMouseDown(event) {
    const config = getConfig();
    const activeKey = getActiveKey();
    if (!config || !activeKey || event.button !== 0 || event.shiftKey) {
      return;
    }

    const pixel = screenToPixel(canvasContainer, display, event.clientX, event.clientY);
    if (pixel.x < 0 || pixel.x >= config.cols || pixel.y < 0 || pixel.y >= config.rows) {
      return;
    }

    drawing = true;
    drawStart = pixel;
    drawEnd = pixel;
    event.preventDefault();
    event.stopPropagation();
  }

  function handleMouseMove(event) {
    const config = getConfig();
    if (!drawing || !config) {
      return;
    }

    const pixel = screenToPixel(canvasContainer, display, event.clientX, event.clientY);
    drawEnd = {
      x: clamp(pixel.x, 0, config.cols),
      y: clamp(pixel.y, 0, config.rows),
    };
    render();
  }

  async function handleMouseUp() {
    if (!drawing) {
      return;
    }

    drawing = false;
    const activeKey = getActiveKey();
    const sequence = getSequence();
    const activeSlot = sequence.find((slot) => slot.key === activeKey) || null;
    const rect = normalizeRect(drawStart, drawEnd);
    drawStart = null;
    drawEnd = null;

    const nextRect = activeSlot ? buildRectForSlot(activeSlot, rect, sequence) : rect;

    if (!activeKey || nextRect.width < 2 || nextRect.height < 2) {
      render();
      return;
    }

    await onCommitRect(activeKey, nextRect);
    setActiveKey(findNextIncomplete(activeKey));
    render();
  }

  canvasContainer.addEventListener("mousedown", handleMouseDown);
  window.addEventListener("mousemove", handleMouseMove);
  window.addEventListener("mouseup", handleMouseUp);

  return {
    render,
    syncTransform,
    clearActive: async () => {
      const activeKey = getActiveKey();
      if (!activeKey) {
        return;
      }
      await onClearRect(activeKey);
      render();
    },
  };
}
