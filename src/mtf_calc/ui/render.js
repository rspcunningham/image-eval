import { clamp } from "./utils.js";

export function percentileFromHistogram(counts, percentile) {
  if (!counts || counts.length === 0) {
    return 0;
  }

  const total = counts.reduce((sum, value) => sum + value, 0);
  const target = total * percentile;
  let cumulative = 0;

  for (let bin = 0; bin < counts.length; bin += 1) {
    cumulative += counts[bin];
    if (cumulative >= target) {
      return bin / (counts.length - 1);
    }
  }

  return 1;
}

export function renderHistogram({
  canvas,
  container,
  counts,
  display,
  dataMin,
  dataMax,
  labels,
  info,
  fmtNum,
}) {
  if (!counts) {
    return;
  }

  const rect = container.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  let maxLog = 0;
  for (let index = 0; index < counts.length; index += 1) {
    maxLog = Math.max(maxLog, Math.log1p(counts[index]));
  }

  const graphTop = 24;
  const graphHeight = rect.height - graphTop - 18;
  const graphWidth = rect.width;
  const winLo = clamp(display.level - display.window / 2, 0, 1);
  const winHi = clamp(display.level + display.window / 2, 0, 1);

  ctx.fillStyle = "rgba(61, 216, 197, 0.12)";
  ctx.fillRect(winLo * graphWidth, graphTop, (winHi - winLo) * graphWidth, graphHeight);

  for (let index = 0; index < counts.length; index += 1) {
    const logValue = Math.log1p(counts[index]);
    const height = maxLog > 0 ? (logValue / maxLog) * graphHeight : 0;
    const x = (index / counts.length) * graphWidth;
    const width = graphWidth / counts.length + 1;
    const binValue = index / (counts.length - 1);

    ctx.fillStyle = binValue >= winLo && binValue <= winHi
      ? "rgba(61, 216, 197, 0.86)"
      : "rgba(255, 255, 255, 0.22)";
    ctx.fillRect(x, rect.height - 14 - height, width, height);
  }

  const centerX = display.level * graphWidth;
  const leftX = winLo * graphWidth;
  const rightX = winHi * graphWidth;

  ctx.strokeStyle = "rgba(61, 216, 197, 1)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(centerX, graphTop);
  ctx.lineTo(centerX, rect.height - 14);
  ctx.stroke();

  ctx.strokeStyle = "rgba(61, 216, 197, 0.7)";
  ctx.beginPath();
  ctx.moveTo(leftX, graphTop);
  ctx.lineTo(leftX, rect.height - 14);
  ctx.moveTo(rightX, graphTop);
  ctx.lineTo(rightX, rect.height - 14);
  ctx.stroke();

  labels.min.textContent = fmtNum(dataMin);
  labels.max.textContent = fmtNum(dataMax);
  info.innerHTML = `window ${fmtNum(display.window)}<br>level ${fmtNum(display.level)}`;
}

export function renderSourceImage({ canvas, context, data, rows, cols, display }) {
  if (!data) {
    return;
  }

  const low = display.level - display.window / 2;
  const scale = display.window > 0 ? 1 / display.window : 1;
  const imageData = context.createImageData(cols, rows);
  const pixels = imageData.data;

  for (let index = 0; index < data.length; index += 1) {
    const normalized = clamp((data[index] - low) * scale, 0, 1);
    const byte = Math.round(normalized * 255);
    const pixelIndex = index * 4;
    pixels[pixelIndex] = byte;
    pixels[pixelIndex + 1] = byte;
    pixels[pixelIndex + 2] = byte;
    pixels[pixelIndex + 3] = 255;
  }

  canvas.width = cols;
  canvas.height = rows;
  context.putImageData(imageData, 0, 0);
}

export function setCanvasTransform(canvas, display) {
  canvas.style.transform = `translate(${display.panX}px, ${display.panY}px) scale(${display.zoom})`;
}

export function applyCanvasTransform({ canvas, display, zoomIndicator }) {
  setCanvasTransform(canvas, display);
  if (zoomIndicator) {
    zoomIndicator.textContent = `${display.zoom.toFixed(2)}x`;
  }
}

export function fitDisplay(container, rows, cols) {
  const width = container.clientWidth;
  const height = container.clientHeight;
  const zoom = clamp(Math.min(width / cols, height / rows) * 0.96, 0.05, 64);

  return {
    zoom,
    panX: (width - cols * zoom) / 2,
    panY: (height - rows * zoom) / 2,
  };
}

export function oneToOneDisplay(container, rows, cols) {
  return {
    zoom: 1,
    panX: (container.clientWidth - cols) / 2,
    panY: (container.clientHeight - rows) / 2,
  };
}

export function screenToPixel(container, display, clientX, clientY) {
  const rect = container.getBoundingClientRect();
  return {
    x: Math.floor((clientX - rect.left - display.panX) / display.zoom),
    y: Math.floor((clientY - rect.top - display.panY) / display.zoom),
  };
}
