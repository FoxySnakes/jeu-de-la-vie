const COLOR_MODES = {
  neon: {
    glowColor: 'rgba(110, 249, 255, 0.55)',
    blur: 18,
    gradient: [
      [0, 24, 42, 89],
      [0.25, 70, 149, 209],
      [0.5, 46, 222, 174],
      [0.75, 254, 228, 92],
      [1, 255, 107, 181],
    ],
  },
  soft: {
    glowColor: 'rgba(164, 190, 255, 0.45)',
    blur: 12,
    gradient: [
      [0, 35, 40, 70],
      [0.2, 126, 154, 250],
      [0.5, 180, 220, 255],
      [0.8, 255, 206, 235],
      [1, 255, 255, 255],
    ],
  },
  mono: {
    glowColor: 'rgba(255, 255, 255, 0.32)',
    blur: 20,
    gradient: [
      [0, 20, 20, 20],
      [1, 255, 255, 255],
    ],
  },
};

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export class Renderer {
  constructor(canvas, grid) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: true });
    this.grid = grid;
    this.scale = 8;
    this.targetScale = this.scale;
    this.minScale = 2;
    this.maxScale = 80;
    this.dpr = window.devicePixelRatio || 1;
    this.originX = 0;
    this.originY = 0;
    this.lastColor = '';
    this.mode = 'neon';
    this.palette = new Array(256);
    this.needsPaletteUpdate = true;
    this.backgroundAlpha = 0.12;

    this.resize();
    this.centerOnGrid();
    this.updatePalette();
  }

  setMode(mode) {
    if (!COLOR_MODES[mode]) return;
    this.mode = mode;
    this.needsPaletteUpdate = true;
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;
    const needResize = this.canvas.width !== width * dpr || this.canvas.height !== height * dpr;
    if (needResize || this.dpr !== dpr) {
      this.canvas.width = width * dpr;
      this.canvas.height = height * dpr;
      this.dpr = dpr;
    }
  }

  centerOnGrid() {
    const viewWidth = this.canvas.clientWidth;
    const viewHeight = this.canvas.clientHeight;
    if (!viewWidth || !viewHeight) return;
    const cellsWide = viewWidth / this.scale;
    const cellsTall = viewHeight / this.scale;
    this.originX = Math.max(0, (this.grid.width - cellsWide) / 2);
    this.originY = Math.max(0, (this.grid.height - cellsTall) / 2);
  }

  setScaleFromSlider(value) {
    const normalized = (value - 1) / 79; // slider 1-80
    const minLog = Math.log2(this.minScale);
    const maxLog = Math.log2(this.maxScale);
    const target = Math.pow(2, lerp(minLog, maxLog, normalized));
    this.targetScale = clamp(target, this.minScale, this.maxScale);
  }

  getSliderValue() {
    const minLog = Math.log2(this.minScale);
    const maxLog = Math.log2(this.maxScale);
    const t = (Math.log2(this.targetScale) - minLog) / (maxLog - minLog);
    return Math.round(1 + t * 79);
  }

  zoomAt(clientX, clientY, delta) {
    const factor = delta > 0 ? 0.9 : 1.1;
    const newTarget = clamp(this.targetScale * factor, this.minScale, this.maxScale);
    const rect = this.canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    const beforeX = this.originX + px / this.scale;
    const beforeY = this.originY + py / this.scale;
    this.targetScale = newTarget;
    this.scale = this.targetScale; // réponse rapide à la molette
    const afterX = beforeX - px / this.scale;
    const afterY = beforeY - py / this.scale;
    this.originX = afterX;
    this.originY = afterY;
    this.clampOrigin();
  }

  pan(deltaX, deltaY) {
    this.originX -= deltaX / this.scale;
    this.originY -= deltaY / this.scale;
    this.clampOrigin();
  }

  clampOrigin() {
    const maxX = Math.max(0, this.grid.width - this.canvas.clientWidth / this.scale);
    const maxY = Math.max(0, this.grid.height - this.canvas.clientHeight / this.scale);
    this.originX = clamp(this.originX, 0, maxX);
    this.originY = clamp(this.originY, 0, maxY);
  }

  screenToCell(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const x = this.originX + (clientX - rect.left) / this.scale;
    const y = this.originY + (clientY - rect.top) / this.scale;
    return { x: Math.floor(x), y: Math.floor(y) };
  }

  updatePalette() {
    if (!this.needsPaletteUpdate) return;
    const mode = COLOR_MODES[this.mode] || COLOR_MODES.neon;
    const stops = mode.gradient;
    this.lastColor = '';
    for (let age = 0; age < 256; age++) {
      if (age === 0) {
        this.palette[age] = 'rgba(0,0,0,0)';
        continue;
      }
      const t = Math.min(1, age / 40);
      let left = stops[0];
      let right = stops[stops.length - 1];
      for (let i = 0; i < stops.length - 1; i++) {
        if (t >= stops[i][0] && t <= stops[i + 1][0]) {
          left = stops[i];
          right = stops[i + 1];
          break;
        }
      }
      const range = right[0] - left[0] || 1;
      const tt = clamp((t - left[0]) / range, 0, 1);
      const r = Math.round(lerp(left[1], right[1], tt));
      const g = Math.round(lerp(left[2], right[2], tt));
      const b = Math.round(lerp(left[3], right[3], tt));
      const alpha = clamp(0.35 + (age / 255) * 0.65, 0.35, 1);
      this.palette[age] = `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
    }
    this.needsPaletteUpdate = false;
  }

  draw() {
    this.updatePalette();
    const ctx = this.ctx;
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;

    // interpolation douce de l'échelle
    const diff = this.targetScale - this.scale;
    if (Math.abs(diff) > 0.01) {
      this.scale = lerp(this.scale, this.targetScale, 0.18);
      this.clampOrigin();
    }

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.globalAlpha = 1;
    ctx.fillStyle = `rgba(8, 10, 24, ${this.backgroundAlpha})`;
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.translate(-this.originX * this.scale, -this.originY * this.scale);
    ctx.scale(this.scale, this.scale);
    ctx.shadowColor = COLOR_MODES[this.mode]?.glowColor || COLOR_MODES.neon.glowColor;
    ctx.shadowBlur = COLOR_MODES[this.mode]?.blur || 14;
    ctx.imageSmoothingEnabled = false;

    const { grid } = this;
    const startX = Math.max(0, Math.floor(this.originX) - 1);
    const startY = Math.max(0, Math.floor(this.originY) - 1);
    const endX = Math.min(grid.width, Math.ceil(this.originX + width / this.scale) + 1);
    const endY = Math.min(grid.height, Math.ceil(this.originY + height / this.scale) + 1);
    const stride = grid.stride;
    let idx = (startY + 1) * stride + (startX + 1);
    for (let y = startY; y < endY; y++) {
      let rowIdx = idx;
      for (let x = startX; x < endX; x++, rowIdx++) {
        const age = grid.current[rowIdx];
        if (age === 0) continue;
        const color = this.palette[age];
        if (color !== this.lastColor) {
          ctx.fillStyle = color;
          this.lastColor = color;
        }
        ctx.fillRect(x, y, 1, 1);
      }
      idx += stride;
    }

    ctx.restore();
  }
}
