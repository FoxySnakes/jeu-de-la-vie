const GRID_WIDTH = 512;
const GRID_HEIGHT = 512;
const DEFAULT_SPEED = 30;
const DEFAULT_CELL_SIZE = 10;
const POPULATION_UPDATE_INTERVAL = 10;

const PATTERN_LIBRARY = new Map([
  [
    'glider',
    {
      name: 'Planeur',
      width: 3,
      height: 3,
      cells: [
        [1, 0],
        [2, 1],
        [0, 2],
        [1, 2],
        [2, 2],
      ],
    },
  ],
  [
    'blinker',
    {
      name: 'Clignotant',
      width: 3,
      height: 1,
      cells: [
        [0, 0],
        [1, 0],
        [2, 0],
      ],
    },
  ],
  [
    'toad',
    {
      name: 'Grenouille',
      width: 4,
      height: 2,
      cells: [
        [1, 0],
        [2, 0],
        [3, 0],
        [0, 1],
        [1, 1],
        [2, 1],
      ],
    },
  ],
  [
    'rpentomino',
    {
      name: 'R-pentomino',
      width: 3,
      height: 3,
      cells: [
        [1, 0],
        [2, 0],
        [0, 1],
        [1, 1],
        [1, 2],
      ],
    },
  ],
  [
    'gosper',
    {
      name: 'Canon de Gosper',
      width: 36,
      height: 9,
      cells: [
        [24, 0],
        [22, 1],
        [24, 1],
        [12, 2],
        [13, 2],
        [20, 2],
        [21, 2],
        [34, 2],
        [35, 2],
        [11, 3],
        [15, 3],
        [20, 3],
        [21, 3],
        [34, 3],
        [35, 3],
        [0, 4],
        [1, 4],
        [10, 4],
        [16, 4],
        [20, 4],
        [21, 4],
        [0, 5],
        [1, 5],
        [10, 5],
        [14, 5],
        [16, 5],
        [17, 5],
        [22, 5],
        [24, 5],
        [10, 6],
        [16, 6],
        [24, 6],
        [11, 7],
        [15, 7],
        [12, 8],
        [13, 8],
      ],
    },
  ],
]);

const THEME_PRESETS = {
  neon: {
    shader: {
      background: [0.039, 0.059, 0.145],
      gridLine: [0.075, 0.118, 0.27],
      aliveCore: [0.86, 0.47, 1.0],
      aliveEdge: [0.34, 0.23, 0.94],
      aliveHalo: [0.18, 0.42, 0.96],
    },
    css: {
      '--bg-color': 'radial-gradient(circle at 20% 20%, rgba(32, 42, 89, 0.85), rgba(6, 8, 20, 0.95))',
      '--glass-bg': 'rgba(20, 25, 45, 0.35)',
      '--glass-border': 'rgba(255, 255, 255, 0.12)',
      '--accent': '#6ef9ff',
      '--accent-strong': '#ff6bb5',
      '--text': 'rgba(255, 255, 255, 0.88)',
      '--text-muted': 'rgba(255, 255, 255, 0.65)',
      '--shadow': '0 30px 60px rgba(0, 0, 0, 0.45)',
    },
    canvasBg: '#0a0f25',
  },
  soft: {
    shader: {
      background: [0.065, 0.078, 0.11],
      gridLine: [0.145, 0.168, 0.24],
      aliveCore: [0.98, 0.78, 0.55],
      aliveEdge: [0.66, 0.53, 0.87],
      aliveHalo: [0.43, 0.62, 0.89],
    },
    css: {
      '--bg-color': 'radial-gradient(circle at 30% 20%, rgba(48, 58, 92, 0.7), rgba(16, 19, 33, 0.92))',
      '--glass-bg': 'rgba(34, 40, 62, 0.45)',
      '--glass-border': 'rgba(255, 255, 255, 0.1)',
      '--accent': '#ffbb89',
      '--accent-strong': '#d57dff',
      '--text': 'rgba(255, 244, 236, 0.9)',
      '--text-muted': 'rgba(255, 244, 236, 0.65)',
      '--shadow': '0 24px 50px rgba(14, 18, 34, 0.55)',
    },
    canvasBg: '#101424',
  },
  mono: {
    shader: {
      background: [0.04, 0.04, 0.055],
      gridLine: [0.12, 0.12, 0.15],
      aliveCore: [0.82, 0.82, 0.82],
      aliveEdge: [0.58, 0.58, 0.6],
      aliveHalo: [0.26, 0.26, 0.3],
    },
    css: {
      '--bg-color': 'radial-gradient(circle at 50% 25%, rgba(64, 64, 76, 0.65), rgba(12, 12, 16, 0.95))',
      '--glass-bg': 'rgba(28, 28, 35, 0.55)',
      '--glass-border': 'rgba(255, 255, 255, 0.08)',
      '--accent': '#9aa5b1',
      '--accent-strong': '#cfd8e3',
      '--text': 'rgba(240, 240, 240, 0.9)',
      '--text-muted': 'rgba(240, 240, 240, 0.6)',
      '--shadow': '0 24px 48px rgba(0, 0, 0, 0.6)',
    },
    canvasBg: '#0b0b12',
  },
};

let styleMode = 'neon';

const canvas = document.getElementById('world');
if (!canvas) {
  throw new Error('Canvas element #world introuvable.');
}

canvas.style.imageRendering = 'pixelated';

const gl = canvas.getContext('webgl2', {
  antialias: false,
  depth: false,
  stencil: false,
  premultipliedAlpha: false,
  preserveDrawingBuffer: false,
});

if (!gl) {
  throw new Error('WebGL2 est requis pour cette application.');
}

gl.disable(gl.DEPTH_TEST);
gl.disable(gl.STENCIL_TEST);
gl.disable(gl.BLEND);
gl.clearColor(0.039, 0.059, 0.145, 1.0);

const playPauseBtn = document.getElementById('play-pause');
const stepBtn = document.getElementById('step');
const resetBtn = document.getElementById('reset');
const randomBtn = document.getElementById('random');
const speedInput = document.getElementById('speed');
const generationLabel = document.getElementById('generation');
const populationLabel = document.getElementById('population');
const dimensionsLabel = document.getElementById('dimensions');
const settingsToggle = document.getElementById('settings-toggle');
const settingsPanel = document.getElementById('settings-panel');
const settingsForm = document.getElementById('settings-form');
const settingsClose = document.getElementById('settings-close');
const widthField = document.getElementById('grid-width');
const heightField = document.getElementById('grid-height');
const cellSizeField = document.getElementById('cell-size');
const randomDensitySlider = document.getElementById('random-density');
const patternSelect = document.getElementById('pattern');
const styleModeSelect = document.getElementById('style-mode');

const quadVao = createFullscreenQuad();

const vertexShaderSource = `#version 300 es
layout(location = 0) in vec2 aPosition;
void main() {
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

const simulationFragmentSource = `#version 300 es
precision highp float;

uniform sampler2D uState;
uniform ivec2 uGridSize;

layout(location = 0) out vec4 fragColor;

int aliveAt(ivec2 coord) {
  if (coord.x < 0 || coord.y < 0 || coord.x >= uGridSize.x || coord.y >= uGridSize.y) {
    return 0;
  }
  float value = texelFetch(uState, coord, 0).r;
  return int(value + 0.5);
}

void main() {
  ivec2 cell = ivec2(gl_FragCoord.xy) - ivec2(0);

  int neighbors = 0;
  neighbors += aliveAt(cell + ivec2(-1, -1));
  neighbors += aliveAt(cell + ivec2(0, -1));
  neighbors += aliveAt(cell + ivec2(1, -1));
  neighbors += aliveAt(cell + ivec2(-1, 0));
  neighbors += aliveAt(cell + ivec2(1, 0));
  neighbors += aliveAt(cell + ivec2(-1, 1));
  neighbors += aliveAt(cell + ivec2(0, 1));
  neighbors += aliveAt(cell + ivec2(1, 1));

  int current = aliveAt(cell);
  int nextState = 0;
  if (current == 1) {
    nextState = (neighbors == 2 || neighbors == 3) ? 1 : 0;
  } else {
    nextState = (neighbors == 3) ? 1 : 0;
  }

  fragColor = vec4(float(nextState), 0.0, 0.0, 1.0);
}`;

const renderFragmentSource = `#version 300 es
precision highp float;

uniform sampler2D uState;
uniform ivec2 uGridSize;
uniform float uCellSize;
uniform float uPixelRatio;
uniform vec3 uBackgroundColor;
uniform vec3 uGridLineColor;
uniform vec3 uAliveCoreColor;
uniform vec3 uAliveEdgeColor;
uniform vec3 uAliveHaloColor;

layout(location = 0) out vec4 fragColor;

float gridMask(vec2 uv, float thickness) {
  vec2 dist = min(uv, 1.0 - uv);
  float edge = min(dist.x, dist.y);
  return smoothstep(thickness, thickness * 1.5, edge);
}

void main() {
  vec2 pixel = gl_FragCoord.xy / uPixelRatio;
  vec2 cellPos = pixel / max(uCellSize, 0.0001);
  ivec2 cell = ivec2(floor(cellPos));
  if (cell.x < 0 || cell.y < 0 || cell.x >= uGridSize.x || cell.y >= uGridSize.y) {
    fragColor = vec4(uBackgroundColor, 1.0);
    return;
  }

  ivec2 storageCoord = ivec2(cell);
  float state = texelFetch(uState, storageCoord, 0).r;
  vec2 uv = fract(cellPos);
  float thickness = clamp(1.0 / max(uCellSize, 1.0), 0.01, 0.12);
  float grid = gridMask(uv, thickness);
  vec3 color = mix(uGridLineColor, uBackgroundColor, grid);

  if (state > 0.5) {
    vec2 centered = uv - 0.5;
    float dist = length(centered);
    float glowCore = smoothstep(0.45, 0.0, dist);
    float glowHalo = smoothstep(0.8, 0.0, dist);
    vec3 neon = mix(uAliveEdgeColor, uAliveCoreColor, glowCore);
    color = mix(color, neon, glowCore);
    color += uAliveHaloColor * pow(glowHalo, 2.0);
  }

  fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}`;

const simulationProgram = createProgram(vertexShaderSource, simulationFragmentSource);
const renderProgram = createProgram(vertexShaderSource, renderFragmentSource);

const simUniforms = {
  uState: gl.getUniformLocation(simulationProgram, 'uState'),
  uGridSize: gl.getUniformLocation(simulationProgram, 'uGridSize'),
};

const renderUniforms = {
  uState: gl.getUniformLocation(renderProgram, 'uState'),
  uGridSize: gl.getUniformLocation(renderProgram, 'uGridSize'),
  uCellSize: gl.getUniformLocation(renderProgram, 'uCellSize'),
  uPixelRatio: gl.getUniformLocation(renderProgram, 'uPixelRatio'),
  uBackgroundColor: gl.getUniformLocation(renderProgram, 'uBackgroundColor'),
  uGridLineColor: gl.getUniformLocation(renderProgram, 'uGridLineColor'),
  uAliveCoreColor: gl.getUniformLocation(renderProgram, 'uAliveCoreColor'),
  uAliveEdgeColor: gl.getUniformLocation(renderProgram, 'uAliveEdgeColor'),
  uAliveHaloColor: gl.getUniformLocation(renderProgram, 'uAliveHaloColor'),
};

let gridWidth = GRID_WIDTH;
let gridHeight = GRID_HEIGHT;
let speed = DEFAULT_SPEED;
let cellSize = DEFAULT_CELL_SIZE;
let running = false;
let generation = 0;
let population = 0;
let frameCounter = 0;
let accumulator = 0;
let lastFrameTime = performance.now();

let displayWidth = GRID_WIDTH * DEFAULT_CELL_SIZE;
let displayHeight = GRID_HEIGHT * DEFAULT_CELL_SIZE;

let stateTextures = [];
let stateFramebuffers = [];
let populationBuffer = new Uint8Array(gridWidth * gridHeight);
let currentIndex = 0;
let nextIndex = 1;

initializeStateResources(gridWidth, gridHeight);
updateCanvasSize();
updateHud();

if (dimensionsLabel) {
  dimensionsLabel.textContent = `${gridWidth} × ${gridHeight}`;
}

if (speedInput) {
  speedInput.value = String(DEFAULT_SPEED);
}

if (widthField) {
  widthField.value = String(gridWidth);
}

if (heightField) {
  heightField.value = String(gridHeight);
}

if (cellSizeField) {
  cellSizeField.value = String(cellSize);
}

if (randomDensitySlider) {
  randomDensitySlider.value = String(30);
}

if (settingsPanel) {
  settingsPanel.classList.add('hidden');
}

populatePatternSelect();
if (patternSelect) {
  patternSelect.value = '';
}

if (styleModeSelect && Object.prototype.hasOwnProperty.call(THEME_PRESETS, styleModeSelect.value)) {
  styleMode = styleModeSelect.value;
}
applyStyleMode(styleMode);

playPauseBtn?.addEventListener('click', () => {
  if (!running && population === 0) {
    return;
  }
  running = !running;
  if (running) {
    lastFrameTime = performance.now();
  } else {
    accumulator = 0;
  }
  updatePlayPauseVisual();
});

stepBtn?.addEventListener('click', () => {
  if (population === 0) {
    return;
  }
  if (running) {
    pauseSimulation();
  }
  performSimulationStep();
  renderFrame();
  updatePopulation(true);
});

resetBtn?.addEventListener('click', () => {
  pauseSimulation();
  clearState();
  generation = 0;
  population = 0;
  updateHud();
  updatePlayPauseVisual();
  renderFrame();
});

randomBtn?.addEventListener('click', () => {
  pauseSimulation();
  randomizeState();
  renderFrame();
  updatePopulation(true);
});

speedInput?.addEventListener('input', () => {
  const value = speedInput.valueAsNumber;
  if (Number.isFinite(value)) {
    speed = sanitizeSpeed(value, speed);
  }
});

speedInput?.addEventListener('change', () => {
  const sanitized = sanitizeSpeed(speedInput.valueAsNumber, speed);
  speed = sanitized;
  speedInput.value = String(sanitized);
});

widthField?.addEventListener('input', pauseSimulation);
heightField?.addEventListener('input', pauseSimulation);
cellSizeField?.addEventListener('input', pauseSimulation);
randomDensitySlider?.addEventListener('input', pauseSimulation);

settingsToggle?.addEventListener('click', () => {
  settingsPanel?.classList.toggle('hidden');
});

settingsClose?.addEventListener('click', () => {
  settingsPanel?.classList.add('hidden');
});

settingsForm?.addEventListener('submit', (event) => {
  event.preventDefault();
  pauseSimulation();
  const targetWidth = Math.max(16, Math.min(4096, Math.floor(Number(widthField?.value ?? gridWidth))));
  const targetHeight = Math.max(16, Math.min(4096, Math.floor(Number(heightField?.value ?? gridHeight))));
  const targetCellSize = Math.max(1, Math.min(128, Math.floor(Number(cellSizeField?.value ?? cellSize))));
  applyGridSettings(targetWidth, targetHeight, targetCellSize);
  if (speedInput) {
    const sanitized = sanitizeSpeed(speedInput.valueAsNumber, speed);
    speed = sanitized;
    speedInput.value = String(sanitized);
  }
  if (randomDensitySlider) {
    const densityValue = Number(randomDensitySlider.value);
    if (!Number.isFinite(densityValue)) {
      randomDensitySlider.value = '30';
    } else {
      const clamped = Math.max(0, Math.min(100, Math.floor(densityValue)));
      randomDensitySlider.value = String(clamped);
    }
  }
  settingsPanel?.classList.add('hidden');
});

patternSelect?.addEventListener('change', () => {
  pauseSimulation();
  const key = patternSelect.value;
  if (!key) {
    clearState();
    generation = 0;
    population = 0;
    updateHud();
    updatePlayPauseVisual();
    renderFrame();
    updatePopulation(true);
    return;
  }
  applyPattern(key);
});

styleModeSelect?.addEventListener('change', () => {
  const selected = styleModeSelect.value;
  applyStyleMode(selected);
  renderFrame();
});

canvas.addEventListener('contextmenu', (event) => event.preventDefault());
canvas.style.touchAction = 'none';

let pointerActive = false;
let pointerWriteValue = 1;

canvas.addEventListener('pointerdown', (event) => {
  if (event.button !== 0 && event.pointerType !== 'touch') {
    return;
  }
  const coords = getCellFromEvent(event);
  if (!coords) {
    return;
  }
  event.preventDefault();
  canvas.setPointerCapture(event.pointerId);
  pointerActive = true;
  const currentState = readCell(coords.x, coords.y);
  pointerWriteValue = currentState ? 0 : 1;
  writeCell(coords.x, coords.y, pointerWriteValue);
  renderFrame();
  updatePopulation(true);
});

canvas.addEventListener('pointermove', (event) => {
  if (!pointerActive) {
    return;
  }
  event.preventDefault();
  const coords = getCellFromEvent(event);
  if (!coords) {
    return;
  }
  writeCell(coords.x, coords.y, pointerWriteValue);
  renderFrame();
});

canvas.addEventListener('pointerup', (event) => {
  if (!pointerActive) {
    return;
  }
  if (event.pointerType !== 'touch' && event.button !== 0) {
    return;
  }
  pointerActive = false;
  canvas.releasePointerCapture(event.pointerId);
  updatePopulation(true);
});

canvas.addEventListener('pointercancel', (event) => {
  if (!pointerActive) {
    return;
  }
  pointerActive = false;
  canvas.releasePointerCapture(event.pointerId);
  updatePopulation(true);
});

window.addEventListener('resize', () => {
  updateCanvasSize();
});

updatePlayPauseVisual();
renderFrame();
requestAnimationFrame(frameLoop);

function frameLoop(now) {
  const delta = (now - lastFrameTime) / 1000;
  lastFrameTime = now;

  if (running) {
    const stepDuration = 1 / speed;
    accumulator += Math.min(delta, 0.25);
    while (accumulator >= stepDuration) {
      performSimulationStep();
      accumulator -= stepDuration;
    }
  }

  renderFrame();

  frameCounter += 1;
  if (frameCounter >= POPULATION_UPDATE_INTERVAL) {
    updatePopulation(false);
    frameCounter = 0;
  }

  requestAnimationFrame(frameLoop);
}

function performSimulationStep() {
  gl.useProgram(simulationProgram);
  gl.bindVertexArray(quadVao);
  gl.viewport(0, 0, gridWidth, gridHeight);

  gl.bindFramebuffer(gl.FRAMEBUFFER, stateFramebuffers[nextIndex]);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, stateTextures[currentIndex]);
  gl.uniform1i(simUniforms.uState, 0);
  gl.uniform2i(simUniforms.uGridSize, gridWidth, gridHeight);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  const temp = currentIndex;
  currentIndex = nextIndex;
  nextIndex = temp;
  generation += 1;
  updateHud();
}

function renderFrame() {
  gl.useProgram(renderProgram);
  gl.bindVertexArray(quadVao);

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, stateTextures[currentIndex]);
  gl.uniform1i(renderUniforms.uState, 0);
  gl.uniform2i(renderUniforms.uGridSize, gridWidth, gridHeight);
  gl.uniform1f(renderUniforms.uCellSize, cellSize);
  gl.uniform1f(renderUniforms.uPixelRatio, window.devicePixelRatio || 1);
  const theme = THEME_PRESETS[styleMode] ?? THEME_PRESETS.neon;
  const shader = theme.shader;
  gl.uniform3f(
    renderUniforms.uBackgroundColor,
    shader.background[0],
    shader.background[1],
    shader.background[2],
  );
  gl.uniform3f(
    renderUniforms.uGridLineColor,
    shader.gridLine[0],
    shader.gridLine[1],
    shader.gridLine[2],
  );
  gl.uniform3f(
    renderUniforms.uAliveCoreColor,
    shader.aliveCore[0],
    shader.aliveCore[1],
    shader.aliveCore[2],
  );
  gl.uniform3f(
    renderUniforms.uAliveEdgeColor,
    shader.aliveEdge[0],
    shader.aliveEdge[1],
    shader.aliveEdge[2],
  );
  gl.uniform3f(
    renderUniforms.uAliveHaloColor,
    shader.aliveHalo[0],
    shader.aliveHalo[1],
    shader.aliveHalo[2],
  );

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function pauseSimulation() {
  if (!running) {
    return;
  }
  running = false;
  accumulator = 0;
  updatePlayPauseVisual();
}

function updateHud() {
  if (generationLabel) {
    generationLabel.textContent = String(generation);
  }
  if (populationLabel) {
    populationLabel.textContent = String(population);
  }
  if (dimensionsLabel) {
    dimensionsLabel.textContent = `${gridWidth} × ${gridHeight}`;
  }
  refreshControlAvailability();
}

function refreshControlAvailability() {
  if (playPauseBtn) {
    const disabled = !running && population === 0;
    playPauseBtn.disabled = disabled;
  }
  if (stepBtn) {
    stepBtn.disabled = population === 0;
  }
}

function updatePopulation(force = false) {
  if (force) {
    frameCounter = 0;
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, stateFramebuffers[currentIndex]);
  gl.readBuffer(gl.COLOR_ATTACHMENT0);
  if (populationBuffer.length !== gridWidth * gridHeight) {
    populationBuffer = new Uint8Array(gridWidth * gridHeight);
  }
  gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
  gl.readPixels(0, 0, gridWidth, gridHeight, gl.RED, gl.UNSIGNED_BYTE, populationBuffer);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  let sum = 0;
  for (let i = 0; i < populationBuffer.length; i += 1) {
    sum += populationBuffer[i] > 127 ? 1 : 0;
  }
  population = sum;
  if (population === 0 && running) {
    pauseSimulation();
  }
  updateHud();
  updatePlayPauseVisual();
}

function clearState() {
  const empty = new Uint8Array(gridWidth * gridHeight);
  uploadState(empty);
}

function randomizeState() {
  const densitySlider = randomDensitySlider ? Number(randomDensitySlider.value) : NaN;
  let density;
  if (Number.isFinite(densitySlider) && densitySlider >= 0 && densitySlider <= 100) {
    density = densitySlider / 100;
  } else {
    density = 0.3;
  }
  const data = new Uint8Array(gridWidth * gridHeight);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = Math.random() < density ? 255 : 0;
  }
  uploadState(data);
  generation = 0;
  updateHud();
}

function populatePatternSelect() {
  if (!patternSelect) {
    return;
  }
  while (patternSelect.options.length > 1) {
    patternSelect.remove(1);
  }
  for (const [key, pattern] of PATTERN_LIBRARY) {
    const option = document.createElement('option');
    option.value = key;
    option.textContent = pattern.name;
    patternSelect.appendChild(option);
  }
}

function applyPattern(key) {
  const pattern = PATTERN_LIBRARY.get(key);
  if (!pattern) {
    return;
  }
  const data = new Uint8Array(gridWidth * gridHeight);
  const offsetX = Math.floor((gridWidth - pattern.width) / 2);
  const offsetY = Math.floor((gridHeight - pattern.height) / 2);
  for (const [cellX, cellY] of pattern.cells) {
    const targetX = offsetX + cellX;
    const targetY = offsetY + cellY;
    if (targetX < 0 || targetY < 0 || targetX >= gridWidth || targetY >= gridHeight) {
      continue;
    }
    data[targetY * gridWidth + targetX] = 255;
  }
  uploadState(data);
  generation = 0;
  updateHud();
  renderFrame();
  updatePopulation(true);
}

function applyStyleMode(modeKey) {
  const hasPreset = Object.prototype.hasOwnProperty.call(THEME_PRESETS, modeKey);
  styleMode = hasPreset ? modeKey : 'neon';
  const preset = THEME_PRESETS[styleMode];
  const root = document.documentElement;
  for (const [property, value] of Object.entries(preset.css)) {
    root.style.setProperty(property, value);
  }
  canvas.style.backgroundColor = preset.canvasBg;
  gl.clearColor(preset.shader.background[0], preset.shader.background[1], preset.shader.background[2], 1.0);
  if (styleModeSelect) {
    styleModeSelect.value = styleMode;
  }
}

function sanitizeSpeed(value, fallback) {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  const floored = Math.floor(value);
  return Math.max(1, Math.min(240, floored));
}

function uploadState(data) {
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.activeTexture(gl.TEXTURE0);
  for (let i = 0; i < stateTextures.length; i += 1) {
    gl.bindTexture(gl.TEXTURE_2D, stateTextures[i]);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gridWidth, gridHeight, gl.RED, gl.UNSIGNED_BYTE, data);
  }
  gl.bindTexture(gl.TEXTURE_2D, stateTextures[currentIndex]);
}

function createStateResources(width, height) {
  const textures = [];
  const framebuffers = [];
  for (let i = 0; i < 2; i += 1) {
    const texture = gl.createTexture();
    if (!texture) {
      throw new Error('Impossible de créer la texture de simulation.');
    }
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, null);

    const framebuffer = gl.createFramebuffer();
    if (!framebuffer) {
      throw new Error('Impossible de créer le framebuffer de simulation.');
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('Framebuffer incomplet pour la simulation.');
    }

    textures.push(texture);
    framebuffers.push(framebuffer);
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return { textures, framebuffers };
}

function initializeStateResources(width, height) {
  if (stateTextures.length) {
    for (const tex of stateTextures) {
      gl.deleteTexture(tex);
    }
  }
  if (stateFramebuffers.length) {
    for (const fb of stateFramebuffers) {
      gl.deleteFramebuffer(fb);
    }
  }

  const resources = createStateResources(width, height);
  stateTextures = resources.textures;
  stateFramebuffers = resources.framebuffers;
  currentIndex = 0;
  nextIndex = 1;
  const empty = new Uint8Array(width * height);
  uploadState(empty);
}

function applyGridSettings(newWidth, newHeight, newCellSize) {
  const targetWidth = Math.max(16, Math.min(4096, newWidth));
  const targetHeight = Math.max(16, Math.min(4096, newHeight));
  const targetCellSize = Math.max(1, Math.min(128, newCellSize));

  const widthChanged = targetWidth !== gridWidth;
  const heightChanged = targetHeight !== gridHeight;
  const cellSizeChanged = targetCellSize !== cellSize;

  if (!widthChanged && !heightChanged && !cellSizeChanged) {
    return;
  }

  if (widthChanged || heightChanged) {
    gridWidth = targetWidth;
    gridHeight = targetHeight;
    populationBuffer = new Uint8Array(gridWidth * gridHeight);
    initializeStateResources(gridWidth, gridHeight);

    gl.useProgram(simulationProgram);
    gl.uniform2i(simUniforms.uGridSize, gridWidth, gridHeight);
    gl.useProgram(renderProgram);
    gl.uniform2i(renderUniforms.uGridSize, gridWidth, gridHeight);

    generation = 0;
    population = 0;
    accumulator = 0;
  }

  if (cellSizeChanged) {
    cellSize = targetCellSize;
  }

  updateCanvasSize();
  updateHud();

  if (widthField) {
    widthField.value = String(gridWidth);
  }
  if (heightField) {
    heightField.value = String(gridHeight);
  }
  if (cellSizeField) {
    cellSizeField.value = String(cellSize);
  }

  updatePlayPauseVisual();
  renderFrame();
  updatePopulation(true);
}

function updateCanvasSize() {
  const ratio = window.devicePixelRatio || 1;
  displayWidth = Math.max(1, Math.floor(gridWidth * cellSize));
  displayHeight = Math.max(1, Math.floor(gridHeight * cellSize));
  canvas.style.width = `${displayWidth}px`;
  canvas.style.height = `${displayHeight}px`;
  canvas.width = Math.max(1, Math.floor(displayWidth * ratio));
  canvas.height = Math.max(1, Math.floor(displayHeight * ratio));
  gl.viewport(0, 0, canvas.width, canvas.height);

  gl.useProgram(renderProgram);
  gl.uniform1f(renderUniforms.uCellSize, cellSize);
  gl.uniform1f(renderUniforms.uPixelRatio, ratio);
  gl.uniform2i(renderUniforms.uGridSize, gridWidth, gridHeight);

  gl.useProgram(simulationProgram);
  gl.uniform2i(simUniforms.uGridSize, gridWidth, gridHeight);
}

function updatePlayPauseVisual() {
  if (!playPauseBtn) {
    return;
  }
  playPauseBtn.textContent = running ? '⏸️' : '▶️';
  playPauseBtn.setAttribute('aria-pressed', running ? 'true' : 'false');
  if (running) {
    playPauseBtn.title = 'Mettre en pause';
    playPauseBtn.setAttribute('aria-label', 'Mettre en pause la simulation');
  } else if (population === 0) {
    playPauseBtn.title = 'Ajoutez des cellules pour démarrer';
    playPauseBtn.setAttribute('aria-label', 'Ajoutez des cellules pour démarrer la simulation');
  } else {
    playPauseBtn.title = 'Démarrer la simulation';
    playPauseBtn.setAttribute('aria-label', 'Démarrer la simulation');
  }
  refreshControlAvailability();
}

function getCellFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const relX = event.clientX - rect.left;
  const relY = event.clientY - rect.top;
  if (relX < 0 || relY < 0 || relX >= rect.width || relY >= rect.height) {
    return null;
  }
  const normalizedX = relX / rect.width;
  const normalizedY = relY / rect.height;
  const col = Math.min(gridWidth - 1, Math.max(0, Math.floor(normalizedX * gridWidth)));
  const row = Math.min(
    gridHeight - 1,
    Math.max(0, Math.floor((1 - normalizedY) * gridHeight)),
  );
  if (col < 0 || row < 0 || col >= gridWidth || row >= gridHeight) {
    return null;
  }
  return { x: col, y: row };
}

function readCell(x, y) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, stateFramebuffers[currentIndex]);
  gl.readBuffer(gl.COLOR_ATTACHMENT0);
  const pixel = new Uint8Array(1);
  gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
  gl.readPixels(x, y, 1, 1, gl.RED, gl.UNSIGNED_BYTE, pixel);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return pixel[0] > 127;
}

function writeCell(x, y, value) {
  const pixelValue = value ? 255 : 0;
  const data = new Uint8Array([pixelValue]);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.activeTexture(gl.TEXTURE0);
  for (let i = 0; i < stateTextures.length; i += 1) {
    gl.bindTexture(gl.TEXTURE_2D, stateTextures[i]);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, x, y, 1, 1, gl.RED, gl.UNSIGNED_BYTE, data);
  }
}

function createFullscreenQuad() {
  const vao = gl.createVertexArray();
  const buffer = gl.createBuffer();
  if (!vao || !buffer) {
    throw new Error('Impossible de créer la géométrie du quad.');
  }
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  const vertices = new Float32Array([
    -1, -1,
    1, -1,
    -1, 1,
    1, 1,
  ]);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  return vao;
}

function createProgram(vertexSource, fragmentSource) {
  const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  if (!program) {
    throw new Error('Impossible de créer le programme WebGL.');
  }
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    gl.deleteProgram(program);
    throw new Error(`Échec du linkage du programme : ${info}`);
  }
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  return program;
}

function compileShader(type, source) {
  const shader = gl.createShader(type);
  if (!shader) {
    throw new Error('Impossible de créer un shader.');
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Erreur de compilation du shader : ${info}`);
  }
  return shader;
}
