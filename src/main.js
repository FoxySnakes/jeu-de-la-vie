const GRID_WIDTH = 512;
const GRID_HEIGHT = 512;
const DEFAULT_SPEED = 30;
const DEFAULT_ZOOM = 10;
const POPULATION_UPDATE_INTERVAL = 10;

const canvas = document.getElementById('world');
if (!canvas) {
  throw new Error('Canvas element #world introuvable.');
}

canvas.style.imageRendering = 'pixelated';
canvas.style.backgroundColor = '#0a0f25';

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
const speedSlider = document.getElementById('speed');
const zoomSlider = document.getElementById('zoom');
const generationLabel = document.getElementById('generation');
const populationLabel = document.getElementById('population');
const dimensionsLabel = document.getElementById('dimensions');
const settingsToggle = document.getElementById('settings-toggle');
const settingsPanel = document.getElementById('settings-panel');
const settingsForm = document.getElementById('settings-form');
const settingsClose = document.getElementById('settings-close');
const widthField = document.getElementById('grid-width');
const heightField = document.getElementById('grid-height');
const randomDensitySlider = document.getElementById('random-density');

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
uniform float uZoom;
uniform float uPixelRatio;

layout(location = 0) out vec4 fragColor;

const vec3 BACKGROUND = vec3(0.039, 0.059, 0.145);
const vec3 NEON_CORE = vec3(0.55, 0.32, 1.0);
const vec3 NEON_EDGE = vec3(0.28, 0.22, 0.80);

void main() {
  vec2 scaledCoord = gl_FragCoord.xy / (uZoom * uPixelRatio);
  ivec2 cell = ivec2(floor(scaledCoord));
  if (cell.x < 0 || cell.y < 0 || cell.x >= uGridSize.x || cell.y >= uGridSize.y) {
    fragColor = vec4(BACKGROUND, 1.0);
    return;
  }

  float state = texelFetch(uState, cell, 0).r;
  if (state < 0.5) {
    fragColor = vec4(BACKGROUND, 1.0);
    return;
  }

  vec2 local = fract(scaledCoord) - 0.5;
  float dist = length(local);
  float glow = clamp(1.0 - smoothstep(0.0, 0.5, dist), 0.0, 1.0);
  vec3 neon = mix(NEON_EDGE, NEON_CORE, glow);
  float halo = pow(glow, 2.2);
  vec3 color = BACKGROUND + neon * (0.35 + 0.65 * halo);
  color = clamp(color, 0.0, 1.0);
  float alpha = 0.65 + 0.35 * halo;
  fragColor = vec4(color, alpha);
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
  uZoom: gl.getUniformLocation(renderProgram, 'uZoom'),
  uPixelRatio: gl.getUniformLocation(renderProgram, 'uPixelRatio'),
};

let gridWidth = GRID_WIDTH;
let gridHeight = GRID_HEIGHT;
let zoom = DEFAULT_ZOOM;
let speed = DEFAULT_SPEED;
let running = false;
let generation = 0;
let population = 0;
let frameCounter = 0;
let accumulator = 0;
let lastFrameTime = performance.now();

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

if (speedSlider) {
  speedSlider.value = String(DEFAULT_SPEED);
}

if (zoomSlider) {
  zoomSlider.value = String(DEFAULT_ZOOM);
}

if (widthField) {
  widthField.value = String(gridWidth);
}

if (heightField) {
  heightField.value = String(gridHeight);
}

if (randomDensitySlider) {
  randomDensitySlider.value = String(30);
}

if (settingsPanel) {
  settingsPanel.classList.add('hidden');
}

playPauseBtn?.addEventListener('click', () => {
  running = !running;
  if (running) {
    lastFrameTime = performance.now();
  }
  updatePlayPauseVisual();
});

stepBtn?.addEventListener('click', () => {
  if (running) {
    running = false;
    updatePlayPauseVisual();
  }
  performSimulationStep();
  renderFrame();
  updatePopulation(true);
});

resetBtn?.addEventListener('click', () => {
  running = false;
  updatePlayPauseVisual();
  clearState();
  generation = 0;
  population = 0;
  updateHud();
  renderFrame();
});

randomBtn?.addEventListener('click', () => {
  randomizeState();
  renderFrame();
  updatePopulation(true);
});

speedSlider?.addEventListener('input', () => {
  const value = Number(speedSlider.value);
  if (Number.isFinite(value) && value > 0) {
    speed = value;
  }
});

zoomSlider?.addEventListener('input', () => {
  const value = Number(zoomSlider.value);
  if (Number.isFinite(value) && value >= 1) {
    zoom = value;
    updateCanvasSize();
  }
});

settingsToggle?.addEventListener('click', () => {
  settingsPanel?.classList.toggle('hidden');
});

settingsClose?.addEventListener('click', () => {
  settingsPanel?.classList.add('hidden');
});

settingsForm?.addEventListener('submit', (event) => {
  event.preventDefault();
  const targetWidth = Math.max(16, Math.min(4096, Math.floor(Number(widthField?.value ?? gridWidth))));
  const targetHeight = Math.max(16, Math.min(4096, Math.floor(Number(heightField?.value ?? gridHeight))));
  resizeGrid(targetWidth, targetHeight);
  settingsPanel?.classList.add('hidden');
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
  gl.uniform1f(renderUniforms.uZoom, zoom);
  gl.uniform1f(renderUniforms.uPixelRatio, window.devicePixelRatio || 1);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
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
  updateHud();
}

function clearState() {
  const empty = new Uint8Array(gridWidth * gridHeight);
  uploadState(empty);
}

function randomizeState() {
  const densitySlider = randomDensitySlider ? Number(randomDensitySlider.value) : NaN;
  let density;
  if (Number.isFinite(densitySlider) && densitySlider >= 0 && densitySlider <= 100) {
    density = Math.max(20, Math.min(60, densitySlider)) / 100;
  } else {
    density = 0.2 + Math.random() * 0.4;
  }
  const data = new Uint8Array(gridWidth * gridHeight);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = Math.random() < density ? 255 : 0;
  }
  uploadState(data);
  generation = 0;
  updateHud();
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

function resizeGrid(newWidth, newHeight) {
  if (newWidth === gridWidth && newHeight === gridHeight) {
    return;
  }
  gridWidth = newWidth;
  gridHeight = newHeight;
  populationBuffer = new Uint8Array(gridWidth * gridHeight);
  initializeStateResources(gridWidth, gridHeight);

  gl.useProgram(simulationProgram);
  gl.uniform2i(simUniforms.uGridSize, gridWidth, gridHeight);
  gl.useProgram(renderProgram);
  gl.uniform2i(renderUniforms.uGridSize, gridWidth, gridHeight);

  generation = 0;
  population = 0;
  accumulator = 0;
  updateCanvasSize();
  updateHud();
  if (widthField) {
    widthField.value = String(gridWidth);
  }
  if (heightField) {
    heightField.value = String(gridHeight);
  }
  renderFrame();
  updatePopulation(true);
}

function updateCanvasSize() {
  const ratio = window.devicePixelRatio || 1;
  const displayWidth = Math.max(1, Math.floor(gridWidth * zoom));
  const displayHeight = Math.max(1, Math.floor(gridHeight * zoom));
  canvas.style.width = `${displayWidth}px`;
  canvas.style.height = `${displayHeight}px`;
  canvas.width = Math.max(1, Math.floor(displayWidth * ratio));
  canvas.height = Math.max(1, Math.floor(displayHeight * ratio));
  gl.viewport(0, 0, canvas.width, canvas.height);

  gl.useProgram(renderProgram);
  gl.uniform1f(renderUniforms.uZoom, zoom);
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
}

function getCellFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const relX = event.clientX - rect.left;
  const relY = event.clientY - rect.top;
  if (relX < 0 || relY < 0 || relX >= rect.width || relY >= rect.height) {
    return null;
  }
  const col = Math.floor((relX / rect.width) * gridWidth);
  let row = Math.floor((relY / rect.height) * gridHeight);
  row = gridHeight - 1 - row;
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
