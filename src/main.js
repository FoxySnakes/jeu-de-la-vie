import { GameOfLifeWebGL } from './gameOfLifeWebGL.js';

const STORAGE_KEY = 'gol-webgl-settings';
const DEFAULT_SETTINGS = {
  width: 512,
  height: 512,
  speed: 30,
};

const numberFormatter = new Intl.NumberFormat('fr-FR');

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return { ...DEFAULT_SETTINGS };
    }
    const parsed = JSON.parse(raw);
    return {
      width: Math.max(16, Math.floor(parsed.width || DEFAULT_SETTINGS.width)),
      height: Math.max(16, Math.floor(parsed.height || DEFAULT_SETTINGS.height)),
      speed: Math.max(0.1, Number(parsed.speed) || DEFAULT_SETTINGS.speed),
    };
  } catch (error) {
    console.warn('Impossible de charger les paramètres, utilisation des valeurs par défaut.', error);
    return { ...DEFAULT_SETTINGS };
  }
}

function saveSettings(settings) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch (error) {
    console.warn('Impossible de sauvegarder les paramètres.', error);
  }
}

function ensureCanvas() {
  let canvas = document.getElementById('gameCanvas');
  if (canvas) {
    return canvas;
  }
  canvas = document.createElement('canvas');
  canvas.id = 'gameCanvas';
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'Jeu de la Vie de Conway');
  const container = document.querySelector('#canvasContainer, .canvas-container, main, #app, body');
  (container || document.body).prepend(canvas);
  return canvas;
}

function updateButtonState(button, active) {
  if (!button) return;
  if (typeof button.classList?.toggle === 'function') {
    button.classList.toggle('is-active', Boolean(active));
  }
}

function attachUi(game, settings) {
  const startBtn = document.getElementById('startBtn');
  const pauseBtn = document.getElementById('pauseBtn');
  const widthInput = document.getElementById('width');
  const heightInput = document.getElementById('height');
  const speedInput = document.getElementById('speed');
  const statsGeneration = document.getElementById('statsGeneration');
  const statsAlive = document.getElementById('statsAlive');
  const statsDimensions = document.getElementById('statsDimensions');

  const stats = {
    generation: statsGeneration,
    alive: statsAlive,
    dimensions: statsDimensions,
  };

  const updateStats = ({ generation, alive, rows, cols }) => {
    if (stats.generation) {
      stats.generation.textContent = numberFormatter.format(generation);
    }
    if (stats.alive) {
      stats.alive.textContent = numberFormatter.format(alive);
    }
    if (stats.dimensions) {
      stats.dimensions.textContent = `${numberFormatter.format(cols)} × ${numberFormatter.format(rows)}`;
    }
  };

  game.setStatsCallback(updateStats);
  updateStats({ generation: game.generation || 0, alive: game.aliveCount || 0, rows: game.rows, cols: game.cols });

  if (widthInput) {
    widthInput.value = settings.width;
    widthInput.addEventListener('change', () => {
      const value = Number(widthInput.value);
      if (!Number.isFinite(value)) {
        widthInput.value = game.cols;
        return;
      }
      const clamped = Math.max(16, Math.min(4096, Math.floor(value)));
      widthInput.value = clamped;
      settings.width = clamped;
      game.resize(game.rows, clamped);
      saveSettings({ width: game.cols, height: game.rows, speed: game.speed });
    });
  }

  if (heightInput) {
    heightInput.value = settings.height;
    heightInput.addEventListener('change', () => {
      const value = Number(heightInput.value);
      if (!Number.isFinite(value)) {
        heightInput.value = game.rows;
        return;
      }
      const clamped = Math.max(16, Math.min(4096, Math.floor(value)));
      heightInput.value = clamped;
      settings.height = clamped;
      game.resize(clamped, game.cols);
      saveSettings({ width: game.cols, height: game.rows, speed: game.speed });
    });
  }

  if (speedInput) {
    speedInput.value = settings.speed;
    speedInput.addEventListener('input', () => {
      const value = Number(speedInput.value);
      if (!Number.isFinite(value)) {
        return;
      }
      const normalized = Math.max(0.1, Number(value));
      game.setSpeed(normalized);
      settings.speed = normalized;
      saveSettings({ width: game.cols, height: game.rows, speed: normalized });
    });
  }

  if (startBtn) {
    startBtn.addEventListener('click', () => {
      game.start();
      updateButtonState(startBtn, true);
      updateButtonState(pauseBtn, false);
    });
  }

  if (pauseBtn) {
    pauseBtn.addEventListener('click', () => {
      game.pause();
      updateButtonState(startBtn, false);
      updateButtonState(pauseBtn, true);
    });
  }

  let pointerActive = false;
  let lastPointerKey = null;
  const pointerToggle = (event) => {
    if (game.failed) return;
    const rect = game.canvas.getBoundingClientRect();
    const offsetX = event.clientX - rect.left;
    const offsetY = event.clientY - rect.top;
    if (offsetX < 0 || offsetY < 0 || offsetX > rect.width || offsetY > rect.height) {
      return;
    }
    const col = Math.floor((offsetX / rect.width) * game.cols);
    const row = Math.floor((offsetY / rect.height) * game.rows);
    const key = `${col}:${row}`;
    if (lastPointerKey === key) {
      return;
    }
    lastPointerKey = key;
    game.toggleCell(col, row);
  };

  game.canvas.addEventListener('pointerdown', (event) => {
    if (typeof event.button === 'number' && event.button !== 0) {
      return;
    }
    pointerActive = true;
    lastPointerKey = null;
    event.preventDefault();
    pointerToggle(event);
    try {
      game.canvas.setPointerCapture(event.pointerId);
    } catch (err) {
      // ignore capture errors (unsupported browsers)
    }
  });
  game.canvas.addEventListener('pointermove', (event) => {
    if (!pointerActive || (event.buttons & 1) === 0) {
      return;
    }
    pointerToggle(event);
  });
  const endPointer = (event) => {
    pointerActive = false;
    lastPointerKey = null;
    try {
      game.canvas.releasePointerCapture(event.pointerId);
    } catch (err) {
      // ignore capture errors (unsupported browsers)
    }
  };
  game.canvas.addEventListener('pointerup', endPointer);
  game.canvas.addEventListener('pointercancel', endPointer);
  game.canvas.addEventListener('contextmenu', (event) => {
    event.preventDefault();
  });

  document.addEventListener('keydown', (event) => {
    if (event.target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) {
      return;
    }
    if (event.key === 'r' || event.key === 'R') {
      event.preventDefault();
      game.randomize();
    } else if (event.key === 'w' || event.key === 'W') {
      event.preventDefault();
      game.clear();
    }
  });

  updateButtonState(startBtn, game.isRunning());
  updateButtonState(pauseBtn, !game.isRunning());
}

function showWebglError(canvas, error) {
  const message = document.createElement('div');
  message.textContent = `WebGL2 indisponible : ${error.message || error}`;
  message.style.padding = '1rem';
  message.style.textAlign = 'center';
  message.style.color = '#ff9aa2';
  message.style.fontWeight = '600';
  canvas.replaceWith(message);
}

document.addEventListener('DOMContentLoaded', () => {
  const canvas = ensureCanvas();
  canvas.style.imageRendering = 'pixelated';
  canvas.style.touchAction = 'none';

  const settings = loadSettings();
  const game = new GameOfLifeWebGL(canvas, {
    rows: settings.height,
    cols: settings.width,
    speed: settings.speed,
    onStats: () => {},
    onError: (error) => showWebglError(canvas, error),
  });

  if (!game.gl || game.failed) {
    return;
  }

  const currentSettings = { width: game.cols, height: game.rows, speed: game.speed };
  saveSettings(currentSettings);
  attachUi(game, currentSettings);
});
