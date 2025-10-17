import { Grid } from './grid.js';
import { Renderer } from './renderer.js';
import { PATTERNS, getPatternById } from './patterns.js';

const DEFAULT_WIDTH = 512;
const DEFAULT_HEIGHT = 512;

export class Controller {
  constructor() {
    this.canvas = document.getElementById('world');
    this.grid = new Grid(DEFAULT_WIDTH, DEFAULT_HEIGHT);
    this.renderer = new Renderer(this.canvas, this.grid);

    this.isRunning = false;
    this.lastTime = 0;
    this.accumulator = 0;
    this.stepDuration = 1000 / 30; // 30 générations par seconde par défaut
    this.isPainting = false;
    this.paintValue = true;
    this.isPanning = false;
    this.panReference = { x: 0, y: 0 };
    this.lastPainted = null;

    this.elements = this.bindElements();
    this.populatePatternList();
    this.elements.zoom.value = this.renderer.getSliderValue();
    this.setSpeed(Number(this.elements.speed.value));
    this.attachEvents();
    this.updateHud();
    this.loop = this.loop.bind(this);
    requestAnimationFrame(this.loop);
  }

  bindElements() {
    return {
      playPause: document.getElementById('play-pause'),
      step: document.getElementById('step'),
      reset: document.getElementById('reset'),
      random: document.getElementById('random'),
      speed: document.getElementById('speed'),
      zoom: document.getElementById('zoom'),
      pattern: document.getElementById('pattern'),
      hudGeneration: document.getElementById('generation'),
      hudPopulation: document.getElementById('population'),
      hudDimensions: document.getElementById('dimensions'),
      settingsToggle: document.getElementById('settings-toggle'),
      settingsPanel: document.getElementById('settings-panel'),
      settingsForm: document.getElementById('settings-form'),
      settingsClose: document.getElementById('settings-close'),
      gridWidth: document.getElementById('grid-width'),
      gridHeight: document.getElementById('grid-height'),
      randomDensity: document.getElementById('random-density'),
      styleMode: document.getElementById('style-mode'),
    };
  }

  populatePatternList() {
    const { pattern } = this.elements;
    if (!pattern) return;
    PATTERNS.forEach((item) => {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = item.name;
      pattern.appendChild(option);
    });
  }

  attachEvents() {
    const {
      playPause,
      step,
      reset,
      random,
      speed,
      zoom,
      pattern,
      settingsToggle,
      settingsPanel,
      settingsForm,
      settingsClose,
      gridWidth,
      gridHeight,
      styleMode,
    } = this.elements;

    window.addEventListener('resize', () => {
      this.renderer.resize();
      this.renderer.clampOrigin();
    });

    playPause.addEventListener('click', () => this.togglePlay());
    step.addEventListener('click', () => this.stepOnce());
    reset.addEventListener('click', () => this.resetGrid());
    random.addEventListener('click', () => this.randomizeGrid());

    speed.addEventListener('input', () => {
      const value = Number(speed.value);
      this.setSpeed(value);
    });

    zoom.addEventListener('input', () => {
      this.renderer.setScaleFromSlider(Number(zoom.value));
    });

    pattern.addEventListener('change', () => {
      const { value } = pattern;
      if (!value) return;
      const selected = getPatternById(value);
      if (selected) {
        const centerX = this.renderer.originX + (this.canvas.clientWidth / this.renderer.scale) / 2;
        const centerY = this.renderer.originY + (this.canvas.clientHeight / this.renderer.scale) / 2;
        this.grid.applyPattern(selected, centerX, centerY);
        this.updateHud();
      }
      pattern.value = '';
    });

    settingsToggle.addEventListener('click', () => {
      settingsPanel.classList.toggle('hidden');
    });

    settingsClose.addEventListener('click', () => {
      settingsPanel.classList.add('hidden');
    });

    settingsForm.addEventListener('submit', (event) => {
      event.preventDefault();
      const width = Number(gridWidth.value) || this.grid.width;
      const height = Number(gridHeight.value) || this.grid.height;
      const mode = styleMode.value || 'neon';
      this.resizeGrid(width, height);
      this.renderer.setMode(mode);
      this.renderer.needsPaletteUpdate = true;
      this.renderer.updatePalette();
      this.renderer.centerOnGrid();
      this.renderer.clampOrigin();
      settingsPanel.classList.add('hidden');
      this.updateHud();
    });

    this.canvas.addEventListener('pointerdown', (event) => this.handlePointerDown(event));
    this.canvas.addEventListener('pointermove', (event) => this.handlePointerMove(event));
    this.canvas.addEventListener('pointerup', (event) => this.handlePointerUp(event));
    this.canvas.addEventListener('pointerleave', (event) => this.handlePointerUp(event));
    this.canvas.addEventListener('wheel', (event) => this.handleWheel(event), { passive: false });
    this.canvas.addEventListener('contextmenu', (event) => event.preventDefault());
  }

  setSpeed(value) {
    const clamped = Math.max(1, Math.min(240, value));
    this.stepDuration = 1000 / clamped;
  }

  handlePointerDown(event) {
    if (event.button === 0 || event.button === 2) {
      this.canvas.setPointerCapture(event.pointerId);
    }
    if (event.button === 0) {
      const { x, y } = this.renderer.screenToCell(event.clientX, event.clientY);
      const idx = this.grid.index(x, y);
      if (idx >= 0) {
        const alive = this.grid.current[idx] === 0;
        this.grid.setCell(x, y, alive);
        this.isPainting = true;
        this.paintValue = alive;
        this.lastPainted = { x, y };
        this.updateHud();
      }
    } else if (event.button === 2) {
      this.isPanning = true;
      this.panReference.x = event.clientX;
      this.panReference.y = event.clientY;
    }
  }

  handlePointerMove(event) {
    if (this.isPainting) {
      const { x, y } = this.renderer.screenToCell(event.clientX, event.clientY);
      if (!this.lastPainted || this.lastPainted.x !== x || this.lastPainted.y !== y) {
        this.grid.setCell(x, y, this.paintValue);
        this.lastPainted = { x, y };
        this.updateHud();
      }
    } else if (this.isPanning) {
      const dx = event.clientX - this.panReference.x;
      const dy = event.clientY - this.panReference.y;
      this.renderer.pan(dx, dy);
      this.panReference.x = event.clientX;
      this.panReference.y = event.clientY;
    }
  }

  handlePointerUp(event) {
    if (this.canvas.hasPointerCapture?.(event.pointerId)) {
      this.canvas.releasePointerCapture(event.pointerId);
    }
    if (event.button === 0 || event.type === 'pointerleave' || event.buttons === 0) {
      this.isPainting = false;
      this.lastPainted = null;
    }
    if (event.button === 2 || event.type === 'pointerleave' || event.buttons === 0) {
      this.isPanning = false;
    }
  }

  handleWheel(event) {
    event.preventDefault();
    this.renderer.zoomAt(event.clientX, event.clientY, event.deltaY);
    this.elements.zoom.value = this.renderer.getSliderValue();
  }

  togglePlay() {
    this.isRunning = !this.isRunning;
    this.elements.playPause.textContent = this.isRunning ? '⏸️' : '▶️';
  }

  stepOnce() {
    this.grid.step();
    this.updateHud();
  }

  resetGrid() {
    this.grid.clear();
    this.renderer.centerOnGrid();
    this.updateHud();
  }

  randomizeGrid() {
    const density = Number(this.elements.randomDensity.value) / 100;
    this.grid.randomize(density);
    this.updateHud();
  }

  resizeGrid(width, height) {
    this.grid.resize(width, height);
    this.renderer.centerOnGrid();
    this.renderer.clampOrigin();
    this.elements.gridWidth.value = width;
    this.elements.gridHeight.value = height;
  }

  updateHud() {
    this.elements.hudGeneration.textContent = this.grid.generation.toString();
    this.elements.hudPopulation.textContent = this.grid.population.toLocaleString('fr-FR');
    this.elements.hudDimensions.textContent = `${this.grid.width} × ${this.grid.height}`;
  }

  loop(time) {
    this.renderer.resize();
    if (!this.lastTime) this.lastTime = time;
    const delta = time - this.lastTime;
    this.lastTime = time;

    if (this.isRunning) {
      this.accumulator += delta;
      while (this.accumulator >= this.stepDuration) {
        this.grid.step();
        this.accumulator -= this.stepDuration;
      }
      this.updateHud();
    }

    this.renderer.draw();
    requestAnimationFrame(this.loop);
  }
}

export function bootstrap() {
  return new Controller();
}
