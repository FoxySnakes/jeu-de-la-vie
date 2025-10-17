const VERTEX_SHADER_SOURCE = `#version 300 es
layout(location = 0) in vec2 a_position;
out vec2 v_texCoord;

void main() {
  v_texCoord = (a_position + 1.0) * 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const STEP_FRAGMENT_SOURCE = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_state;
uniform vec2 u_texelSize;

void main() {
  float state = texture(u_state, v_texCoord).r;
  float ageNorm = texture(u_state, v_texCoord).g;

  float neighbors = 0.0;
  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      if (x == 0 && y == 0) {
        continue;
      }
      vec2 offset = vec2(float(x), float(y)) * u_texelSize;
      neighbors += texture(u_state, v_texCoord + offset).r;
    }
  }

  float newState = state;
  if (state > 0.5) {
    if (neighbors < 2.0 || neighbors > 3.0) {
      newState = 0.0;
    }
  } else {
    if (neighbors == 3.0) {
      newState = 1.0;
    } else {
      newState = 0.0;
    }
  }

  float newAge;
  if (newState < 0.5) {
    newAge = 0.0;
  } else if (state < 0.5) {
    newAge = 1.0 / 255.0;
  } else {
    newAge = min(ageNorm + 1.0 / 255.0, 1.0);
  }

  outColor = vec4(newState, newAge, 0.0, 1.0);
}
`;

const RENDER_FRAGMENT_SOURCE = `#version 300 es
precision highp float;

out vec4 outColor;

uniform sampler2D u_state;
uniform vec2 u_stateSize;
uniform vec2 u_canvasToState;
uniform vec2 u_texelSize;

vec3 palette(float t) {
  vec3 c1 = vec3(0.05, 0.20, 0.60);
  vec3 c2 = vec3(0.00, 0.85, 1.00);
  vec3 c3 = vec3(0.85, 0.25, 1.00);
  float mid = smoothstep(0.0, 0.5, t);
  float high = smoothstep(0.3, 1.0, t);
  vec3 color = mix(c1, c2, mid);
  color = mix(color, c3, high);
  return color;
}

const vec2 NEIGHBOR_OFFSETS[8] = vec2[](
  vec2(-1.0, -1.0), vec2(0.0, -1.0), vec2(1.0, -1.0),
  vec2(-1.0, 0.0), vec2(1.0, 0.0),
  vec2(-1.0, 1.0), vec2(0.0, 1.0), vec2(1.0, 1.0)
);

void main() {
  vec2 statePx = floor(gl_FragCoord.xy * u_canvasToState);
  vec2 uv = (statePx + 0.5) / u_stateSize;
  vec4 texel = texture(u_state, uv);
  float alive = texel.r;
  float ageNorm = texel.g;

  float halo = 0.0;
  for (int i = 0; i < 8; ++i) {
    vec2 sampleUv = uv + NEIGHBOR_OFFSETS[i] * u_texelSize;
    halo += texture(u_state, sampleUv).r;
  }
  halo *= 0.065;

  float intensity = alive * (0.65 + ageNorm * 0.35) + halo;
  intensity = clamp(intensity, 0.0, 1.0);

  vec3 baseColor = palette(ageNorm);
  baseColor *= mix(0.35, 1.35, pow(ageNorm, 0.6));
  vec3 color = baseColor * intensity;

  outColor = vec4(color, intensity);
}
`;

const REDUCE_FRAGMENT_SOURCE = `#version 300 es
precision highp float;

out vec4 outColor;

uniform sampler2D u_state;
uniform vec2 u_invStateSize;
uniform ivec2 u_blockDim;
uniform ivec2 u_stateSize;

void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * u_blockDim;
  int maxY = clamp(u_stateSize.y - base.y, 0, u_blockDim.y);
  int maxX = clamp(u_stateSize.x - base.x, 0, u_blockDim.x);
  float sum = 0.0;
  for (int by = 0; by < 16; ++by) {
    if (by >= maxY) {
      break;
    }
    for (int bx = 0; bx < 16; ++bx) {
      if (bx >= maxX) {
        break;
      }
      vec2 uv = (vec2(base + ivec2(bx, by)) + 0.5) * u_invStateSize;
      sum += texture(u_state, uv).r;
    }
  }
  outColor = vec4(sum / 255.0, 0.0, 0.0, 1.0);
}
`;

function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  if (!shader) {
    throw new Error('Impossible de créer le shader');
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Erreur de compilation shader: ${log}`);
  }
  return shader;
}

function createProgram(gl, vertexSource, fragmentSource) {
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  if (!program) {
    throw new Error('Impossible de créer le programme shader');
  }
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Erreur de linkage shader: ${log}`);
  }
  return program;
}

function ensureContext(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, alpha: true, premultipliedAlpha: false });
  if (!gl) {
    throw new Error('WebGL2 non disponible');
  }
  return gl;
}

export class GameOfLifeWebGL {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.rows = options.rows || 512;
    this.cols = options.cols || 512;
    this.speed = options.speed || 30;
    this.onStats = options.onStats || (() => {});
    this.onError = options.onError || (() => {});

    this.devicePixelRatio = window.devicePixelRatio || 1;

    this.gl = null;
    this.stateTextures = [];
    this.framebuffers = [];
    this.ping = 0;

    this.reduceTexture = null;
    this.reduceFramebuffer = null;
    this.reduceSize = { width: 0, height: 0 };
    this.reduceBlock = { x: 1, y: 1 };

    this.stepProgram = null;
    this.renderProgram = null;
    this.reduceProgram = null;
    this.fullscreenVao = null;

    this.running = false;
    this.accumulator = 0;
    this.stepInterval = 1 / this.speed;
    this.lastFrameTime = 0;
    this.animationFrame = null;

    this.generation = 0;
    this.aliveCount = 0;
    this.lastAliveUpdate = 0;

    this.stateBuffer = null;
    this.pixelWriteBuffer = new Uint8Array(4);
    this.pixelReadBuffer = new Uint8Array(4);
    this.failed = false;

    try {
      this.initGL();
    } catch (error) {
      this.failed = true;
      this.onError(error);
      return;
    }

    this.randomize();
    this.pause();
    this.renderFrame(0);
  }

  initGL() {
    const gl = ensureContext(this.canvas);
    this.gl = gl;

    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);

    this.stepProgram = createProgram(gl, VERTEX_SHADER_SOURCE, STEP_FRAGMENT_SOURCE);
    this.renderProgram = createProgram(gl, VERTEX_SHADER_SOURCE, RENDER_FRAGMENT_SOURCE);
    this.reduceProgram = createProgram(gl, VERTEX_SHADER_SOURCE, REDUCE_FRAGMENT_SOURCE);

    this.createFullscreenQuad();
    this.resizeCanvas();
    this.allocateStateTextures();
    this.setupReduceTarget();

    this.loop = this.loop.bind(this);
    this.handleResize = this.handleResize.bind(this);
    window.addEventListener('resize', this.handleResize);
    this.animationFrame = requestAnimationFrame(this.loop);
  }

  createFullscreenQuad() {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    const vertices = new Float32Array([
      -1, -1,
      1, -1,
      -1, 1,
      -1, 1,
      1, -1,
      1, 1,
    ]);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    this.fullscreenVao = vao;
  }

  resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    this.devicePixelRatio = dpr;
    const width = Math.max(1, Math.floor(this.cols));
    const height = Math.max(1, Math.floor(this.rows));
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
  }

  handleResize() {
    if (!this.gl) return;
    this.resizeCanvas();
  }

  allocateStateTextures() {
    const gl = this.gl;
    const width = this.cols;
    const height = this.rows;

    this.deleteStateResources();

    this.stateTextures = [gl.createTexture(), gl.createTexture()];
    this.framebuffers = [gl.createFramebuffer(), gl.createFramebuffer()];

    for (let i = 0; i < 2; i += 1) {
      const texture = this.stateTextures[i];
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

      const fbo = this.framebuffers[i];
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Framebuffer d\'état invalide');
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.stateBuffer = new Uint8Array(width * height * 4);
    this.ping = 0;
  }

  setupReduceTarget() {
    const gl = this.gl;
    if (this.reduceTexture) {
      gl.deleteTexture(this.reduceTexture);
      this.reduceTexture = null;
    }
    if (this.reduceFramebuffer) {
      gl.deleteFramebuffer(this.reduceFramebuffer);
      this.reduceFramebuffer = null;
    }

    const blockCap = 8;
    const blockX = Math.min(blockCap, Math.max(1, this.cols));
    const blockY = Math.min(blockCap, Math.max(1, this.rows));
    const reduceWidth = Math.ceil(this.cols / blockX);
    const reduceHeight = Math.ceil(this.rows / blockY);

    this.reduceTexture = gl.createTexture();
    this.reduceFramebuffer = gl.createFramebuffer();
    this.reduceSize = { width: reduceWidth, height: reduceHeight };
    this.reduceBlock = { x: blockX, y: blockY };

    gl.bindTexture(gl.TEXTURE_2D, this.reduceTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, reduceWidth, reduceHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.reduceFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.reduceTexture, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('Framebuffer de réduction invalide');
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.reduceReadBuffer = new Uint8Array(reduceWidth * reduceHeight * 4);
  }

  deleteStateResources() {
    if (!this.gl) return;
    for (const tex of this.stateTextures) {
      if (tex) {
        this.gl.deleteTexture(tex);
      }
    }
    for (const fbo of this.framebuffers) {
      if (fbo) {
        this.gl.deleteFramebuffer(fbo);
      }
    }
    this.stateTextures = [];
    this.framebuffers = [];
  }

  loop(time) {
    if (!this.gl) {
      return;
    }
    if (!this.lastFrameTime) {
      this.lastFrameTime = time;
    }
    const delta = (time - this.lastFrameTime) / 1000;
    this.lastFrameTime = time;

    if (this.running) {
      this.accumulator += delta;
      while (this.accumulator >= this.stepInterval) {
        this.stepSimulation();
        this.accumulator -= this.stepInterval;
      }
    }

    this.renderFrame(time);

    if (time - this.lastAliveUpdate > 100) {
      this.updateAliveCount();
      this.lastAliveUpdate = time;
    }

    this.animationFrame = requestAnimationFrame(this.loop);
  }

  bindStateTextures(program) {
    const gl = this.gl;
    gl.useProgram(program);
    const stateLocation = gl.getUniformLocation(program, 'u_state');
    gl.uniform1i(stateLocation, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.stateTextures[this.ping]);
  }

  stepSimulation() {
    const gl = this.gl;

    gl.useProgram(this.stepProgram);
    gl.bindVertexArray(this.fullscreenVao);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers[1 - this.ping]);
    gl.viewport(0, 0, this.cols, this.rows);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.stateTextures[this.ping]);
    const texelLoc = gl.getUniformLocation(this.stepProgram, 'u_texelSize');
    gl.uniform2f(texelLoc, 1 / this.cols, 1 / this.rows);
    const stateLoc = gl.getUniformLocation(this.stepProgram, 'u_state');
    gl.uniform1i(stateLoc, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    this.ping = 1 - this.ping;
    this.generation += 1;
    this.onStats({ generation: this.generation, alive: this.aliveCount, rows: this.rows, cols: this.cols });
  }

  renderFrame() {
    const gl = this.gl;
    gl.useProgram(this.renderProgram);
    gl.bindVertexArray(this.fullscreenVao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.stateTextures[this.ping]);

    const stateSizeLoc = gl.getUniformLocation(this.renderProgram, 'u_stateSize');
    gl.uniform2f(stateSizeLoc, this.cols, this.rows);
    const canvasToStateLoc = gl.getUniformLocation(this.renderProgram, 'u_canvasToState');
    gl.uniform2f(canvasToStateLoc, this.cols / this.canvas.width, this.rows / this.canvas.height);
    const texelLoc = gl.getUniformLocation(this.renderProgram, 'u_texelSize');
    gl.uniform2f(texelLoc, 1 / this.cols, 1 / this.rows);
    const stateLoc = gl.getUniformLocation(this.renderProgram, 'u_state');
    gl.uniform1i(stateLoc, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  updateAliveCount() {
    const gl = this.gl;
    gl.useProgram(this.reduceProgram);
    gl.bindVertexArray(this.fullscreenVao);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.reduceFramebuffer);
    gl.viewport(0, 0, this.reduceSize.width, this.reduceSize.height);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.stateTextures[this.ping]);

    const invSizeLoc = gl.getUniformLocation(this.reduceProgram, 'u_invStateSize');
    gl.uniform2f(invSizeLoc, 1 / this.cols, 1 / this.rows);
    const blockLoc = gl.getUniformLocation(this.reduceProgram, 'u_blockDim');
    gl.uniform2i(blockLoc, this.reduceBlock.x, this.reduceBlock.y);
    const stateSizeLoc = gl.getUniformLocation(this.reduceProgram, 'u_stateSize');
    gl.uniform2i(stateSizeLoc, this.cols, this.rows);
    const stateLoc = gl.getUniformLocation(this.reduceProgram, 'u_state');
    gl.uniform1i(stateLoc, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    gl.readPixels(0, 0, this.reduceSize.width, this.reduceSize.height, gl.RGBA, gl.UNSIGNED_BYTE, this.reduceReadBuffer);

    let total = 0;
    const data = this.reduceReadBuffer;
    for (let i = 0; i < data.length; i += 4) {
      total += data[i];
    }

    this.aliveCount = total;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    this.onStats({ generation: this.generation, alive: this.aliveCount, rows: this.rows, cols: this.cols });
  }

  ensureBuffers() {
    if (!this.stateBuffer || this.stateBuffer.length !== this.rows * this.cols * 4) {
      this.stateBuffer = new Uint8Array(this.rows * this.cols * 4);
    }
  }

  randomize() {
    if (!this.gl) return;
    this.ensureBuffers();
    const buffer = this.stateBuffer;
    const length = this.rows * this.cols;
    let aliveTotal = 0;
    for (let i = 0; i < length; i += 1) {
      const alive = Math.random() < 0.5 ? 255 : 0;
      const idx = i * 4;
      const aliveBit = alive > 0 ? 1 : 0;
      buffer[idx] = aliveBit ? 255 : 0;
      buffer[idx + 1] = aliveBit;
      buffer[idx + 2] = 0;
      buffer[idx + 3] = 255;
      aliveTotal += aliveBit;
    }
    this.uploadState(buffer);
    this.generation = 0;
    this.aliveCount = aliveTotal;
    this.accumulator = 0;
    this.onStats({ generation: this.generation, alive: this.aliveCount, rows: this.rows, cols: this.cols });
    this.updateAliveCount();
  }

  clear() {
    if (!this.gl) return;
    this.ensureBuffers();
    this.stateBuffer.fill(0);
    for (let i = 3; i < this.stateBuffer.length; i += 4) {
      this.stateBuffer[i] = 255;
    }
    this.uploadState(this.stateBuffer);
    this.generation = 0;
    this.aliveCount = 0;
    this.accumulator = 0;
    this.onStats({ generation: this.generation, alive: this.aliveCount, rows: this.rows, cols: this.cols });
    this.updateAliveCount();
  }

  uploadState(buffer) {
    const gl = this.gl;
    for (let i = 0; i < 2; i += 1) {
      gl.bindTexture(gl.TEXTURE_2D, this.stateTextures[i]);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.cols, this.rows, gl.RGBA, gl.UNSIGNED_BYTE, buffer);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  start() {
    if (!this.gl) return;
    this.running = true;
  }

  pause() {
    this.running = false;
  }

  isRunning() {
    return this.running;
  }

  stepOnce() {
    if (!this.gl) return;
    this.stepSimulation();
    this.updateAliveCount();
  }

  resize(rows, cols) {
    const newRows = Math.max(1, Math.floor(rows));
    const newCols = Math.max(1, Math.floor(cols));
    if (newRows === this.rows && newCols === this.cols) {
      this.randomize();
      return;
    }
    this.rows = newRows;
    this.cols = newCols;
    this.resizeCanvas();
    this.allocateStateTextures();
    this.setupReduceTarget();
    this.accumulator = 0;
    this.lastFrameTime = 0;
    this.randomize();
  }

  setSpeed(generationsPerSecond) {
    const speed = Math.max(0.1, generationsPerSecond);
    this.speed = speed;
    this.stepInterval = 1 / this.speed;
  }

  toggleCell(col, row) {
    if (!this.gl) return;
    if (col < 0 || col >= this.cols || row < 0 || row >= this.rows) {
      return;
    }
    const gl = this.gl;
    const texY = this.rows - 1 - row;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers[this.ping]);
    gl.readPixels(col, texY, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, this.pixelReadBuffer);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    const wasAlive = this.pixelReadBuffer[0] > 127;
    const newState = wasAlive ? 0 : 255;
    const newAge = wasAlive ? 0 : 1;
    this.pixelWriteBuffer[0] = newState;
    this.pixelWriteBuffer[1] = newAge ? 1 : 0;
    this.pixelWriteBuffer[2] = 0;
    this.pixelWriteBuffer[3] = 255;

    for (let i = 0; i < 2; i += 1) {
      gl.bindTexture(gl.TEXTURE_2D, this.stateTextures[i]);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, col, texY, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, this.pixelWriteBuffer);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.updateAliveCount();
  }

  setStatsCallback(callback) {
    this.onStats = callback;
  }

  destroy() {
    if (!this.gl) return;
    cancelAnimationFrame(this.animationFrame);
    if (this.handleResize) {
      window.removeEventListener('resize', this.handleResize);
    }
    this.deleteStateResources();
    if (this.reduceTexture) {
      this.gl.deleteTexture(this.reduceTexture);
    }
    if (this.reduceFramebuffer) {
      this.gl.deleteFramebuffer(this.reduceFramebuffer);
    }
    if (this.fullscreenVao) {
      this.gl.deleteVertexArray(this.fullscreenVao);
    }
    if (this.stepProgram) {
      this.gl.deleteProgram(this.stepProgram);
    }
    if (this.renderProgram) {
      this.gl.deleteProgram(this.renderProgram);
    }
    if (this.reduceProgram) {
      this.gl.deleteProgram(this.reduceProgram);
    }
    this.gl = null;
  }
}
