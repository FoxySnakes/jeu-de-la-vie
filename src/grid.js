/**
 * Gestionnaire de grille optimisé pour le Jeu de la Vie.
 * Utilise un tampon étendu (bordure) pour limiter les conditions aux frontières
 * et deux buffers pour éviter les allocations à chaque génération.
 */
export class Grid {
  constructor(width, height) {
    this.generation = 0;
    this.resize(width, height);
  }

  resize(width, height) {
    this.width = width | 0;
    this.height = height | 0;
    this.stride = this.width + 2;
    this.length = (this.width + 2) * (this.height + 2);
    this.current = new Uint8Array(this.length);
    this.next = new Uint8Array(this.length);
    this.generation = 0;
    this.population = 0;
  }

  /**
   * Retourne l'index linéaire du tampon étendu.
   */
  index(x, y) {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
      return -1;
    }
    return (y + 1) * this.stride + (x + 1);
  }

  toggle(x, y) {
    const idx = this.index(x, y);
    if (idx < 0) return;
    if (this.current[idx] > 0) {
      this.current[idx] = 0;
      this.population--;
    } else {
      this.current[idx] = 1;
      this.population++;
    }
  }

  setCell(x, y, alive, age = 1) {
    const idx = this.index(x, y);
    if (idx < 0) return;
    const wasAlive = this.current[idx] > 0;
    if (alive) {
      this.current[idx] = Math.max(age, 1);
      if (!wasAlive) this.population++;
    } else {
      this.current[idx] = 0;
      if (wasAlive) this.population--;
    }
  }

  clear() {
    this.current.fill(0);
    this.next.fill(0);
    this.population = 0;
    this.generation = 0;
  }

  randomize(probability) {
    const threshold = Math.max(0, Math.min(1, probability));
    let count = 0;
    const { current, width, height, stride } = this;
    current.fill(0);
    this.next.fill(0);
    for (let y = 0; y < height; y++) {
      let idx = (y + 1) * stride + 1;
      for (let x = 0; x < width; x++, idx++) {
        const alive = Math.random() < threshold;
        current[idx] = alive ? 1 : 0;
        if (alive) count++;
      }
    }
    this.population = count;
    this.generation = 0;
  }

  applyPattern(pattern, offsetX, offsetY) {
    const { width, height } = this;
    const ox = Math.floor(offsetX - pattern.width / 2);
    const oy = Math.floor(offsetY - pattern.height / 2);
    for (const [px, py] of pattern.cells) {
      const x = ox + px;
      const y = oy + py;
      if (x >= 0 && x < width && y >= 0 && y < height) {
        const idx = this.index(x, y);
        if (this.current[idx] === 0) {
          this.population++;
        }
        this.current[idx] = 1;
      }
    }
  }

  /**
   * Avance la simulation d'une génération.
   * Retourne le nombre de cellules vivantes après mise à jour.
   */
  step() {
    const { current, next, stride, width, height } = this;
    let newPopulation = 0;
    let idx = stride + 1;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++, idx++) {
        const north = idx - stride;
        const south = idx + stride;
        const alive = current[idx] > 0;
        let neighbors = 0;
        neighbors += current[north - 1] > 0;
        neighbors += current[north] > 0;
        neighbors += current[north + 1] > 0;
        neighbors += current[idx - 1] > 0;
        neighbors += current[idx + 1] > 0;
        neighbors += current[south - 1] > 0;
        neighbors += current[south] > 0;
        neighbors += current[south + 1] > 0;

        let nextValue = 0;
        if (neighbors === 3 || (alive && neighbors === 2)) {
          const age = current[idx] ? Math.min(current[idx] + 1, 255) : 1;
          nextValue = age;
          newPopulation++;
        }
        next[idx] = nextValue;
      }
      idx += 2; // saut de la bordure droite puis cell de gauche prochaine ligne
    }
    // Nettoyage des bordures horizontales
    next.fill(0, 0, stride);
    next.fill(0, next.length - stride, next.length);

    const lastRowStart = height * stride;
    for (let i = stride; i <= lastRowStart; i += stride) {
      next[i] = 0;
      next[i + stride - 1] = 0;
    }

    this.population = newPopulation;
    this.generation++;

    // échange des buffers
    const temp = this.current;
    this.current = this.next;
    this.next = temp;

    return newPopulation;
  }
}
