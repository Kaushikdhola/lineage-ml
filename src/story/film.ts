/**
 * Radiograph renderer. A film is a grid of cells (tokens, or documents for leak matrices) drawn as an
 * exposure on a dark base: brightness is the measured value. The film reads like a scanner display:
 * plate and run kind in the top-left corner, window and level in the top-right, the value under the
 * pointer in the bottom-right, and measurement brackets (regions of interest) instead of free-hand
 * marks. Pure view code: every value it shows comes from measured data or src/core.
 */
export interface Annotation {
  rows?: [number, number]; // inclusive row range to bracket; omit for a film-wide note
  cols?: [number, number]; // optional column range; default: the whole drawn width of those rows
  note: string;
}

export interface FilmSpec {
  rows: number[][]; // luminance source values, one array per row (ragged allowed)
  scale: (v: number) => number; // value -> luminance in [0, 1]; ignored while a window is set
  rowLabels?: string[];
  edge: string; // "Plate n   Kind   what it is" (two or more spaces between fields)
  annotations?: Annotation[];
  separators?: number[]; // draw a boundary before these row indices (micro-batch edges)
  tooltip?: (r: number, c: number, v: number) => string;
  square?: boolean; // matrix films: square cells
  summary: string; // accessible description
  intended?: number[][]; // what the equation says, per cell (adds a column to the screen-reader table)
  /**
   * Window and level, as on a radiology display: luminance = (v − level) / width + ½. When set, the
   * reader can drag across the film to change them (horizontal: width, vertical: level).
   */
  window?: { level: number; width: number };
  corner?: string; // bottom-left readout, e.g. the number of sequences and targets
}

const meanOf = (r: number[]) => r.reduce((a, b) => a + b, 0) / r.length;
const num = (x: number) => (x !== 0 && Math.abs(x) < 0.01 ? x.toExponential(2) : x.toFixed(3));
const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

/** Film response: base fog, a cool silver mid-tone, then bone. Three stops, linear in between. */
const STOPS: [number, number[]][] = [
  [0, [5, 10, 14]],
  [0.5, [104, 124, 134]],
  [1, [246, 249, 247]],
];
function exposure(t: number): number[] {
  const x = clamp01(t);
  const i = x <= STOPS[1][0] ? 0 : 1;
  const [a, ca] = STOPS[i];
  const [b, cb] = STOPS[i + 1];
  const u = (x - a) / (b - a);
  return ca.map((c, k) => Math.round(c + (cb[k] - c) * u));
}

let grainTile: HTMLCanvasElement | null = null;
function grain(): HTMLCanvasElement {
  if (grainTile) return grainTile;
  const c = document.createElement("canvas");
  c.width = c.height = 160;
  const g = c.getContext("2d")!;
  const img = g.createImageData(160, 160);
  let s = 1234567;
  for (let i = 0; i < img.data.length; i += 4) {
    s = (s * 1103515245 + 12345) >>> 0;
    const v = 110 + ((s >>> 16) % 90);
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  return (grainTile = c);
}

const SVG = "http://www.w3.org/2000/svg";

/** Four corner brackets around a box: a region of interest, as measurement tools draw it. */
function bracket(x: number, y: number, w: number, h: number): string {
  const k = Math.max(6, Math.min(14, w / 5, h / 2));
  const x2 = x + w;
  const y2 = y + h;
  return [
    `M${x},${y + k} V${y} H${x + k}`,
    `M${x2 - k},${y} H${x2} V${y + k}`,
    `M${x2},${y2 - k} V${y2} H${x2 - k}`,
    `M${x + k},${y2} H${x} V${y2 - k}`,
  ].join(" ");
}

export class Film {
  private canvas = document.createElement("canvas");
  private bloom = document.createElement("canvas");
  private svg = document.createElementNS(SVG, "svg");
  private tl = document.createElement("div");
  private tr = document.createElement("div");
  private bl = document.createElement("div");
  private tip = document.createElement("div");
  private cursor = document.createElement("div");
  private stage = document.createElement("div");
  private table = document.createElement("table");
  private spec: FilmSpec | null = null;
  private wl: { level: number; width: number } | null = null;
  private geom = { cell: 8, rowH: 8, gap: 2, rowGap: 5, padX: 20, padTop: 30, labelW: 0, labelH: 0 };
  private ro: ResizeObserver;
  private drag: { x: number; y: number; level: number; width: number } | null = null;
  private frame = 0;

  constructor(host: HTMLElement) {
    this.stage.className = "film";
    this.tl.className = "film-corner tl";
    this.tr.className = "film-corner tr";
    this.bl.className = "film-corner bl";
    this.tip.className = "film-corner br";
    this.tip.setAttribute("role", "status");
    this.cursor.className = "film-cursor";
    this.cursor.hidden = true;
    this.svg.classList.add("roi");
    this.svg.setAttribute("aria-hidden", "true");
    this.canvas.setAttribute("role", "img");
    this.table.className = "sr-only";
    this.stage.append(this.canvas, this.svg, this.cursor, this.tl, this.tr, this.bl, this.tip, this.table);
    host.prepend(this.stage);
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(this.stage);
    this.ro.observe(this.tl); // the plate title can re-wrap when the fonts arrive
    this.canvas.addEventListener("pointermove", (e) => this.hover(e));
    this.canvas.addEventListener("pointerleave", () => this.idle());
    this.canvas.addEventListener("pointerdown", (e) => this.startDrag(e));
    this.canvas.addEventListener("pointerup", () => (this.drag = null));
    this.canvas.addEventListener("pointercancel", () => (this.drag = null));
    this.canvas.addEventListener("dblclick", () => this.resetWindow());
    this.canvas.addEventListener("keydown", (e) => this.key(e));
  }

  show(spec: FilmSpec): void {
    const sameWindow = this.spec?.window && spec.window && this.spec.window.level === spec.window.level;
    this.spec = spec;
    if (!sameWindow) this.wl = spec.window ? { ...spec.window } : null;
    const [plate, kind, ...what] = spec.edge.split(/\s{2,}/);
    this.tl.innerHTML = "";
    const line1 = document.createElement("span");
    line1.className = "plate-no";
    line1.textContent = plate ?? "";
    const k = document.createElement("span");
    k.className = `kind kind-${(kind ?? "").toLowerCase().replace(/\s+/g, "-")}`;
    k.textContent = kind ?? "";
    const line2 = document.createElement("span");
    line2.className = "what";
    line2.textContent = what.join(", ");
    this.tl.append(line1, k, line2);
    this.bl.textContent = spec.corner ?? "";
    this.canvas.setAttribute("aria-label", spec.summary);
    if (spec.window) {
      this.canvas.tabIndex = 0;
      this.canvas.classList.add("windowed");
      this.canvas.setAttribute(
        "aria-roledescription",
        "film; drag or use the arrow keys to change window and level, double-click to reset",
      );
    } else {
      this.canvas.removeAttribute("tabindex");
      this.canvas.classList.remove("windowed");
      this.canvas.removeAttribute("aria-roledescription");
    }
    this.describe(spec);
    this.idle();
    this.draw();
  }

  /** A visually hidden table of each row's mean value, for screen readers (not for matrix films). */
  private describe(s: FilmSpec): void {
    if (s.square) {
      this.table.replaceChildren();
      return;
    }
    const head = `<tr><th scope="col">Row</th><th scope="col">Tokens</th><th scope="col">Mean weight, measured</th>${
      s.intended ? '<th scope="col">Mean weight, intended</th>' : ""
    }</tr>`;
    const body = s.rows
      .map((r, i) => {
        const label = s.rowLabels?.[i] ?? `Sequence ${i + 1}`;
        const want = s.intended ? `<td>${num(meanOf(s.intended[i]))}</td>` : "";
        return `<tr><th scope="row">${label}</th><td>${r.length}</td><td>${num(meanOf(r))}</td>${want}</tr>`;
      })
      .join("");
    this.table.innerHTML = `<caption>${s.edge.replace(/\s{2,}/g, ", ")}: mean weight per row</caption><thead>${head}</thead><tbody>${body}</tbody>`;
  }

  private lum(v: number): number {
    const s = this.spec!;
    if (this.wl) return clamp01((v - this.wl.level) / this.wl.width + 0.5);
    return clamp01(s.scale(v));
  }

  private windowText(): void {
    if (!this.wl) {
      this.tr.textContent = "";
      this.tr.hidden = true;
      return;
    }
    this.tr.hidden = false;
    this.tr.textContent = `W ${this.wl.width.toFixed(2)}  L ${this.wl.level.toFixed(2)}`;
  }

  private draw(): void {
    const s = this.spec;
    if (!s) return;
    const W = this.stage.clientWidth;
    if (!W) return;
    this.windowText();
    const cols = Math.max(...s.rows.map((r) => r.length));
    const narrow = W < 560;
    const notes = s.annotations ?? [];
    const noteMode: "none" | "gutter" | "strip" = !notes.length
      ? "none"
      : W >= 680 && !s.square
        ? "gutter"
        : "strip";
    const gutter = noteMode === "gutter" ? Math.min(260, Math.round(W * 0.26)) : 0;
    const labelW = s.rowLabels && !narrow ? Math.min(230, Math.round(W * 0.24)) : 0;
    const labelH = s.rowLabels && narrow ? 16 : 0;
    const padX = narrow ? 14 : 28;
    // the corner print sets the margins: the grid starts below the plate's title, however it wraps
    const small = W < 600;
    this.tip.hidden = small && this.tip.classList.contains("hint");
    this.bl.hidden = !s.corner || small;
    const padTop = Math.max(narrow ? 60 : 70, this.tl.offsetHeight + 36);
    const padBottom = this.tip.hidden && this.bl.hidden ? 22 : narrow ? 36 : 44;
    const avail = W - padX * 2 - labelW - gutter;
    const gap = s.square ? 4 : cols > 60 ? 1 : 2;
    let cell = Math.max(2, Math.floor((avail - gap * (cols - 1)) / cols));
    if (s.square) cell = Math.min(cell, 56);
    else cell = Math.min(cell, 34);
    const rowH = s.square
      ? cell
      : Math.max(s.rowLabels && !narrow ? 14 : 9, Math.min(22, Math.round(cell * 0.9)));
    const rowGap = s.square ? gap : Math.max(4, Math.round(rowH * 0.32));
    const gridH = s.rows.length * (rowH + labelH) + (s.rows.length - 1) * rowGap;
    const stripH = noteMode === "strip" ? notes.length * 24 + 14 : 0;
    const H = padTop + gridH + stripH + padBottom;
    this.geom = { cell, rowH, gap, rowGap, padX, padTop, labelW, labelH };

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    for (const c of [this.canvas, this.bloom]) {
      c.width = Math.round(W * dpr);
      c.height = Math.round(H * dpr);
    }
    this.canvas.style.width = `${W}px`;
    this.canvas.style.height = `${H}px`;
    const g = this.canvas.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);

    // film base: deep fog with a faint backlight in the middle
    g.fillStyle = "rgb(5,10,14)";
    g.fillRect(0, 0, W, H);
    const back = g.createRadialGradient(W * 0.45, H * 0.45, 0, W * 0.45, H * 0.45, Math.max(W, H) * 0.75);
    back.addColorStop(0, "rgba(70,100,115,0.16)");
    back.addColorStop(1, "rgba(70,100,115,0)");
    g.fillStyle = back;
    g.fillRect(0, 0, W, H);

    const rowY = (r: number) => padTop + r * (rowH + rowGap + labelH) + labelH;
    const colX = (c: number) => padX + labelW + c * (cell + gap);

    // exposure: cells drawn once to an offscreen layer, then composited with a soft halation
    const b = this.bloom.getContext("2d")!;
    b.setTransform(dpr, 0, 0, dpr, 0, 0);
    b.clearRect(0, 0, W, H);
    const radius = Math.min(2.5, cell / 4, rowH / 4);
    for (let r = 0; r < s.rows.length; r++) {
      s.rows[r].forEach((v, c) => {
        const t = this.lum(v);
        b.fillStyle = `rgb(${exposure(t).join(",")})`;
        b.beginPath();
        b.roundRect(colX(c), rowY(r), cell, rowH, radius);
        b.fill();
      });
    }
    // halation: a blurred copy of the exposure added on top (downsample, then upsample: works everywhere)
    const low = document.createElement("canvas");
    const f = 8;
    low.width = Math.max(1, Math.round((W * dpr) / f));
    low.height = Math.max(1, Math.round((H * dpr) / f));
    const sg = low.getContext("2d")!;
    sg.imageSmoothingEnabled = true;
    sg.drawImage(this.bloom, 0, 0, low.width, low.height);
    g.save();
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.imageSmoothingEnabled = true;
    g.globalCompositeOperation = "lighter";
    g.globalAlpha = cell < 14 ? 0.16 : 0.3;
    g.drawImage(low, 0, 0, this.canvas.width, this.canvas.height);
    g.globalAlpha = 1;
    g.globalCompositeOperation = "source-over";
    g.drawImage(this.bloom, 0, 0);
    g.restore();

    // grain and vignette: the film, not a chart
    g.save();
    g.globalAlpha = 0.07;
    g.globalCompositeOperation = "overlay";
    g.fillStyle = g.createPattern(grain(), "repeat")!;
    g.fillRect(0, 0, W, H);
    g.restore();
    const vig = g.createRadialGradient(
      W / 2,
      H / 2,
      Math.min(W, H) * 0.35,
      W / 2,
      H / 2,
      Math.max(W, H) * 0.72,
    );
    vig.addColorStop(0, "rgba(0,0,0,0)");
    vig.addColorStop(1, "rgba(0,0,0,0.42)");
    g.fillStyle = vig;
    g.fillRect(0, 0, W, H);

    // micro-batch boundaries: a hairline across the film, with a tick in the margin
    g.strokeStyle = "rgba(220,232,236,0.16)";
    g.lineWidth = 1;
    g.setLineDash([2, 4]);
    for (const r of s.separators ?? []) {
      const y = Math.round(rowY(r) - labelH - rowGap / 2) + 0.5;
      g.beginPath();
      g.moveTo(padX + labelW - 6, y);
      g.lineTo(W - padX - gutter, y);
      g.stroke();
    }
    g.setLineDash([]);

    // row labels in the scanner's type
    if (s.rowLabels) {
      g.fillStyle = "rgba(214,226,231,0.72)";
      g.font = "500 12.5px 'IBM Plex Sans Condensed', system-ui, sans-serif";
      for (let r = 0; r < s.rows.length; r++) {
        if (narrow) {
          g.textBaseline = "alphabetic";
          g.fillText(s.rowLabels[r], padX, rowY(r) - 4);
        } else {
          g.textBaseline = "middle";
          g.fillText(s.rowLabels[r], padX, rowY(r) + rowH / 2, labelW - 14);
        }
      }
    }

    // regions of interest
    this.svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    this.svg.setAttribute("width", String(W));
    this.svg.setAttribute("height", String(H));
    this.svg.replaceChildren();
    const text = (x: number, y: number, str: string, cls: string) => {
      const t = document.createElementNS(SVG, "text");
      t.setAttribute("class", cls);
      t.setAttribute("x", String(x));
      t.setAttribute("y", String(y));
      t.textContent = str;
      this.svg.append(t);
      return t;
    };
    const stripY = (k: number) => padTop + gridH + 30 + k * 24;
    notes.forEach((a, k) => {
      const tag = notes.length > 1 ? `${k + 1}` : "";
      if (!a.rows) {
        if (noteMode === "gutter") {
          text(W - gutter + 4, padTop + gridH / 2 + 5, a.note, "roi-note");
        } else {
          text(padX, stripY(k), a.note, "roi-note");
        }
        return;
      }
      const [r0, r1] = a.rows;
      const c0 = a.cols ? a.cols[0] : 0;
      const c1 = a.cols ? a.cols[1] : Math.max(...s.rows.slice(r0, r1 + 1).map((row) => row.length)) - 1;
      const pad = 6;
      const x = colX(c0) - pad;
      const y = rowY(r0) - pad;
      const w = colX(c1) + cell - colX(c0) + pad * 2;
      const h = rowY(r1) + rowH - rowY(r0) + pad * 2;
      const box = document.createElementNS(SVG, "rect");
      box.setAttribute("class", "roi-fill");
      box.setAttribute("x", String(x));
      box.setAttribute("y", String(y));
      box.setAttribute("width", String(w));
      box.setAttribute("height", String(h));
      this.svg.append(box, this.path(bracket(x, y, w, h), "roi-line"));
      if (noteMode === "gutter") {
        const ty = y + h / 2;
        const tx = W - gutter + 4;
        this.svg.append(this.path(`M${x + w + 6},${ty} H${tx - 10}`, "roi-line roi-lead"));
        if (tag) text(tx, ty - 9, `ROI ${tag}`, "roi-tag");
        text(tx, ty + (tag ? 9 : 5), a.note, "roi-note");
      } else {
        if (tag) text(Math.min(x + w + 4, W - 14), y + 11, tag, "roi-tag");
        text(padX, stripY(k), tag ? `${tag}   ${a.note}` : a.note, "roi-note");
      }
    });
  }

  private path(d: string, cls: string): SVGPathElement {
    const p = document.createElementNS(SVG, "path");
    p.setAttribute("d", d);
    p.setAttribute("class", cls);
    p.setAttribute("pathLength", "1");
    return p;
  }

  private cellAt(e: PointerEvent): { r: number; c: number; v: number } | null {
    const s = this.spec;
    if (!s) return null;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const { cell, rowH, gap, rowGap, padX, padTop, labelW, labelH } = this.geom;
    const r = Math.floor((y - padTop) / (rowH + rowGap + labelH));
    const yIn = y - padTop - r * (rowH + rowGap + labelH) - labelH;
    const c = Math.floor((x - padX - labelW) / (cell + gap));
    const v = s.rows[r]?.[c];
    if (v === undefined || yIn < 0 || yIn > rowH) return null;
    return { r, c, v };
  }

  private idle(): void {
    this.cursor.hidden = true;
    const s = this.spec;
    if (!s) return;
    this.tip.textContent = s.window
      ? "Drag across the film to change window and level"
      : s.tooltip
        ? "Point at a cell to read it"
        : "";
    this.tip.classList.add("hint");
    this.tip.hidden = this.stage.clientWidth < 600;
  }

  private hover(e: PointerEvent): void {
    if (this.drag) {
      this.moveDrag(e);
      return;
    }
    const s = this.spec;
    const hit = this.cellAt(e);
    if (!s?.tooltip || !hit) {
      this.idle();
      return;
    }
    const { cell, rowH, gap, rowGap, padX, padTop, labelW, labelH } = this.geom;
    this.cursor.hidden = false;
    this.cursor.style.left = `${padX + labelW + hit.c * (cell + gap) - 2}px`;
    this.cursor.style.top = `${padTop + hit.r * (rowH + rowGap + labelH) + labelH - 2}px`;
    this.cursor.style.width = `${cell + 4}px`;
    this.cursor.style.height = `${rowH + 4}px`;
    this.tip.classList.remove("hint");
    this.tip.hidden = false;
    this.tip.textContent = s.tooltip(hit.r, hit.c, hit.v);
  }

  private startDrag(e: PointerEvent): void {
    if (!this.wl) return;
    this.drag = { x: e.clientX, y: e.clientY, ...this.wl };
    this.canvas.setPointerCapture(e.pointerId);
  }

  private moveDrag(e: PointerEvent): void {
    const d = this.drag!;
    const width = Math.max(0.02, Math.min(8, d.width * Math.exp((e.clientX - d.x) / 160)));
    const level = d.level - ((e.clientY - d.y) / 240) * d.width;
    this.setWindow(level, width);
  }

  private key(e: KeyboardEvent): void {
    if (!this.wl) return;
    const { level, width } = this.wl;
    const step = {
      ArrowRight: [0, 1.15],
      ArrowLeft: [0, 1 / 1.15],
      ArrowUp: [0.05, 1],
      ArrowDown: [-0.05, 1],
    }[e.key];
    if (step) {
      e.preventDefault();
      this.setWindow(level + step[0] * width, Math.max(0.02, width * step[1]));
    } else if (e.key === "Home") {
      e.preventDefault();
      this.resetWindow();
    }
  }

  private setWindow(level: number, width: number): void {
    this.wl = { level, width };
    cancelAnimationFrame(this.frame);
    this.frame = requestAnimationFrame(() => this.draw());
  }

  private resetWindow(): void {
    if (this.spec?.window) this.setWindow(this.spec.window.level, this.spec.window.width);
  }

  /** Replay the measurement strokes (used when a scrolly step becomes active). */
  redraw(): void {
    this.svg.querySelectorAll(".roi-line").forEach((p) => {
      p.classList.remove("drawn");
      void (p as SVGPathElement).getBBox();
      p.classList.add("drawn");
    });
  }
}
