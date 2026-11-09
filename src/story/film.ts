/**
 * Radiograph renderer. A film is a grid of cells (tokens, or documents for leak matrices) drawn as
 * luminance on a blue-black base, with grease-pencil annotations in an SVG overlay. Pure view code:
 * every value it shows comes from measured data or src/core.
 */
export interface Annotation {
  rows?: [number, number]; // inclusive row range to circle; omit for a film-wide note
  cols?: [number, number]; // optional column range; default: the whole drawn width of those rows
  note: string;
}

export interface FilmSpec {
  rows: number[][]; // luminance source values, one array per row (ragged allowed)
  scale: (v: number) => number; // value -> luminance in [0, 1]
  rowLabels?: string[];
  edge: string; // the text printed along the film edge
  annotations?: Annotation[];
  separators?: number[]; // draw a boundary tick before these row indices (micro-batch edges)
  tooltip?: (r: number, c: number, v: number) => string;
  square?: boolean; // matrix films: square cells
  summary: string; // accessible description
  intended?: number[][]; // what the equation says, per cell (adds a column to the screen-reader table)
}

const meanOf = (r: number[]) => r.reduce((a, b) => a + b, 0) / r.length;
const num = (x: number) => (x !== 0 && Math.abs(x) < 0.01 ? x.toExponential(2) : x.toFixed(3));

const FILM = [13, 25, 34];
const BONE = [238, 243, 241];

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A hand-drawn closed loop around a box, deterministic per seed. */
function pencilLoop(x: number, y: number, w: number, h: number, seed: number): string {
  const rnd = mulberry(seed);
  const cx = x + w / 2;
  const cy = y + h / 2;
  const rx = w / 2 + 10;
  const ry = h / 2 + 9;
  const pts: [number, number][] = [];
  const n = 14;
  const start = rnd() * Math.PI * 2;
  for (let i = 0; i <= n + 2; i++) {
    const a = start + (i / n) * Math.PI * 2.12;
    const j = 1 + (rnd() - 0.5) * 0.08;
    pts.push([cx + Math.cos(a) * rx * j, cy + Math.sin(a) * ry * j]);
  }
  let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[i + 1];
    d += ` Q${x1.toFixed(1)},${y1.toFixed(1)} ${((x1 + x2) / 2).toFixed(1)},${((y1 + y2) / 2).toFixed(1)}`;
  }
  return d;
}

export class Film {
  private canvas = document.createElement("canvas");
  private svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  private edge = document.createElement("div");
  private tip = document.createElement("div");
  private table = document.createElement("table");
  private stage = document.createElement("div");
  private spec: FilmSpec | null = null;
  private geom = { cell: 8, rowH: 8, gap: 2, padX: 20, padY: 30, labelW: 0 };
  private ro: ResizeObserver;

  constructor(host: HTMLElement) {
    this.stage.className = "film";
    this.edge.className = "film-edge";
    this.tip.className = "film-tip";
    this.tip.setAttribute("role", "status");
    this.svg.classList.add("pencil");
    this.svg.setAttribute("aria-hidden", "true");
    this.canvas.setAttribute("role", "img");
    this.table.className = "sr-only";
    this.stage.append(this.edge, this.canvas, this.svg, this.tip, this.table);
    host.prepend(this.stage);
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(this.stage);
    this.canvas.addEventListener("pointermove", (e) => this.hover(e));
    this.canvas.addEventListener("pointerleave", () => (this.tip.hidden = true));
    this.tip.hidden = true;
  }

  show(spec: FilmSpec): void {
    this.spec = spec;
    this.edge.textContent = spec.edge;
    this.canvas.setAttribute("aria-label", spec.summary);
    this.describe(spec);
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

  private draw(): void {
    const s = this.spec;
    if (!s) return;
    const W = this.stage.clientWidth;
    if (!W) return;
    const cols = Math.max(...s.rows.map((r) => r.length));
    const narrow = W < 520;
    const notes = s.annotations ?? [];
    const noteMode: "none" | "gutter" | "strip" = !notes.length
      ? "none"
      : W >= 640 && !s.square
        ? "gutter"
        : "strip";
    const gutter = noteMode === "gutter" ? Math.min(250, Math.round(W * 0.27)) : 0;
    const labelW = s.rowLabels ? (narrow ? 0 : 200) : 0;
    const padX = narrow ? 12 : 22;
    const padY = 34;
    const avail = W - padX * 2 - labelW - gutter;
    const gap = s.square ? 4 : cols > 60 ? 1 : 2;
    let cell = Math.max(2, Math.floor((avail - gap * (cols - 1)) / cols));
    if (s.square) cell = Math.min(cell, 54);
    const rowH = s.square ? cell : Math.max(8, Math.min(18, cell * 1.6));
    const rowGap = s.square ? gap : 5;
    const gridH =
      s.rows.length * rowH + (s.rows.length - 1) * rowGap + (s.rowLabels && narrow ? s.rows.length * 14 : 0);
    const stripH = noteMode === "strip" ? notes.length * 30 + 8 : 0;
    const H = padY * 2 + gridH + stripH;
    this.geom = { cell, rowH, gap, padX, padY, labelW };

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(W * dpr);
    this.canvas.height = Math.round(H * dpr);
    this.canvas.style.width = `${W}px`;
    this.canvas.style.height = `${H}px`;
    const g = this.canvas.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);

    // film base with a faint vertical exposure gradient and grain
    const base = g.createLinearGradient(0, 0, 0, H);
    base.addColorStop(0, "rgb(16,30,40)");
    base.addColorStop(1, `rgb(${FILM.join(",")})`);
    g.fillStyle = base;
    g.fillRect(0, 0, W, H);
    const rnd = mulberry(7);
    for (let i = 0; i < (W * H) / 90; i++) {
      g.fillStyle = `rgba(200,220,230,${(rnd() * 0.035).toFixed(3)})`;
      g.fillRect(rnd() * W, rnd() * H, 1, 1);
    }

    const rowY = (r: number) => padY + r * (rowH + rowGap) + (s.rowLabels && narrow ? (r + 1) * 14 : 0);
    const colX = (c: number) => padX + labelW + c * (cell + gap);

    // micro-batch boundary ticks
    g.strokeStyle = "rgba(238,243,241,0.35)";
    g.lineWidth = 1;
    for (const r of s.separators ?? []) {
      const y = rowY(r) - rowGap / 2;
      g.beginPath();
      g.moveTo(padX - 10, y);
      g.lineTo(padX - 2, y);
      g.stroke();
    }

    for (let r = 0; r < s.rows.length; r++) {
      if (s.rowLabels) {
        g.fillStyle = "rgba(238,243,241,0.62)";
        g.font = "500 12.5px 'Atkinson Hyperlegible Next', system-ui, sans-serif";
        g.textBaseline = narrow ? "alphabetic" : "middle";
        if (narrow) g.fillText(s.rowLabels[r], padX, rowY(r) - 4);
        else g.fillText(s.rowLabels[r], padX, rowY(r) + rowH / 2);
      }
      s.rows[r].forEach((v, c) => {
        const lum = Math.max(0, Math.min(1, s.scale(v)));
        const t = Math.pow(lum, 0.85);
        const col = FILM.map((f, k) => Math.round(f + (BONE[k] - f) * t));
        g.shadowColor = `rgba(${BONE.join(",")},${(0.55 * t).toFixed(3)})`;
        g.shadowBlur = 7 * t;
        g.fillStyle = `rgb(${col.join(",")})`;
        g.fillRect(colX(c), rowY(r), cell, rowH);
      });
      g.shadowBlur = 0;
    }

    // grease pencil
    this.svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    this.svg.setAttribute("width", String(W));
    this.svg.setAttribute("height", String(H));
    this.svg.replaceChildren();
    notes.forEach((a, k) => {
      if (!a.rows) {
        const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
        t.setAttribute("class", "pencil-note");
        t.setAttribute("x", String(noteMode === "gutter" ? W - gutter + 14 : padX));
        t.setAttribute("y", String(noteMode === "gutter" ? padY + gridH / 2 : padY + gridH + 26 + k * 30));
        t.textContent = a.note;
        this.svg.append(t);
        return;
      }
      const [r0, r1] = a.rows;
      const c0 = a.cols ? a.cols[0] : 0;
      const c1 = a.cols ? a.cols[1] : Math.max(...s.rows.slice(r0, r1 + 1).map((row) => row.length)) - 1;
      const x = colX(c0);
      const y = rowY(r0);
      const w = colX(c1) + cell - x;
      const h = rowY(r1) + rowH - y;
      this.svg.append(this.path(pencilLoop(x, y, w, h, 11 + k * 7), "pencil-line"));
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("class", "pencil-note");
      let tx: number;
      let ty: number;
      let lead: string;
      if (noteMode === "gutter") {
        tx = W - gutter + 14;
        ty = y + h / 2 + 7;
        lead = `M${(x + w + 12).toFixed(1)},${(y + h / 2).toFixed(1)} L${(tx - 8).toFixed(1)},${(y + h / 2 + 1).toFixed(1)}`;
      } else {
        // narrow films: number the loop, write the note in the strip below the film
        tx = padX;
        ty = padY + gridH + 26 + k * 30;
        lead = "";
        const num = document.createElementNS("http://www.w3.org/2000/svg", "text");
        num.setAttribute("class", "pencil-note");
        num.setAttribute("x", String(Math.min(x + w + 12, W - 16)));
        num.setAttribute("y", String(y + 6));
        num.textContent = String(k + 1);
        if (notes.length > 1) this.svg.append(num);
      }
      if (lead) this.svg.append(this.path(lead, "pencil-line pencil-lead"));
      text.setAttribute("x", String(tx));
      text.setAttribute("y", String(ty));
      text.textContent = noteMode === "strip" && notes.length > 1 ? `${k + 1}  ${a.note}` : a.note;
      this.svg.append(text);
    });
  }

  private path(d: string, cls: string): SVGPathElement {
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", d);
    p.setAttribute("class", cls);
    p.setAttribute("pathLength", "1");
    return p;
  }

  private hover(e: PointerEvent): void {
    const s = this.spec;
    if (!s?.tooltip) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const { cell, rowH, gap, padX, padY, labelW } = this.geom;
    const narrow = rect.width < 520;
    const rowGap = s.square ? gap : 5;
    const extra = s.rowLabels && narrow ? 14 : 0;
    const r = Math.floor((y - padY) / (rowH + rowGap + extra));
    const c = Math.floor((x - padX - labelW) / (cell + gap));
    const v = s.rows[r]?.[c];
    if (v === undefined) {
      this.tip.hidden = true;
      return;
    }
    this.tip.textContent = s.tooltip(r, c, v);
    this.tip.hidden = false;
    this.tip.style.left = `${Math.min(x + 14, rect.width - 220)}px`;
    this.tip.style.top = `${y + 18}px`;
  }

  /** Replay the pencil strokes (used when a scrolly step becomes active). */
  redraw(): void {
    this.svg.querySelectorAll(".pencil-line").forEach((p) => {
      p.classList.remove("drawn");
      void (p as SVGPathElement).getBBox();
      p.classList.add("drawn");
    });
  }
}
