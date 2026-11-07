import type { SkyLayout } from "../atlas/layout";
import { FAMILY_LABEL, type Edge } from "../atlas/schema";
import type { SkyCallbacks } from "../scene/sky";

/**
 * The 2D constellation: the same lineage disc seen from above, as SVG. Used on narrow screens
 * and when WebGL is unavailable. Same surface API as the 3D sky, so navigation is identical.
 * Drag pans, pinch or wheel zooms, tap selects.
 */
const NS = "http://www.w3.org/2000/svg";

interface View {
  cx: number;
  cy: number;
  span: number;
}

export class Constellation2D {
  private svg: SVGSVGElement;
  private view: View;
  private home: View;
  private size = { w: 1, h: 1 };
  private ro: ResizeObserver;
  private anim = 0;
  private starEls = new Map<string, SVGGElement>();
  private edgeEls: SVGPathElement[] = [];
  private hoverId: string | null = null;
  private selectedId: string | null = null;
  private pathIds: string[] = [];
  private pos = new Map<string, { x: number; y: number }>();
  readonly stats = { cpuMs: 0, calls: 0 };
  private disposers: (() => void)[] = [];

  constructor(
    private readonly el: HTMLElement,
    private readonly layout: SkyLayout,
    private readonly edges: Edge[],
    private readonly cb: SkyCallbacks,
    private readonly reducedMotion: boolean,
  ) {
    for (const s of layout.stars) this.pos.set(s.id, { x: s.x, y: s.z });
    this.svg = document.createElementNS(NS, "svg");
    this.svg.classList.add("constellation");
    this.svg.setAttribute("role", "img");
    this.svg.setAttribute(
      "aria-label",
      "Constellation of formulas: rings are generations, sectors are families",
    );
    el.append(this.svg);
    this.build();
    const R = layout.outerRadius + 4;
    this.home = { cx: 0, cy: 0, span: 2 * R };
    this.view = { ...this.home };
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(el);
    this.resize();
    this.bindInput();
  }

  private node<K extends keyof SVGElementTagNameMap>(
    tag: K,
    attrs: Record<string, string | number>,
    parent: Element,
  ) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    parent.append(n);
    return n;
  }

  private build(): void {
    const defs = this.node("defs", {}, this.svg);
    const grad = this.node("radialGradient", { id: "c2d-glow" }, defs);
    this.node("stop", { offset: "0", "stop-color": "#dde6e8", "stop-opacity": "0.55" }, grad);
    this.node("stop", { offset: "1", "stop-color": "#dde6e8", "stop-opacity": "0" }, grad);
    const gradHub = this.node("radialGradient", { id: "c2d-glow-hub" }, defs);
    this.node("stop", { offset: "0", "stop-color": "#5ee3d6", "stop-opacity": "0.7" }, gradHub);
    this.node("stop", { offset: "1", "stop-color": "#5ee3d6", "stop-opacity": "0" }, gradHub);

    const g = this.node("g", {}, this.svg);
    for (const ring of this.layout.rings)
      this.node("circle", { class: "c2d-ring", cx: 0, cy: 0, r: ring.radius }, g);
    for (const ring of this.layout.rings) {
      const t = this.node("text", { class: "c2d-gen", x: 0, y: -ring.radius - 0.35 }, g);
      t.textContent = String(ring.generation);
    }

    // family names along arcs at each sector's outer edge, upright on both halves of the disc
    this.layout.sectors.forEach((s, i) => {
      const r = s.edgeRadius + 0.6;
      const mid = (s.start + s.end) / 2;
      const lower = Math.sin(mid) > 0;
      const a0 = lower ? s.end : s.start;
      const a1 = lower ? s.start : s.end;
      const id = `c2d-arc-${i}`;
      this.node(
        "path",
        {
          id,
          d: `M ${r * Math.cos(a0)} ${r * Math.sin(a0)} A ${r} ${r} 0 0 ${lower ? 0 : 1} ${r * Math.cos(a1)} ${r * Math.sin(a1)}`,
          fill: "none",
        },
        defs,
      );
      const text = this.node("text", { class: "c2d-family", dy: lower ? "1.2" : "0" }, g);
      const tp = this.node("textPath", { href: `#${id}`, startOffset: "50%", "text-anchor": "middle" }, text);
      tp.textContent = FAMILY_LABEL[s.family].toUpperCase();
    });

    const eg = this.node("g", { class: "c2d-edges" }, g);
    for (const e of this.edges) {
      const a = this.pos.get(e.from)!;
      const b = this.pos.get(e.to)!;
      // bow each filament slightly toward the centre so overlapping edges separate
      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2;
      const cx = mx * 0.94;
      const cy = my * 0.94;
      const p = this.node(
        "path",
        { d: `M ${a.x} ${a.y} Q ${cx} ${cy} ${b.x} ${b.y}`, class: `c2d-edge ${e.kind}` },
        eg,
      );
      this.edgeEls.push(p);
    }

    const sg = this.node("g", { class: "c2d-stars" }, g);
    for (const s of this.layout.stars) {
      const p = this.pos.get(s.id)!;
      const star = this.node(
        "g",
        { class: `c2d-star${s.hub ? " hub" : ""}`, transform: `translate(${p.x} ${p.y})` },
        sg,
      );
      this.node(
        "circle",
        { r: s.hub ? 2.2 : 1.25, fill: `url(#${s.hub ? "c2d-glow-hub" : "c2d-glow"})` },
        star,
      );
      this.node("circle", { class: "c2d-core", r: s.hub ? 0.42 : 0.26 }, star);
      this.node("circle", { class: "c2d-sel", r: 0.75 }, star);
      this.starEls.set(s.id, star);
    }
  }

  // ---------- view ----------

  private applyView(): void {
    const { w, h } = this.size;
    const { cx, cy, span } = this.view;
    // span covers the shorter side
    const sx = w >= h ? (span * w) / h : span;
    const sy = w >= h ? span : (span * h) / w;
    this.svg.setAttribute("viewBox", `${cx - sx / 2} ${cy - sy / 2} ${sx} ${sy}`);
    // close in, the lettering would loom over the stars; let it recede
    this.svg.classList.toggle("is-zoomed", span < this.home.span * 0.6);
  }

  private resize(): void {
    this.size = { w: this.el.clientWidth || 1, h: this.el.clientHeight || 1 };
    this.applyView();
  }

  private tween(to: View, dur: number): Promise<void> {
    cancelAnimationFrame(this.anim);
    if (this.reducedMotion || dur <= 0) {
      this.view = to;
      this.applyView();
      this.cb.frame();
      return Promise.resolve();
    }
    const from = { ...this.view };
    const t0 = performance.now();
    return new Promise((done) => {
      const step = () => {
        const k = Math.min((performance.now() - t0) / dur, 1);
        const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
        this.view = {
          cx: from.cx + (to.cx - from.cx) * e,
          cy: from.cy + (to.cy - from.cy) * e,
          span: from.span * Math.pow(to.span / from.span, e),
        };
        this.applyView();
        this.cb.frame();
        if (k < 1) this.anim = requestAnimationFrame(step);
        else done();
      };
      this.anim = requestAnimationFrame(step);
    });
  }

  overview(dur = 900): Promise<void> {
    return this.tween({ ...this.home }, dur);
  }

  /** Centres a star in the upper part of the view, leaving the bottom for the card. */
  focus(id: string, _shift = 0, dur = 900): Promise<void> {
    const p = this.pos.get(id)!;
    const span = 16;
    const { w, h } = this.size;
    const sy = w >= h ? span : (span * h) / w;
    return this.tween({ cx: p.x, cy: p.y + sy * 0.2, span }, dur);
  }

  jumpTo(id: string, shift: number): void {
    void this.focus(id, shift, 0);
  }

  // ---------- state ----------

  setHover(id: string | null): void {
    this.hoverId = id;
    this.paint();
  }
  setSelected(id: string | null): void {
    this.selectedId = id;
    this.paint();
  }
  setPath(ids: string[]): void {
    this.pathIds = ids;
    this.paint();
  }

  private paint(): void {
    const path = new Set(this.pathIds);
    for (const [id, el] of this.starEls) {
      el.classList.toggle("is-hover", id === this.hoverId);
      el.classList.toggle("is-selected", id === this.selectedId);
      el.classList.toggle("is-path", path.has(id));
    }
    this.svg.classList.toggle("has-selection", !!this.selectedId);
    const pairs = new Set<string>();
    for (let i = 1; i < this.pathIds.length; i++) pairs.add(`${this.pathIds[i - 1]}>${this.pathIds[i]}`);
    const focus = this.selectedId ?? this.hoverId;
    this.edges.forEach((e, i) => {
      const lit =
        pairs.has(`${e.from}>${e.to}`) ||
        (!this.pathIds.length && !!focus && (e.from === focus || e.to === focus));
      this.edgeEls[i].classList.toggle("is-lit", lit);
    });
  }

  // ---------- projection & input ----------

  project(id: string): { x: number; y: number; visible: boolean } {
    const p = this.pos.get(id)!;
    const ctm = this.svg.getScreenCTM();
    const r = this.svg.getBoundingClientRect();
    if (!ctm) return { x: 0, y: 0, visible: false };
    const x = ctm.a * p.x + ctm.c * p.y + ctm.e - r.left;
    const y = ctm.b * p.x + ctm.d * p.y + ctm.f - r.top;
    return { x, y, visible: x >= 0 && y >= 0 && x <= r.width && y <= r.height };
  }

  private pick(x: number, y: number): string | null {
    let best: string | null = null;
    let bestD = 26;
    for (const s of this.layout.stars) {
      const p = this.project(s.id);
      const d = Math.hypot(p.x - x, p.y - y);
      if (d < bestD) {
        best = s.id;
        bestD = d;
      }
    }
    return best;
  }

  private bindInput(): void {
    const svg = this.svg;
    svg.style.touchAction = "none";
    const pointers = new Map<number, { x: number; y: number }>();
    let start: { view: View; x: number; y: number; dist: number; moved: boolean } | null = null;
    const local = (e: PointerEvent) => {
      const r = svg.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    const centroid = () => {
      const ps = [...pointers.values()];
      return {
        x: ps.reduce((a, p) => a + p.x, 0) / ps.length,
        y: ps.reduce((a, p) => a + p.y, 0) / ps.length,
      };
    };
    const spread = () => {
      const ps = [...pointers.values()];
      return ps.length < 2 ? 0 : Math.hypot(ps[0].x - ps[1].x, ps[0].y - ps[1].y);
    };
    const reset = () => {
      const c = centroid();
      start = { view: { ...this.view }, x: c.x, y: c.y, dist: spread(), moved: start?.moved ?? false };
    };
    const down = (e: PointerEvent) => {
      svg.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, local(e));
      cancelAnimationFrame(this.anim);
      reset();
      if (pointers.size === 1) start!.moved = false;
    };
    const move = (e: PointerEvent) => {
      if (!pointers.has(e.pointerId) || !start) return;
      pointers.set(e.pointerId, local(e));
      const c = centroid();
      if (Math.hypot(c.x - start.x, c.y - start.y) > 6 || pointers.size > 1) start.moved = true;
      if (!start.moved) return;
      const k = pointers.size > 1 && start.dist > 0 ? start.dist / spread() : 1;
      const span = Math.min(Math.max(start.view.span * k, 8), this.home.span * 1.4);
      const u = start.view.span / Math.min(this.size.w, this.size.h);
      this.view = { cx: start.view.cx - (c.x - start.x) * u, cy: start.view.cy - (c.y - start.y) * u, span };
      this.applyView();
      this.cb.frame();
    };
    const up = (e: PointerEvent) => {
      if (!pointers.has(e.pointerId)) return;
      const p = local(e);
      pointers.delete(e.pointerId);
      const wasTap = start && !start.moved && pointers.size === 0;
      if (pointers.size) reset();
      else start = null;
      if (!wasTap) return;
      const id = this.pick(p.x, p.y);
      if (id) this.cb.select(id);
      else this.cb.background();
    };
    const hover = (e: PointerEvent) => {
      if (e.pointerType !== "mouse" || pointers.size) return;
      const p = local(e);
      const id = this.pick(p.x, p.y);
      svg.style.cursor = id ? "pointer" : "grab";
      if (id !== this.hoverId) this.cb.hover(id);
    };
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const span = Math.min(Math.max(this.view.span * Math.exp(e.deltaY * 0.0012), 8), this.home.span * 1.4);
      this.view = { ...this.view, span };
      this.applyView();
      this.cb.frame();
    };
    svg.addEventListener("pointerdown", down);
    svg.addEventListener("pointermove", move);
    svg.addEventListener("pointermove", hover);
    svg.addEventListener("pointerup", up);
    svg.addEventListener("pointercancel", up);
    svg.addEventListener("wheel", wheel, { passive: false });
    this.disposers.push(() => {
      svg.removeEventListener("pointerdown", down);
      svg.removeEventListener("pointermove", move);
      svg.removeEventListener("pointermove", hover);
      svg.removeEventListener("pointerup", up);
      svg.removeEventListener("pointercancel", up);
      svg.removeEventListener("wheel", wheel);
    });
  }

  start(): void {}
  stop(): void {
    cancelAnimationFrame(this.anim);
  }
  dispose(): void {
    this.stop();
    this.ro.disconnect();
    this.disposers.forEach((d) => d());
    this.svg.remove();
  }
}
