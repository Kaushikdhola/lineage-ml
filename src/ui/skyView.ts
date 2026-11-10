import { EDGES, FORMULAS } from "../atlas/formulas";
import { childrenOf, getFormula, parentsOf, pathBetween } from "../atlas/graph";
import { computeLayout, type SkyLayout } from "../atlas/layout";
import { FAMILY_LABEL } from "../atlas/schema";
import { renderTex } from "../math/katex";
import { FormulaStage } from "../math/morph";
import { Sky, type SkyCallbacks } from "../scene/sky";
import { Constellation2D } from "./constellation2d";
import { esc, isMac, prefersReducedMotion } from "./dom";

export interface SkyHooks {
  /** Lift the selected formula into the lab. `from` is the card's rendered formula. */
  open(id: string, from: HTMLElement): void;
  search(): void;
}

type Surface = Sky | Constellation2D;

export const TOUR = { from: "dot-product", to: "multi-head-attention" };

function webglAvailable(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") ?? c.getContext("webgl"));
  } catch {
    return false;
  }
}

export function wants2D(): boolean {
  const forced = new URLSearchParams(location.search).get("sky");
  if (forced === "2d") return true;
  if (forced === "3d") return false;
  return window.matchMedia("(max-width: 720px)").matches || !webglAvailable();
}

const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      clearTimeout(t);
      reject(new DOMException("tour stopped", "AbortError"));
    });
  });

export class SkyView {
  readonly root: HTMLElement;
  readonly mode: "3d" | "2d";
  private surface: Surface;
  private layout: SkyLayout;
  private stage: FormulaStage;
  private hoverId: string | null = null;
  private selected: string | null = null;
  private tour: AbortController | null = null;
  private visible = false;
  private reduced = prefersReducedMotion();
  private gen = new Map<string, number>();
  private measure: HTMLElement;
  private $ = <T extends HTMLElement>(sel: string) => this.root.querySelector<T>(sel)!;

  constructor(
    host: HTMLElement,
    private readonly hooks: SkyHooks,
  ) {
    this.mode = wants2D() ? "2d" : "3d";
    this.layout = computeLayout(FORMULAS, EDGES);
    for (const s of this.layout.stars) this.gen.set(s.id, s.generation);
    const families = new Set(FORMULAS.map((f) => f.family)).size;

    this.root = document.createElement("section");
    this.root.className = `sky sky-${this.mode}`;
    this.root.setAttribute("aria-label", "The sky of formulas");
    this.root.innerHTML = `
      <div class="sky-canvas"></div>
      <svg class="sky-leader" aria-hidden="true"><line x1="0" y1="0" x2="0" y2="0" /></svg>
      <header class="sky-top">
        <div class="sky-brand">
          <h1 class="mark">LINEAGE</h1>
          <p class="sky-tag">The family tree of AI math. Every formula is computed, not drawn.</p>
        </div>
        <div class="sky-actions">
          <button class="chip" data-act="search"><span>Search</span><kbd>${isMac() ? "⌘" : "Ctrl "}K</kbd></button>
          <button class="chip" data-act="tour" title="Walk the chain from the dot product to multi-head attention"><span>Tour</span></button>
        </div>
      </header>
      <p class="sky-legend">
        <span>Each star is a formula; the brightest is softmax, the hub.</span>
        <span>Rings count generations: a formula always sits outside everything it derives from.</span>
        <span class="sky-hint">${this.mode === "3d" ? "Drag to turn, scroll to zoom, click a star to approach it." : "Drag to pan, pinch to zoom, tap a star."}</span>
        <a class="sky-story" href="/">Token X-ray: radiographs of real training steps</a>
      </p>
      <p class="sky-count">${FORMULAS.length} formulas<br />${families} families, ${EDGES.length} lineage edges</p>
      <div class="sky-hover" role="tooltip" hidden></div>
      <article class="card" aria-live="polite" hidden>
        <p class="card-meta"></p>
        <h2 class="card-name"></h2>
        <div class="card-formula"></div>
        <p class="card-idea"></p>
        <div class="card-links"></div>
        <div class="card-actions">
          <button class="btn primary" data-act="open"><span>Open in the lab</span><kbd>↵</kbd></button>
          <button class="btn" data-act="close"><span>Back to the sky</span><kbd>Esc</kbd></button>
        </div>
      </article>
      <div class="tour-caption" aria-live="polite" hidden></div>
      <div class="formula-measure" aria-hidden="true"></div>`;
    host.append(this.root);
    this.measure = this.$(".formula-measure");
    this.stage = new FormulaStage(this.$(".card-formula"));

    const cb: SkyCallbacks = {
      hover: (id) => this.onHover(id),
      select: (id) => {
        this.stopTour();
        if (id === this.selected) this.open();
        else void this.select(id);
      },
      background: () => {
        this.stopTour();
        if (this.selected) void this.deselect();
      },
      frame: () => this.onFrame(),
    };
    const el = this.$(".sky-canvas");
    this.surface =
      this.mode === "3d"
        ? new Sky(el, this.layout, EDGES, cb, this.reduced)
        : new Constellation2D(el, this.layout, EDGES, cb, this.reduced);

    this.root.addEventListener("click", (e) => {
      const t = e.target as HTMLElement;
      const star = t.closest<HTMLElement>("[data-star]")?.dataset.star;
      if (star) {
        this.stopTour();
        return void this.select(star);
      }
      const act = t.closest<HTMLElement>("[data-act]")?.dataset.act;
      if (act === "open") this.open();
      if (act === "close") {
        this.stopTour();
        void this.deselect();
      }
      if (act === "search") this.hooks.search();
      if (act === "tour") void this.startTour();
    });
    document.addEventListener("keydown", this.onKey);
    document.fonts?.addEventListener?.("loadingdone", () => this.fitCard());
  }

  /** For QA: frame cost of the 3D sky. */
  get stats() {
    return this.surface.stats;
  }

  get selectedId(): string | null {
    return this.selected;
  }

  /** The card's rendered formula, the element that lifts into the lab. */
  cardFormula(): HTMLElement | null {
    return this.$(".card-formula").querySelector<HTMLElement>(".formula-layer:last-child");
  }

  cardFormulaHost(): HTMLElement {
    return this.$(".card-formula");
  }

  show(): void {
    this.visible = true;
    this.root.hidden = false;
    this.surface.start();
  }

  hide(): void {
    this.visible = false;
    this.stopTour();
    this.surface.stop();
    this.root.hidden = true;
    this.onHover(null);
  }

  /** First arrival on the home page: one orchestrated settle from far out. */
  intro(): void {
    if (this.reduced || !(this.surface instanceof Sky)) return;
    void this.surface.intro();
  }

  /** Arrive from the lab: camera already at the star, card already open. */
  arrive(id: string): void {
    this.selected = id;
    this.surface.setSelected(id);
    this.surface.jumpTo(id, this.shift());
    this.fillCard(id, false);
  }

  private shift(): number {
    return this.mode === "3d" && window.innerWidth > 900 ? 0.17 : 0;
  }

  async select(id: string): Promise<void> {
    const morph = !!this.selected && !this.$(".card").hidden;
    this.selected = id;
    this.onHover(null);
    this.surface.setSelected(id);
    this.fillCard(id, morph);
    await this.surface.focus(id, this.shift());
  }

  async deselect(): Promise<void> {
    this.selected = null;
    this.surface.setSelected(null);
    this.surface.setPath([]);
    this.$(".card").hidden = true;
    this.$(".tour-caption").hidden = true;
    this.root.classList.remove("has-card");
    await this.surface.overview();
  }

  private open(): void {
    const from = this.cardFormula();
    if (this.selected && from) this.hooks.open(this.selected, from);
  }

  private fontFor(tex: string): number {
    // largest size, up to 22px, at which the formula fits the card's width
    const host = this.$(".card-formula");
    const avail = Math.max(host.clientWidth || (this.mode === "2d" ? window.innerWidth - 40 : 380), 200);
    this.measure.style.fontSize = "20px";
    renderTex(tex, this.measure);
    const w = this.measure.firstElementChild?.getBoundingClientRect().width || avail;
    return Math.max(11, Math.min(22, (20 * avail) / w));
  }

  /** Post-render correction: KaTeX fonts may load after the first measurement. */
  private fitCard(): void {
    const host = this.$(".card-formula");
    const k = host.querySelector(".formula-layer:last-child .katex");
    if (!k || !host.clientWidth) return;
    const size = parseFloat(host.style.fontSize) || 20;
    const w = k.getBoundingClientRect().width;
    if (!w) return;
    const next = Math.max(11, Math.min(22, (size * host.clientWidth * 0.98) / w));
    if (Math.abs(next - size) > 0.25) host.style.fontSize = `${next}px`;
  }

  private fillCard(id: string, morph: boolean): void {
    const f = getFormula(id);
    const card = this.$(".card");
    const wasHidden = card.hidden;
    card.hidden = false;
    this.root.classList.add("has-card");
    const g = this.gen.get(id)!;
    this.$(".card-meta").textContent = f.hub
      ? `${FAMILY_LABEL[f.family]}, the hub at the centre`
      : `${FAMILY_LABEL[f.family]}, generation ${g}`;
    this.$(".card-name").textContent = f.name;
    this.$(".card-idea").textContent = f.idea;
    const chips = (ids: string[]) =>
      ids
        .map((p) => `<button class="star-chip" data-star="${p}">${esc(getFormula(p).name)}</button>`)
        .join("");
    const ps = parentsOf(id).map((e) => e.from);
    const cs = childrenOf(id).map((e) => e.to);
    this.$(".card-links").innerHTML =
      (ps.length
        ? `<div><h3>Comes from</h3>${chips(ps)}</div>`
        : `<div><h3>Comes from</h3><p class="muted">A root of the atlas.</p></div>`) +
      (cs.length
        ? `<div><h3>Evolves into</h3>${chips(cs)}</div>`
        : `<div><h3>Evolves into</h3><p class="muted">A frontier, for now.</p></div>`);
    const host = this.$(".card-formula");
    host.style.fontSize = `${this.fontFor(f.tex)}px`;
    if (morph && !wasHidden) void this.stage.morphTo(f.tex).then(() => this.fitCard());
    else {
      this.stage.show(f.tex);
      this.fitCard();
    }
  }

  private onHover(id: string | null): void {
    this.hoverId = id;
    this.surface.setHover(id);
    const tip = this.$(".sky-hover");
    if (!id || id === this.selected) {
      tip.hidden = true;
      return;
    }
    const f = getFormula(id);
    tip.innerHTML = `<strong>${esc(f.name)}</strong><span class="tip-meta">${FAMILY_LABEL[f.family]}, ${f.hub ? "the hub" : `generation ${this.gen.get(id)}`}</span><span class="tip-idea">${esc(f.idea)}</span>`;
    tip.hidden = false;
    this.onFrame();
  }

  private onFrame(): void {
    if (!this.visible) return;
    const tip = this.$(".sky-hover");
    if (this.hoverId && !tip.hidden) {
      const p = this.surface.project(this.hoverId);
      const w = tip.offsetWidth;
      const flip = p.x + 24 + w > window.innerWidth - 16;
      const y = Math.min(Math.max(p.y - 18, 12), window.innerHeight - tip.offsetHeight - 12);
      tip.style.transform = `translate(${flip ? p.x - 24 - w : p.x + 24}px, ${y}px)`;
      tip.classList.toggle("flip", flip);
    }
    // hairline leader from the selected star to the card
    const line = this.$<HTMLElement>(".sky-leader").querySelector("line")!;
    const card = this.$(".card");
    if (this.selected && !card.hidden && this.mode === "3d" && window.innerWidth > 900) {
      const p = this.surface.project(this.selected);
      const r = card.getBoundingClientRect();
      line.setAttribute("x1", String(p.x + 14));
      line.setAttribute("y1", String(p.y));
      line.setAttribute("x2", String(r.left - 10));
      line.setAttribute("y2", String(Math.min(Math.max(p.y, r.top + 24), r.bottom - 24)));
      line.style.opacity = p.visible ? "1" : "0";
    } else line.style.opacity = "0";
  }

  private onKey = (e: KeyboardEvent): void => {
    if (!this.visible || (e.target as HTMLElement).closest("input, dialog")) return;
    if (e.key === "Escape") {
      if (this.tour) this.stopTour();
      else if (this.selected) void this.deselect();
    }
    if (e.key === "Enter" && this.selected) {
      e.preventDefault();
      this.open();
    }
    if (this.selected && (e.key === "ArrowRight" || e.key === "ArrowLeft")) {
      this.stopTour();
      const next =
        e.key === "ArrowRight"
          ? (childrenOf(this.selected).find((x) => x.kind === "historical") ?? childrenOf(this.selected)[0])
              ?.to
          : (parentsOf(this.selected).find((x) => x.kind === "historical") ?? parentsOf(this.selected)[0])
              ?.from;
      if (next) void this.select(next);
    }
  };

  // ---------- tour ----------

  /** Walks the chain from the dot product to multi-head attention, one step every few seconds. */
  async startTour(): Promise<void> {
    this.stopTour();
    const chain = pathBetween(TOUR.from, TOUR.to);
    if (!chain) return;
    const ctl = new AbortController();
    this.tour = ctl;
    this.root.classList.add("is-touring");
    const caption = this.$(".tour-caption");
    try {
      this.surface.setPath([chain[0]]);
      await this.select(chain[0]);
      caption.hidden = false;
      caption.innerHTML = `<span class="step">Tour, ${chain.length - 1} steps</span><span class="op">${esc(getFormula(chain[0]).name)} to ${esc(getFormula(chain.at(-1)!).name)}</span>`;
      await sleep(2600, ctl.signal);
      for (let i = 1; i < chain.length; i++) {
        const edge = EDGES.find((e) => e.from === chain[i - 1] && e.to === chain[i])!;
        caption.innerHTML = `<span class="step">Step ${i} of ${chain.length - 1}</span><span class="op">${esc(edge.op)}</span><span class="kind ${edge.kind}">${edge.kind}</span>`;
        caption.classList.remove("pulse");
        void caption.offsetWidth;
        caption.classList.add("pulse");
        this.surface.setPath(chain.slice(0, i + 1));
        await this.select(chain[i]);
        await sleep(2400, ctl.signal);
      }
      caption.innerHTML = `<span class="step">Tour complete</span><span class="op">Press Enter to open multi-head attention in the lab</span>`;
    } catch (err) {
      if ((err as DOMException).name !== "AbortError") throw err;
    } finally {
      if (this.tour === ctl) this.tour = null;
      this.root.classList.remove("is-touring");
    }
  }

  stopTour(): void {
    if (!this.tour) return;
    this.tour.abort();
    this.tour = null;
    this.surface.setPath([]);
    this.$(".tour-caption").hidden = true;
  }

  dispose(): void {
    this.stopTour();
    document.removeEventListener("keydown", this.onKey);
    this.surface.dispose();
    this.root.remove();
  }
}
