import { EDGES, FORMULAS } from "../atlas/formulas";
import { childrenOf, getFormula, lineagePath, parentsOf, tokenKeys, topologicalOrder } from "../atlas/graph";
import { FAMILY_LABEL, type Family, type Formula } from "../atlas/schema";
import { FormulaStage } from "../math/morph";
import { applyFocus, SymbolBus } from "../math/symbols";
import { mountAblation, type AblationController } from "./ablation";
import { PROBES } from "./probes";
import { esc } from "./dom";

export interface LabHooks {
  /** The lab moved to a formula (for the URL). */
  navigate(id: string): void;
  /** The reader asked to go back to the sky. */
  back(): void;
  search(): void;
  /** A term was deleted from (or restored to) the formula on screen. */
  ablation(id: string, token: string | null): void;
}

export interface Lab {
  readonly root: HTMLElement;
  readonly stageEl: HTMLElement;
  go(id: string): void;
  current(): string | null;
  /** Clears the stage so the next go() renders instantly instead of morphing. */
  reset(): void;
  setActive(active: boolean): void;
  /** Delete a term immediately (deep links). */
  ablate(token: string): void;
}

/** The lab: one formula at a time, morphing to its parents and children, with live probes. */
export function createLab(root: HTMLElement, hooks: LabHooks): Lab {
  root.classList.add("lab");
  root.innerHTML = `
    <header class="top">
      <div class="brand"><button class="mark" data-act="back" title="Back to the sky">LINEAGE</button><span class="tag">The family tree of AI math. Every formula computed, not drawn.</span></div>
      <div class="top-actions">
        <span class="count">${FORMULAS.length} formulas, ${EDGES.length} lineage edges</span>
        <button class="chip" data-act="search"><span>Search</span><kbd>⌘K</kbd></button>
        <button class="chip" data-act="back"><span>Back to the sky</span><kbd>Esc</kbd></button>
      </div>
    </header>
    <nav class="rail" aria-label="Atlas"></nav>
    <main class="center">
      <ol class="path" aria-label="Lineage"></ol>
      <div class="eyebrow"></div>
      <h1 class="title"></h1>
      <div class="arrival" aria-live="polite"></div>
      <div class="stage"></div>
      <section class="ablate" aria-label="Ablation" hidden></section>
      <p class="idea"></p>
      <section class="moves">
        <div class="from"></div>
        <div class="to"></div>
      </section>
      <section class="notation" aria-label="Notation"></section>
      <footer class="cite"></footer>
    </main>
    <aside class="probe-panel" aria-label="Live computation">
      <div class="probe-head"><span>Live</span><code class="compute"></code></div>
      <div class="probe"></div>
    </aside>`;

  const $ = <T extends HTMLElement>(sel: string) => root.querySelector<T>(sel)!;
  const stageEl = $(".stage");
  const bus = new SymbolBus();
  const stage = new FormulaStage(stageEl, bus);
  let current: Formula | null = null;
  let disposeProbe: (() => void) | null = null;
  let ablation: AblationController | null = null;
  let active = false;

  const order = topologicalOrder();
  const families = Object.keys(FAMILY_LABEL) as Family[];
  $(".rail").innerHTML = families
    .filter((fam) => FORMULAS.some((f) => f.family === fam))
    .map(
      (fam) =>
        `<section><h2>${FAMILY_LABEL[fam]}</h2><ul>${order
          .map(getFormula)
          .filter((f) => f.family === fam)
          .map(
            (f) =>
              `<li><button data-id="${f.id}">${esc(f.name)}${f.hub ? '<span class="hub">hub</span>' : ""}</button></li>`,
          )
          .join("")}</ul></section>`,
    )
    .join("");

  root.addEventListener("click", (e) => {
    const t = e.target as HTMLElement;
    const id =
      t.closest<HTMLElement>("[data-id]")?.dataset.id ?? t.closest<HTMLElement>("[data-go]")?.dataset.go;
    if (id) return go(id);
    const act = t.closest<HTMLElement>("[data-act]")?.dataset.act;
    if (act === "back") hooks.back();
    if (act === "search") hooks.search();
  });
  $(".notation").addEventListener("pointerover", (e) => {
    const sym = (e.target as HTMLElement).closest<HTMLElement>("[data-sym]")?.dataset.sym;
    bus.focus(sym ?? null, "notation");
  });
  $(".notation").addEventListener("pointerleave", () => bus.focus(null, "notation"));
  bus.onFocus((key) => {
    applyFocus($(".notation"), key);
    applyFocus($(".probe"), key);
  });

  document.addEventListener("keydown", (e) => {
    if (!active || !current || (e.target as HTMLElement).closest("input, dialog")) return;
    if (e.key === "Escape") hooks.back();
    if (e.key === "ArrowRight") {
      const c = childrenOf(current.id);
      const next = c.find((x) => x.kind === "historical") ?? c[0];
      if (next) go(next.to);
    }
    if (e.key === "ArrowLeft") {
      const p = parentsOf(current.id);
      const prev = p.find((x) => x.kind === "historical") ?? p[0];
      if (prev) go(prev.from);
    }
  });

  function go(id: string): void {
    const f = getFormula(id);
    if (current?.id === id) return;
    const via = current ? EDGES.find((e) => e.from === current!.id && e.to === id) : undefined;
    const back = current ? EDGES.find((e) => e.to === current!.id && e.from === id) : undefined;
    const firstRender = !current;
    current = f;
    hooks.navigate(id);

    let rendered: Promise<void>;
    if (firstRender) {
      stage.show(f.tex);
      rendered = Promise.resolve();
    } else rendered = stage.morphTo(f.tex);
    ablation?.dispose();
    ablation = mountAblation(stage, stageEl, $(".ablate"), f, rendered, {
      state: (token) => hooks.ablation(id, token),
    });

    const arrival = $(".arrival");
    arrival.innerHTML = via
      ? `<span class="op">${esc(via.op)}</span><span class="kind ${via.kind}">${via.kind}</span>`
      : back
        ? `<span class="op back">back to the parent</span>`
        : "";
    arrival.classList.remove("show");
    void arrival.offsetWidth;
    if (via || back) arrival.classList.add("show");

    $(".eyebrow").textContent = `${FAMILY_LABEL[f.family]}${f.hub ? ", hub" : ""}`;
    $(".title").textContent = f.name;
    $(".idea").textContent = f.idea;

    const path = lineagePath(id);
    $(".path").innerHTML = path
      .map(
        (pid) =>
          `<li><button data-go="${pid}" class="${pid === id ? "here" : ""}">${esc(getFormula(pid).name)}</button></li>`,
      )
      .join("");

    const parents = parentsOf(id);
    const kids = childrenOf(id);
    $(".moves .from").innerHTML = parents.length
      ? `<h3>Comes from</h3>${parents.map((e) => `<button data-go="${e.from}"><b>${esc(getFormula(e.from).name)}</b><span>${esc(e.op)}</span></button>`).join("")}`
      : `<h3>Comes from</h3><p class="muted">A root of the atlas.</p>`;
    $(".moves .to").innerHTML = kids.length
      ? `<h3>Evolves into</h3>${kids.map((e) => `<button data-go="${e.to}"><b>${esc(getFormula(e.to).name)}</b><span>${esc(e.op)}</span></button>`).join("")}`
      : `<h3>Evolves into</h3><p class="muted">A frontier, for now.</p>`;

    // notation rows reuse the rendered glyphs from the stage
    requestAnimationFrame(() => {
      const keys = [...new Set(tokenKeys(f.tex))];
      $(".notation").innerHTML = `<h3>Notation</h3>${keys
        .map((k) => {
          const glyph = stageEl.querySelector<HTMLElement>(`.formula-layer:last-child [data-tok="${k}"]`);
          return `<div class="sym" data-sym="${k}"><span class="glyph katex">${glyph ? glyph.innerHTML : esc(k)}</span><span>${esc(f.symbols[k])}</span></div>`;
        })
        .join("")}`;
    });

    const c = f.citation;
    $(".cite").innerHTML = c
      ? `<span class="badge ${c.status}">${c.status === "verified" ? "source verified" : "source check pending"}</span>
         ${esc(c.authors)} (${c.year}). ${c.url ? `<a href="${c.url}" target="_blank" rel="noopener">${esc(c.title)}</a>` : esc(c.title)}${c.venue ? `. ${esc(c.venue)}` : ""}${c.locator ? `, ${esc(c.locator)}` : ""}`
      : `<span class="muted">Classical result; no single origin cited.</span>`;

    root
      .querySelectorAll<HTMLElement>(".rail button")
      .forEach((b) => b.classList.toggle("here", b.dataset.id === id));

    disposeProbe?.();
    disposeProbe = null;
    $(".compute").textContent = f.compute ? `core.${f.compute}()` : "no computation";
    const probe = PROBES[id];
    const panel = $(".probe");
    if (probe) disposeProbe = probe(panel, bus);
    else
      panel.innerHTML = `<p class="probe-note">${
        f.compute
          ? `Computed by <code>core.${f.compute}()</code> and verified against the NumPy/SciPy oracle. ${f.computeNote ? esc(f.computeNote) + ". " : ""}An interactive instrument for this formula arrives with its scene.`
          : esc(f.computeNote ?? "")
      }</p>`;
  }

  return {
    root,
    stageEl,
    go,
    current: () => current?.id ?? null,
    ablate(token) {
      ablation?.ablate(token, true);
    },
    reset() {
      ablation?.dispose();
      ablation = null;
      // dispose the probe (and any scene it owns) when the lab is left
      disposeProbe?.();
      disposeProbe = null;
      current = null;
    },
    setActive(a) {
      active = a;
    },
  };
}
