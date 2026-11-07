import "katex/dist/katex.min.css";
import "./ui/styles.css";
import { EDGES, FORMULAS } from "./atlas/formulas";
import { childrenOf, getFormula, lineagePath, parentsOf, tokenKeys, topologicalOrder } from "./atlas/graph";
import { FAMILY_LABEL, type Family, type Formula } from "./atlas/schema";
import { FormulaStage } from "./math/morph";
import { applyFocus, SymbolBus } from "./math/symbols";
import { PROBES } from "./ui/probes";

const $ = <T extends HTMLElement>(sel: string) => document.querySelector<T>(sel)!;
const esc = (s: string) =>
  s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

document.querySelector("#app")!.innerHTML = `
  <header class="top">
    <div class="brand"><span class="mark">LINEAGE</span><span class="tag">The family tree of AI math · every formula computed, not drawn</span></div>
    <div class="count">${FORMULAS.length} formulas · ${EDGES.length} lineage edges</div>
  </header>
  <nav class="rail" aria-label="Atlas"></nav>
  <main class="center">
    <ol class="path" aria-label="Lineage"></ol>
    <div class="eyebrow"></div>
    <h1 class="title"></h1>
    <div class="arrival" aria-live="polite"></div>
    <div class="stage" id="stage"></div>
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

const bus = new SymbolBus();
const stage = new FormulaStage($("#stage"), bus);
let current: Formula | null = null;
let disposeProbe: (() => void) | null = null;

// rail, grouped by family in topological order
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
$(".rail").addEventListener("click", (e) => {
  const id = (e.target as HTMLElement).closest<HTMLElement>("[data-id]")?.dataset.id;
  if (id) go(id);
});

// notation + moves + path are all clickable/hoverable
document.addEventListener("click", (e) => {
  const id = (e.target as HTMLElement).closest<HTMLElement>("[data-go]")?.dataset.go;
  if (id) go(id);
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
  if (!current || (e.target as HTMLElement).closest("input")) return;
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
  history.replaceState(null, "", `?f=${id}`);

  if (firstRender) stage.show(f.tex);
  else void stage.morphTo(f.tex);

  const arrival = $(".arrival");
  arrival.innerHTML = via
    ? `<span class="op">${esc(via.op)}</span><span class="kind ${via.kind}">${via.kind}</span>`
    : back
      ? `<span class="op back">back to the parent</span>`
      : "";
  arrival.classList.remove("show");
  void arrival.offsetWidth;
  if (via || back) arrival.classList.add("show");

  $(".eyebrow").textContent = `${FAMILY_LABEL[f.family]}${f.hub ? " · hub" : ""}`;
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
        const glyph = $("#stage").querySelector<HTMLElement>(`.formula-layer:last-child [data-tok="${k}"]`);
        return `<div class="sym" data-sym="${k}"><span class="glyph katex">${glyph ? glyph.innerHTML : esc(k)}</span><span>${esc(f.symbols[k])}</span></div>`;
      })
      .join("")}`;
  });

  const c = f.citation;
  $(".cite").innerHTML = c
    ? `<span class="badge ${c.status}">${c.status === "verified" ? "source verified" : "source check pending"}</span>
       ${esc(c.authors)} (${c.year}). ${c.url ? `<a href="${c.url}" target="_blank" rel="noopener">${esc(c.title)}</a>` : esc(c.title)}${c.venue ? `. ${esc(c.venue)}` : ""}${c.locator ? ` · ${esc(c.locator)}` : ""}`
    : `<span class="muted">Classical result; no single origin cited.</span>`;

  document
    .querySelectorAll(".rail button")
    .forEach((b) => b.classList.toggle("here", (b as HTMLElement).dataset.id === id));

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

const start = new URLSearchParams(location.search).get("f");
go(start && FORMULAS.some((f) => f.id === start) ? start : "softmax");
