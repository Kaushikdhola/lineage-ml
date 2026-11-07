import "katex/dist/katex.min.css";
import "./story.css";
import { groupWeights, type GroupObjective } from "../core/aggregation";
import { renderTex } from "../math/katex";
import { FormulaStage } from "../math/morph";
import { Film, type Annotation, type FilmSpec } from "./film";

interface Panel {
  id: string;
  title: string;
  kind: "real" | "emulation" | "reference";
  note: string;
  source: string | null;
  weights: number[][];
  intended: number[][];
  maxDeviation: number;
  minCos: number;
}
interface LeakPanel {
  id: string;
  title: string;
  kind: string;
  note: string;
  matrix: number[][];
  maxLeak: number;
}
interface XrayData {
  versions: Record<string, string>;
  aggregation: { lengths: number[]; micro: number; panels: Panel[] };
  leak: { lengths: number[]; panels: LeakPanel[] };
}

const t = String.raw;
const TEX = {
  tokenMean: t`\mathcal{L} = \frac{1}{\tok{N}{N}} \sum_{t=1}^{N} \tok{lt}{\ell_t} \qquad \frac{\partial \mathcal{L}}{\partial \tok{lt}{\ell_t}} = \tok{N}{\frac{1}{N}}`,
  grpo: t`\mathcal{L} = -\tok{norm}{\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}}\sum_{i,t} \tok{adv}{\hat{A}_i}\, \tok{ratio}{\rho_{i,t}}, \qquad \tok{adv}{\hat{A}_i} = \frac{\tok{centre}{r_i - \bar{r}}}{\tok{std}{\operatorname{std}(r)}}`,
  "dr-grpo": t`\mathcal{L} = -\tok{norm}{\frac{1}{G\, L_{\max}}}\sum_{i,t} \tok{adv}{\hat{A}_i}\, \tok{ratio}{\rho_{i,t}}, \qquad \tok{adv}{\hat{A}_i} = \tok{centre}{r_i - \bar{r}}`,
  dapo: t`\mathcal{L} = -\tok{norm}{\frac{1}{\sum_{i}|o_i|}}\sum_{i,t} \tok{adv}{\hat{A}_i}\, \tok{ratio}{\rho_{i,t}}, \qquad \tok{adv}{\hat{A}_i} = \frac{\tok{centre}{r_i - \bar{r}}}{\tok{std}{\operatorname{std}(r)}}`,
};

const fmt = (x: number, d = 2) => x.toFixed(d);
const clean = (title: string) => title.replace(/^Emulation: /, "");
const kindLabel = (k: string) =>
  k === "emulation" ? "Emulation" : k === "reference" ? "Reference" : "Real run";

/** w·N films: 1.0 (correct) sits at mid luminance, so both louder and quieter tokens are visible. */
const weightScale = (v: number) => 0.42 * Math.pow(Math.max(0, v), 1.5);

function weightFilm(p: Panel, data: XrayData, plate: number, annotations: Annotation[] = []): FilmSpec {
  const { lengths, micro } = data.aggregation;
  const seps = Array.from({ length: Math.floor((lengths.length - 1) / micro) }, (_, i) => (i + 1) * micro);
  return {
    rows: p.weights,
    scale: weightScale,
    edge: `Plate ${plate}   ${kindLabel(p.kind)}   ${clean(p.title)}`,
    separators: seps,
    annotations,
    tooltip: (r, c, v) =>
      `Sequence ${r + 1}, token ${c + 1}: weight ${fmt(v, 3)}×  (intended ${fmt(p.intended[r][c], 2)}×)`,
    summary: `${clean(p.title)}. ${p.note} Largest deviation from the intended weights: ${fmt(p.maxDeviation, 3)}.`,
  };
}

function microbatchExtremes(p: Panel, micro: number): Annotation[] {
  const means = [];
  for (let i = 0; i < p.weights.length; i += micro) {
    const rows = p.weights.slice(i, i + micro);
    const all = rows.flat();
    means.push({ i, m: all.reduce((s, x) => s + x, 0) / all.length });
  }
  const hi = means.reduce((a, b) => (b.m > a.m ? b : a));
  const lo = means.reduce((a, b) => (b.m < a.m ? b : a));
  return [
    { rows: [hi.i, hi.i + micro - 1], note: `${fmt(hi.m)}× the fair share` },
    { rows: [lo.i, lo.i + micro - 1], note: `only ${fmt(lo.m)}×` },
  ];
}

function setVersions(data: XrayData) {
  const v = data.versions;
  document.querySelectorAll("[data-version]").forEach((el) => {
    const key = (el as HTMLElement).dataset.version!;
    el.textContent = `${key} ${v[key] ?? "unknown"}`;
  });
  document.querySelectorAll("[data-versions]").forEach((el) => {
    el.textContent = Object.entries(v)
      .map(([k, ver]) => `${k} ${ver}`)
      .join(", ");
  });
}

function caption(fig: Element, html: string) {
  const cap = fig.querySelector(".plate-cap");
  if (cap) cap.innerHTML = html;
}

async function main() {
  const data: XrayData = await fetch("/xray/data.json").then((r) => {
    if (!r.ok) throw new Error(`could not load measurements (${r.status})`);
    return r.json();
  });
  setVersions(data);
  const panel = (id: string) => data.aggregation.panels.find((p) => p.id === id)!;
  const fig = (id: string) => document.querySelector<HTMLElement>(`[data-film="${id}"]`)!;
  const micro = data.aggregation.micro;

  // hero: the 2024 accumulation mechanism
  const heroPanel = panel("emu-mean-of-means");
  new Film(fig("hero")).show(weightFilm(heroPanel, data, 1, microbatchExtremes(heroPanel, micro)));
  caption(
    fig("hero"),
    `One training step, twelve sequences, every cell a token. Brightness is the gradient weight it received. The bands are a bug.`,
  );

  // chapter 1
  renderTex(TEX.tokenMean, document.querySelector<HTMLElement>('[data-tex="token-mean"]')!);
  const ref = panel("reference");
  new Film(fig("reference")).show(weightFilm(ref, data, 2));
  caption(
    fig("reference"),
    `Plain PyTorch, global token mean. Largest deviation: ${ref.maxDeviation.toExponential(1)}.`,
  );

  // chapter 2: scrolly
  const accFig = fig("accumulation");
  const acc = new Film(accFig);
  const stepNotes: Record<string, () => Annotation[]> = {
    "hf-ga": () => [],
    "emu-mean-of-means": () => microbatchExtremes(panel("emu-mean-of-means"), micro),
    "emu-dropped": () => [
      { rows: [micro, data.aggregation.lengths.length - 1], note: "never reached the optimizer" },
    ],
  };
  const showStep = (id: string) => {
    const p = panel(id);
    acc.show(weightFilm(p, data, 3, stepNotes[id]()));
    acc.redraw();
    caption(
      accFig,
      `${kindLabel(p.kind)}: ${clean(p.title)}. Largest deviation from the equation: ${fmt(p.maxDeviation, 3)}.${
        p.source ? ` <a href="${p.source}" rel="noopener" target="_blank">Source</a>` : ""
      }`,
    );
    document
      .querySelectorAll(".step")
      .forEach((s) => s.classList.toggle("active", (s as HTMLElement).dataset.step === id));
  };
  showStep("hf-ga");
  const io = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((e) => e.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible) showStep((visible.target as HTMLElement).dataset.step!);
    },
    { rootMargin: "-40% 0px -40% 0px", threshold: [0, 0.25, 0.5, 1] },
  );
  document.querySelectorAll(".step").forEach((s) => io.observe(s));

  // chapter 3: objectives
  setupObjectives(fig("objectives"));

  // chapter 4: settings
  const left = panel("hf-left");
  const leftMean = left.weights.flat().reduce((s, x) => s + x, 0) / left.weights.flat().length;
  new Film(fig("hf-left")).show(weightFilm(left, data, 5, [{ note: `every token at ${fmt(leftMean)}×` }]));
  caption(fig("hf-left"), `Left padding. ${left.note}`);
  const smooth = panel("hf-smoothing");
  new Film(fig("hf-smoothing")).show(
    weightFilm(smooth, data, 6, [{ note: `cosine ${smooth.minCos.toFixed(6)}, not a pure rescaling` }]),
  );
  caption(fig("hf-smoothing"), `Label smoothing 0.1. ${smooth.note}`);
  const eos = panel("emu-pad-eos");
  const lastCols = eos.weights[0].length - 1;
  new Film(fig("emu-pad-eos")).show(
    weightFilm(eos, data, 7, [
      { rows: [0, 0], cols: [lastCols, lastCols], note: "end-of-sequence: weight 0" },
    ]),
  );
  caption(fig("emu-pad-eos"), `Pad token set to EOS. ${eos.note}`);

  // chapter 5: leaks
  const leakRow = document.querySelector<HTMLElement>("[data-leaks]")!;
  data.leak.panels.forEach((lp, k) => {
    const f = document.createElement("figure");
    f.className = "plate small square";
    f.innerHTML = `<figcaption class="plate-cap"></figcaption>`;
    leakRow.append(f);
    const K = lp.matrix.length;
    const leaks = lp.maxLeak > 1e-9;
    new Film(f).show({
      rows: lp.matrix,
      scale: (v) => (v >= 0.999 ? 0.55 : Math.min(1, v * 6)),
      square: true,
      edge: `Plate ${8 + k}   Real run   ${lp.title}`,
      annotations: leaks
        ? [{ rows: [1, K - 1], cols: [0, K - 2], note: "documents reading each other" }]
        : [],
      tooltip: (r, c, v) =>
        r === c
          ? `Document ${r + 1}: its own gradient`
          : `Loss of document ${r + 1} → gradient into document ${c + 1}: ${fmt(v, 4)}`,
      summary: `${lp.title}. Largest cross-document gradient: ${fmt(lp.maxLeak, 4)} of the document's own.`,
    });
    caption(f, `${lp.title}. ${leaks ? `Leaks up to ${fmt(lp.maxLeak * 100, 1)}%.` : "No leak."} ${lp.note}`);
  });

  // the one orchestrated moment: the lightbox switches on
  requestAnimationFrame(() => document.body.classList.add("lit"));
}

function setupObjectives(figEl: HTMLElement) {
  const film = new Film(figEl);
  const eq = new FormulaStage(document.querySelector<HTMLElement>('[data-tex="objective"]')!);
  const note = document.querySelector<HTMLElement>(".objective-note")!;
  const slider = document.querySelector<HTMLInputElement>('[data-control="long-wrong"]')!;
  const out = document.querySelector<HTMLOutputElement>('[data-out="long-wrong"]')!;
  const rewards = [1, 0, 1, 1, 0, 0];
  const base = [12, 40, 7, 25, 60, 18];
  const LONG = 4;
  const SHORT = 5;
  let objective: GroupObjective = "grpo";
  eq.show(TEX.grpo);

  const render = () => {
    const lengths = base.slice();
    lengths[LONG] = Number(slider.value);
    out.textContent = `${lengths[LONG]} tokens`;
    const cur = groupWeights(lengths, rewards, objective);
    const gmax = Math.max(...cur.weights.flat().map(Math.abs));
    const perTok = (i: number) => Math.abs(cur.weights[i][0]);
    const ratio = perTok(SHORT) / perTok(LONG);
    film.show({
      rows: cur.weights,
      scale: (v) => 0.12 + 0.88 * (Math.abs(v) / gmax),
      rowLabels: lengths.map(
        (L, i) =>
          `${rewards[i] ? "Right" : "Wrong"}, ${L} tokens, ${cur.weights[i][0] < 0 ? "reinforced" : "penalized"}`,
      ),
      edge: `Plate 4   Computed   ${objective === "grpo" ? "GRPO" : objective === "dapo" ? "DAPO" : "Dr. GRPO"}, on-policy, clipping inactive`,
      annotations:
        objective === "grpo"
          ? [{ rows: [LONG, LONG], note: `pushed ${fmt(ratio, 1)}× less per token` }]
          : [{ rows: [LONG, SHORT], note: "same push per token" }],
      tooltip: (r, c, v) => `Answer ${r + 1}, token ${c + 1}: coefficient ${v.toExponential(2)}`,
      summary: `Per-token weights for six answers under ${objective}.`,
    });
    note.textContent =
      objective === "grpo"
        ? `Each answer's loss is averaged over its own tokens, so every token of the ${lengths[LONG]}-token wrong answer is pushed ${fmt(ratio, 1)}× less than a token of the ${lengths[SHORT]}-token wrong one. Long wrong answers get off lightly: the length bias described by Liu et al. (2025).`
        : objective === "dr-grpo"
          ? "Dr. GRPO divides by a constant instead of each answer's length, and drops the division by the rewards' standard deviation. Every token of a wrong answer is pushed equally, whatever its length."
          : "DAPO averages over all tokens in the group rather than per answer, so long answers lose their discount. It keeps the standard-deviation normalisation.";
  };

  document.querySelectorAll<HTMLButtonElement>("[data-objective]").forEach((b) =>
    b.addEventListener("click", () => {
      const next = b.dataset.objective as GroupObjective;
      if (next === objective) return;
      objective = next;
      document
        .querySelectorAll("[data-objective]")
        .forEach((x) => x.setAttribute("aria-selected", String(x === b)));
      void eq.morphTo(TEX[next]);
      render();
    }),
  );
  slider.addEventListener("input", render);
  render();
}

main().catch((err) => {
  const box = document.createElement("p");
  box.className = "load-error";
  box.textContent = `The measurements could not be loaded: ${(err as Error).message}. Reload the page to try again.`;
  document.querySelector("main")?.prepend(box);
});
