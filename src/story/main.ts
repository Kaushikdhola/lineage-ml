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
interface SignedPanel {
  weights: number[][];
  intended: number[][];
  maxRelDeviation: number;
  residual: number;
}
interface DpoPanel extends SignedPanel {
  title: string;
  beta: number;
  rowLabels: string[];
  note: string;
}
interface GrpoRun {
  id: string;
  title: string;
  config: { loss_type: string; epsilon: number; num_iterations: number };
  lengths: number[];
  rewards: number[];
  steps: (SignedPanel & { step: number; ratios: number[][]; advantages: number[]; clipped: number })[];
}
interface XrayData {
  versions: Record<string, string>;
  aggregation: { lengths: number[]; micro: number; panels: Panel[] };
  leak: { lengths: number[]; panels: LeakPanel[] };
  trl: { sft: { leaks: LeakPanel[]; bf16Deviation: number }; dpo: DpoPanel; grpo: GrpoRun[] };
}

const t = String.raw;
const TEX = {
  tokenMean: t`\mathcal{L} = \frac{1}{\tok{N}{N}} \sum_{t=1}^{N} \tok{lt}{\ell_t} \qquad \frac{\partial \mathcal{L}}{\partial \tok{lt}{\ell_t}} = \tok{N}{\frac{1}{N}}`,
  grpo: t`\mathcal{L} = -\frac{1}{G}\sum_{i=1}^{G}\tok{norm}{\frac{1}{|o_i|}}\sum_{t=1}^{|o_i|} \tok{adv}{\hat{A}_i}\, \tok{ratio}{\rho_{i,t}}, \qquad \tok{adv}{\hat{A}_i} = \frac{\tok{centre}{r_i - \bar{r}}}{\tok{std}{\operatorname{std}(r) + \varepsilon}}`,
  "dr-grpo": t`\mathcal{L} = -\tok{norm}{\frac{1}{G\, L_{\max}}}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|} \tok{adv}{\hat{A}_i}\, \tok{ratio}{\rho_{i,t}}, \qquad \tok{adv}{\hat{A}_i} = \tok{centre}{r_i - \bar{r}}`,
  dapo: t`\mathcal{L} = -\tok{norm}{\frac{1}{\sum_{j}|o_j|}}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|} \tok{adv}{\hat{A}_i}\, \tok{ratio}{\rho_{i,t}}, \qquad \tok{adv}{\hat{A}_i} = \frac{\tok{centre}{r_i - \bar{r}}}{\tok{std}{\operatorname{std}(r) + \varepsilon}}`,
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
  document
    .querySelectorAll("[data-bf16]")
    .forEach((el) => (el.textContent = data.trl.sft.bf16Deviation.toExponential(1)));
  const panel = (id: string) => data.aggregation.panels.find((p) => p.id === id)!;
  const fig = (id: string) => document.querySelector<HTMLElement>(`[data-film="${id}"]`)!;
  const micro = data.aggregation.micro;

  // hero: a real run of the Trainer released before the October 2024 accumulation fix
  const heroPanel = panel("hf445-ga") as Panel & { versions: Record<string, string> };
  new Film(fig("hero")).show(weightFilm(heroPanel, data, 1, microbatchExtremes(heroPanel, micro)));
  caption(
    fig("hero"),
    `One training step of the transformers ${heroPanel.versions.transformers} Trainer, the release before the October 2024 fix: twelve sequences in four micro-batches, every cell a token. Brightness is the gradient weight it received. The bands are the bug.`,
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
    "hf445-ga": () => microbatchExtremes(panel("hf445-ga"), micro),
    "emu-dropped": () => [
      { rows: [micro, data.aggregation.lengths.length - 1], note: "never reached the optimizer" },
    ],
    "hf-ddp-per-rank": () => {
      const p = panel("hf-ddp-per-rank") as Panel & { ranks: number[][]; rankTargets: number[] };
      return p.ranks.map((seqs, r) => ({
        rows: [Math.min(...seqs), Math.max(...seqs)] as [number, number],
        note: `rank ${r}: ×${fmt(p.weights[seqs[0]][0], 3)}`,
      }));
    },
  };
  const showStep = (id: string) => {
    const p = panel(id);
    acc.show(weightFilm(p, data, 3, stepNotes[id]()));
    acc.redraw();
    const ranks = (p as Panel & { rankTargets?: number[] }).rankTargets;
    const avg = id === "hf-ddp-per-rank" ? panel("hf-ddp-avg") : null;
    caption(
      accFig,
      `${kindLabel(p.kind)}: ${clean(p.title)}.${ranks ? ` Target tokens per rank: ${ranks.join(" and ")}.` : ""} Largest deviation from the equation: ${fmt(p.maxDeviation, 3)}.${avg ? ` With the default (True), measured on the same split: ${avg.maxDeviation.toExponential(1)}, flat again.` : ""}${
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
  const smoothMean = smooth.weights.flat().reduce((a, b) => a + b, 0) / smooth.weights.flat().length;
  caption(
    fig("hf-smoothing"),
    `Label smoothing 0.1, a declared configuration. On this untrained probe the predictions are near-uniform, so smoothing looks almost like a ${fmt(smoothMean, 2)}× rescale (cosine ${smooth.minCos.toFixed(7)}); on a trained model it changes the gradient's direction far more. The cosine flags it either way.`,
  );
  const eos = panel("hf-pad-eos");
  const lastCols = eos.weights[0].length - 1;
  new Film(fig("hf-pad-eos")).show(
    weightFilm(eos, data, 7, [
      { rows: [0, 0], cols: [lastCols, lastCols], note: "end-of-sequence: weight 0" },
    ]),
  );
  caption(fig("hf-pad-eos"), `Pad token set to EOS. ${eos.note}`);

  // chapter 5: leaks (Plates 8–11 from transformers, Plate 12 through TRL's SFTTrainer)
  const leakRow = document.querySelector<HTMLElement>("[data-leaks]")!;
  const leakPanels = [...data.leak.panels, data.trl.sft.leaks.find((l) => l.id === "sft-leak-padding-free")!];
  leakPanels.forEach((lp, k) => {
    const f = document.createElement("figure");
    f.className = "plate small square";
    f.innerHTML = `<figcaption class="plate-cap"></figcaption>`;
    leakRow.append(f);
    leakFilm(f, lp, 8 + k);
  });

  // chapter 6: preference and RL trainers
  setupTrainers(data, fig);

  // the one orchestrated moment: the lightbox switches on
  requestAnimationFrame(() => document.body.classList.add("lit"));
}

function leakFilm(f: HTMLElement, lp: LeakPanel, plate: number) {
  const K = lp.matrix.length;
  const leaks = lp.maxLeak > 1e-9;
  new Film(f).show({
    rows: lp.matrix,
    scale: (v) => (v >= 0.999 ? 0.55 : Math.min(1, v * 6)),
    square: true,
    edge: `Plate ${plate}   Real run   ${lp.title}`,
    annotations: leaks ? [{ rows: [1, K - 1], cols: [0, K - 2], note: "documents reading each other" }] : [],
    tooltip: (r, c, v) =>
      r === c
        ? `Document ${r + 1}: its own gradient`
        : `Loss of document ${r + 1} → gradient into document ${c + 1}: ${fmt(v, 4)}`,
    summary: `${lp.title}. Largest cross-document gradient: ${fmt(lp.maxLeak, 4)} of the document's own.`,
  });
  caption(
    f,
    `${lp.title}. ${leaks ? `Leaks up to ${fmt(lp.maxLeak * 100, 1)}% on this untrained probe.` : "No leak."} ${lp.note}`,
  );
}

/** Signed films: brightness is |weight| on a shared scale, a weight of exactly 0 stays dark. */
const signedScale = (gmax: number) => (v: number) =>
  Math.abs(v) < 1e-12 ? 0 : 0.12 + 0.88 * (Math.abs(v) / gmax);
const sci = (x: number) => x.toExponential(1);

function setupTrainers(data: XrayData, fig: (id: string) => HTMLElement) {
  const dpo = data.trl.dpo;
  const dmax = Math.max(...dpo.weights.flat().map(Math.abs));
  new Film(fig("trl-dpo")).show({
    rows: dpo.weights,
    scale: signedScale(dmax),
    rowLabels: dpo.rowLabels.map(
      (_, i) =>
        `Pair ${Math.floor(i / 2) + 1} ${i % 2 ? "rejected" : "chosen"}, ${dpo.weights[i][0] > 0 ? "up" : "down"}`,
    ),
    edge: `Plate 13   Real run   ${dpo.title}, beta ${dpo.beta}`,
    annotations: [{ note: "one weight per answer" }],
    tooltip: (r, c, v) =>
      `${dpo.rowLabels[r]}, token ${c + 1}: ${v.toExponential(3)} (equation ${dpo.intended[r][c].toExponential(3)})`,
    summary: `${dpo.title}: per-token weights for three preference pairs.`,
  });
  caption(
    fig("trl-dpo"),
    `${dpo.title}. ${dpo.note} The margin, and so each weight's size, depends on the model's state: these values describe this untrained probe, while the closed form they match holds for any model. Matches it to ${sci(dpo.maxRelDeviation)} (relative).`,
  );

  const run = data.trl.grpo.find((g) => g.id === "grpo-default")!;
  const gmax = Math.max(...run.steps.flatMap((s) => s.weights.flat().map(Math.abs)));
  const labels = run.lengths.map(
    (L, i) =>
      `Reward ${run.rewards[i]}, ${L} tokens, ${run.steps[0].advantages[i] > 0 ? "reinforced" : "penalized"}`,
  );
  const [s1, s2] = run.steps;
  const ratios = s2.ratios.flat();
  const lo = Math.min(...ratios);
  const hi = Math.max(...ratios);
  run.steps.forEach((st, k) => {
    new Film(fig(`trl-grpo-${k + 1}`)).show({
      rows: st.weights,
      scale: signedScale(gmax),
      rowLabels: labels,
      edge: `Plate ${14 + k}   Real run   GRPOTrainer, default loss (${run.config.loss_type}), KL off (β = 0), step ${st.step}`,
      annotations: st.clipped ? [{ note: `${st.clipped} tokens past the clip range: weight 0` }] : [],
      tooltip: (r, c, v) =>
        `Answer ${r + 1}, token ${c + 1}: ${v.toExponential(3)}, ratio ${fmt(st.ratios[r][c], 3)} (equation ${st.intended[r][c].toExponential(3)})`,
      summary: `GRPOTrainer step ${st.step}: per-token weights for four scripted answers.`,
    });
  });
  caption(
    fig("trl-grpo-1"),
    `First step. The installed default loss type is ${run.config.loss_type}: every ratio is 1 and each token gets its answer's advantage divided by the batch's total completion tokens (here one group). TRL's "dapo" is DAPO's token-level normalisation only, with symmetric clipping ε = ${run.config.epsilon}. Matches the closed form to ${sci(s1.maxRelDeviation)}.`,
  );
  caption(
    fig("trl-grpo-2"),
    `Second step on the same answers (num_iterations ${run.config.num_iterations}). Ratios now run from ${fmt(lo, 2)} to ${fmt(hi, 2)}; ${s2.clipped} tokens moved past 1 ± ${run.config.epsilon} on the side their advantage pushes and received exactly zero, as the clipped objective prescribes. Which tokens clip depends on how this untrained probe moved in one step. Matches to ${sci(s2.maxRelDeviation)}.`,
  );
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
  const L_MAX = 160; // = the slider's max attribute in story/index.html
  const SHORT = 5;
  let objective: GroupObjective = "grpo";
  eq.show(TEX.grpo);

  const render = () => {
    const lengths = base.slice();
    lengths[LONG] = Number(slider.value);
    out.textContent = `${lengths[LONG]} tokens`;
    // L_max for Dr. GRPO is a constant (the slider's maximum), as TRL's max_completion_length is
    const cur = groupWeights(lengths, rewards, objective, { maxLength: L_MAX });
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
      edge: `Plate 4   Computed   ${objective === "grpo" ? "GRPO" : objective === "dapo" ? "DAPO" : "Dr. GRPO"}, on-policy, clipping inactive, KL off (β = 0)`,
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
