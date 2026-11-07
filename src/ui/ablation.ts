import type { Ablation, Formula } from "../atlas/schema";
import { ablateTex } from "../math/ablate";
import type { FormulaStage } from "../math/morph";
import { SIMULATORS, type SimReport } from "../sim";
import { esc, prefersReducedMotion } from "./dom";
import { simChart } from "./simChart";

/**
 * Ablation mode: "delete a term, watch AI break".
 *
 *   idle      the deletable term is marked in the formula; drag it out, or focus it and press Delete
 *   asking    the formula re-renders without it (FormulaStage exit), the reader guesses what happens;
 *             the simulator is already running in the background
 *   revealed  verdict, then the evidence: curves with 95% bands over N seeds, and the numbers
 *
 * Drag the deleted term back onto the formula (or press it) to heal. The UI computes nothing:
 * every number comes from the simulator report.
 */

interface Hooks {
  /** The formula's ablation state changed (for the URL). */
  state(token: string | null): void;
}

export interface AblationController {
  ablate(token: string, instant?: boolean): void;
  dispose(): void;
}

// results and guesses outlive navigation, so coming back is instant and the question is not re-asked
const results = new Map<string, Promise<SimReport>>();
const progress = new Map<string, { done: number; total: number }>();
const guesses = new Map<string, number>();

function runSimulator(id: string, onProgress: () => void): Promise<SimReport> {
  let p = results.get(id);
  if (!p) {
    p = SIMULATORS[id].run({
      seeds: 20,
      onProgress: (done, total) => {
        progress.set(id, { done, total });
        onProgress();
      },
    });
    p.catch(() => results.delete(id));
    results.set(id, p);
  }
  return p;
}

const fmt = (x: number, d: number) => (Math.abs(x) < 10 ** -d && x !== 0 ? x.toExponential(1) : x.toFixed(d));

export function mountAblation(
  stage: FormulaStage,
  stageEl: HTMLElement,
  panel: HTMLElement,
  f: Formula,
  rendered: Promise<void>,
  hooks: Hooks,
): AblationController {
  const list = f.ablations ?? [];
  let active: Ablation | null = null;
  let glyphs = new Map<string, string>();
  let disposed = false;
  let listeners: (() => void)[] = [];
  const reduced = prefersReducedMotion();

  panel.hidden = list.length === 0;
  if (!list.length) return { ablate() {}, dispose() {} };

  const on = <K extends keyof HTMLElementEventMap>(
    el: HTMLElement | Window,
    type: K,
    fn: (e: HTMLElementEventMap[K]) => void,
  ) => {
    el.addEventListener(type, fn as EventListener);
    listeners.push(() => el.removeEventListener(type, fn as EventListener));
  };

  const layer = () => stageEl.querySelector<HTMLElement>(".formula-layer:last-child");

  /** Mark deletable terms in the rendered formula and remember their glyphs. */
  function mark(): void {
    const l = layer();
    if (!l) return;
    for (const ab of list) {
      const els = l.querySelectorAll<HTMLElement>(`[data-tok="${ab.token}"]`);
      els.forEach((el) => {
        el.classList.add("is-ablatable");
        el.tabIndex = 0;
        el.setAttribute("role", "button");
        el.setAttribute("aria-label", `${ab.label}: press Delete to remove it from the equation`);
        el.dataset.ablate = ab.token;
      });
      if (els[0]) glyphs.set(ab.token, els[0].innerHTML);
    }
  }

  const glyph = (token: string) => `<span class="katex ab-glyph">${glyphs.get(token) ?? esc(token)}</span>`;

  // ---------- panel states ----------

  function renderIdle(): void {
    panel.className = "ablate is-idle";
    panel.innerHTML = list
      .map(
        (ab) =>
          `<p class="ab-hint"><span class="ab-k">Ablation</span> What breaks without ${glyph(ab.token)}? Drag it out of the equation, or select it and press Delete.</p>`,
      )
      .join("");
  }

  function renderAsking(ab: Ablation): void {
    const key = `${f.id}:${ab.token}`;
    const guess = guesses.get(key);
    panel.className = "ablate is-asking";
    panel.innerHTML = `
      <div class="ab-tray">
        <span class="ab-k">Deleted</span>
        <button class="tok-chip" data-heal="${ab.token}" aria-label="Put ${esc(ab.label.replace(/^Delete\s*/, ""))} back">${glyph(ab.token)}</button>
        <span class="ab-tray-hint">Drag it back into the equation, or press it, to restore.</span>
      </div>
      <fieldset class="ab-q">
        <legend>${esc(ab.question)}</legend>
        <div class="ab-choices">${ab.choices
          .map(
            (c, i) =>
              `<button class="ab-choice${guess === i ? " is-chosen" : ""}${guess !== undefined && i === ab.answer ? " is-answer" : ""}" data-choice="${i}" ${guess !== undefined ? "disabled" : ""}><span class="ab-letter">${"ABC"[i]}</span><span>${esc(c)}</span></button>`,
          )
          .join("")}</div>
      </fieldset>
      <div class="ab-result" aria-live="polite"></div>`;
    if (guess !== undefined) void reveal(ab);
  }

  async function reveal(ab: Ablation): Promise<void> {
    const key = `${f.id}:${ab.token}`;
    const guess = guesses.get(key)!;
    const out = panel.querySelector<HTMLElement>(".ab-result");
    if (!out) return;
    const right = guess === ab.answer;
    const status = () => {
      const p = progress.get(ab.simulator);
      const s = out.querySelector(".ab-status");
      if (s)
        s.textContent = p
          ? `Running seed ${p.done} of ${p.total} in your browser…`
          : "Starting the simulator…";
    };
    out.innerHTML = `<p class="ab-verdict ${right ? "right" : "wrong"}">${right ? "Right." : `Not quite. The answer is ${"ABC"[ab.answer]}.`}</p><p class="ab-status"></p>`;
    status();
    let rep: SimReport;
    try {
      rep = await runSimulator(ab.simulator, status);
    } catch (err) {
      out.innerHTML += `<p class="ab-error">The simulator failed: ${esc(String(err))}</p>`;
      return;
    }
    if (disposed || active !== ab) return;
    out.innerHTML = `
      <p class="ab-verdict ${right ? "right" : "wrong"}">${right ? "Right." : `Not quite. The answer is ${"ABC"[ab.answer]}.`} <span>${esc(rep.claim)}</span></p>
      <figure class="ab-figure">
        <figcaption><span>${esc(rep.chart.title)}</span><span class="ab-n">${rep.seeds} seeds, mean with 95% interval</span></figcaption>
        ${simChart(rep)}
      </figure>
      <table class="ab-stats">
        <thead><tr><th></th><th class="intact">${esc(rep.labels.intact)}</th><th class="ablated">${esc(rep.labels.ablated)}</th></tr></thead>
        <tbody>${rep.stats
          .map(
            (s) =>
              `<tr><th scope="row">${esc(s.label)}<small>${esc(s.hint)}</small></th><td>${fmt(s.intact.mean, s.digits)}<small>[${fmt(s.intact.lo, s.digits)}, ${fmt(s.intact.hi, s.digits)}]</small></td><td class="ablated">${fmt(s.ablated.mean, s.digits)}<small>[${fmt(s.ablated.lo, s.digits)}, ${fmt(s.ablated.hi, s.digits)}]</small></td></tr>`,
          )
          .join("")}</tbody>
      </table>
      <details class="ab-method"><summary>How this was run</summary><p>${esc(rep.method)}</p><p>Every number here is computed in <code>src/core</code>, checked against a NumPy oracle, and the claim above is asserted by a test across all ${rep.seeds} seeds.</p></details>`;
  }

  // ---------- actions ----------

  function ablate(token: string, instant = false): void {
    const ab = list.find((a) => a.token === token);
    if (!ab || active) return;
    active = ab;
    // start computing while the reader thinks
    void runSimulator(ab.simulator, () => {
      const s = panel.querySelector(".ab-status");
      const p = progress.get(ab.simulator);
      if (s && p) s.textContent = `Running seed ${p.done} of ${p.total} in your browser…`;
    });
    const tex = ablateTex(f.tex, token);
    if (instant || reduced) stage.show(tex);
    else void stage.morphTo(tex);
    stageEl.classList.add("is-ablated");
    renderAsking(ab);
    hooks.state(token);
  }

  function heal(): void {
    if (!active) return;
    active = null;
    stageEl.classList.remove("is-ablated");
    const done = reduced ? (stage.show(f.tex), Promise.resolve()) : stage.morphTo(f.tex);
    void done.then(() => {
      if (disposed || active) return;
      mark();
      renderIdle();
    });
    renderIdle();
    hooks.state(null);
  }

  // ---------- drag ----------

  /** Carries a copy of `source` with the pointer; resolves with where it was released. */
  function drag(source: HTMLElement, e: PointerEvent): Promise<{ x: number; y: number; moved: boolean }> {
    const r = source.getBoundingClientRect();
    const sx = e.clientX;
    const sy = e.clientY;
    let ghost: HTMLElement | null = null;
    source.setPointerCapture(e.pointerId);
    return new Promise((resolve) => {
      const move = (ev: PointerEvent) => {
        const dx = ev.clientX - sx;
        const dy = ev.clientY - sy;
        if (!ghost && Math.hypot(dx, dy) > 4) {
          ghost = document.createElement("div");
          ghost.className = "tok-ghost katex";
          ghost.innerHTML = source.innerHTML;
          ghost.style.fontSize = getComputedStyle(source).fontSize;
          ghost.style.left = `${r.left}px`;
          ghost.style.top = `${r.top}px`;
          document.body.append(ghost);
          source.classList.add("is-lifted");
          stageEl.classList.add("is-dragging");
        }
        if (ghost)
          ghost.style.transform = `translate(${dx}px, ${dy}px) rotate(${Math.max(-6, Math.min(6, dx / 30))}deg)`;
      };
      const up = (ev: PointerEvent) => {
        source.removeEventListener("pointermove", move);
        source.removeEventListener("pointerup", up);
        source.removeEventListener("pointercancel", up);
        source.classList.remove("is-lifted");
        stageEl.classList.remove("is-dragging");
        const g = ghost;
        if (g) {
          if (reduced) g.remove();
          else
            void g
              .animate([{ opacity: 1 }, { opacity: 0, transform: `${g.style.transform} scale(0.6)` }], {
                duration: 220,
              })
              .finished.then(
                () => g.remove(),
                () => g.remove(),
              );
        }
        resolve({ x: ev.clientX, y: ev.clientY, moved: !!g });
      };
      source.addEventListener("pointermove", move);
      source.addEventListener("pointerup", up);
      source.addEventListener("pointercancel", up);
    });
  }

  const inside = (x: number, y: number, el: HTMLElement, pad = 0) => {
    const r = el.getBoundingClientRect();
    return x >= r.left - pad && x <= r.right + pad && y >= r.top - pad && y <= r.bottom + pad;
  };

  /**
   * The deletable term under a point. Geometric, not e.target: inside a KaTeX fraction the
   * denominator is overlapped by the fraction's own layout boxes, so events land on those.
   */
  const tokenAt = (x: number, y: number): HTMLElement | null =>
    [...(layer()?.querySelectorAll<HTMLElement>(".is-ablatable") ?? [])].find((el) => inside(x, y, el, 3)) ??
    null;

  on(stageEl, "pointermove", (e) => {
    if (active) return;
    const tok = tokenAt(e.clientX, e.clientY);
    stageEl
      .querySelectorAll(".is-ablatable.is-hot")
      .forEach((el) => el !== tok && el.classList.remove("is-hot"));
    tok?.classList.add("is-hot");
    stageEl.style.cursor = tok ? "grab" : "";
  });
  on(stageEl, "pointerleave", () => {
    stageEl.querySelectorAll(".is-hot").forEach((el) => el.classList.remove("is-hot"));
    stageEl.style.cursor = "";
  });
  on(stageEl, "pointerdown", (e) => {
    const tok = tokenAt(e.clientX, e.clientY);
    if (!tok || active || e.button !== 0) return;
    e.preventDefault();
    void drag(tok, e).then(({ x, y, moved }) => {
      // released clear of the equation: delete it. A plain click selects it for the keyboard.
      if (moved && !inside(x, y, stageEl, -12)) ablate(tok.dataset.ablate!);
      else if (moved) {
        const r = tok.getBoundingClientRect();
        if (Math.hypot(x - (r.left + r.width / 2), y - (r.top + r.height / 2)) > 110)
          ablate(tok.dataset.ablate!);
      } else tok.focus();
    });
  });
  on(stageEl, "keydown", (e) => {
    const tok = (e.target as HTMLElement).closest<HTMLElement>(".is-ablatable");
    if (!tok || active) return;
    if (["Delete", "Backspace", "Enter", " "].includes(e.key)) {
      e.preventDefault();
      ablate(tok.dataset.ablate!);
    }
  });
  on(panel, "pointerdown", (e) => {
    const chip = (e.target as HTMLElement).closest<HTMLElement>("[data-heal]");
    if (!chip || e.button !== 0) return;
    e.preventDefault();
    void drag(chip, e).then(({ x, y, moved }) => {
      if (!moved || inside(x, y, stageEl, 16)) heal();
    });
  });
  on(panel, "keydown", (e) => {
    if ((e.target as HTMLElement).closest("[data-heal]") && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      heal();
    }
  });
  on(panel, "click", (e) => {
    const c = (e.target as HTMLElement).closest<HTMLElement>("[data-choice]");
    if (!c || !active || guesses.has(`${f.id}:${active.token}`)) return;
    const i = Number(c.dataset.choice);
    guesses.set(`${f.id}:${active.token}`, i);
    panel.querySelectorAll<HTMLButtonElement>("[data-choice]").forEach((b) => {
      b.disabled = true;
      const j = Number(b.dataset.choice);
      b.classList.toggle("is-chosen", j === i);
      b.classList.toggle("is-answer", j === active!.answer);
    });
    void reveal(active);
  });

  renderIdle();
  void rendered.then(() => {
    if (disposed || active) return;
    mark();
    renderIdle(); // now with the rendered glyphs
  });

  return {
    ablate,
    dispose() {
      disposed = true;
      listeners.forEach((u) => u());
      listeners = [];
      glyphs = new Map();
      stageEl.classList.remove("is-ablated", "is-dragging");
      panel.innerHTML = "";
      panel.hidden = true;
    },
  };
}
