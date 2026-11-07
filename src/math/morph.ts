import { renderTex } from "./katex";
import { applyFocus, type SymbolBus } from "./symbols";
import { diffTokens, tokenIds } from "./tokens";

export interface MorphOptions {
  duration?: number;
}

const reducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

/**
 * Renders formulas and morphs between them term by term (FLIP):
 *  - terms present in both formulas glide from their old position to their new one,
 *  - new terms fade in with an accent flash (the "diff"),
 *  - removed terms fade out with the old layer.
 * Fading uses `color` rather than opacity, so persisting terms stay solid while their
 * surroundings appear (KaTeX draws rules and radicals in currentColor).
 */
export class FormulaStage {
  private current: HTMLElement | null = null;
  private busy: Promise<void> = Promise.resolve();

  constructor(
    private readonly root: HTMLElement,
    private readonly bus?: SymbolBus,
  ) {
    root.classList.add("formula-stage");
    if (bus) {
      root.addEventListener("pointerover", (e) => {
        const tok = (e.target as HTMLElement).closest<HTMLElement>("[data-tok]");
        bus.focus(tok?.dataset.tok ?? null, "formula");
      });
      root.addEventListener("pointerleave", () => bus.focus(null, "formula"));
      bus.onFocus((key) => this.current && applyFocus(this.current, key));
    }
  }

  private layer(tex: string): HTMLElement {
    const el = document.createElement("div");
    el.className = "formula-layer";
    renderTex(tex, el);
    return el;
  }

  show(tex: string): void {
    this.root.replaceChildren();
    this.current = this.layer(tex);
    this.root.append(this.current);
  }

  /** Morphs to a new formula. Calls are queued so rapid navigation never tears. */
  morphTo(tex: string, opts: MorphOptions = {}): Promise<void> {
    this.busy = this.busy.then(() => this.run(tex, opts));
    return this.busy;
  }

  private async run(tex: string, { duration = 900 }: MorphOptions): Promise<void> {
    const old = this.current;
    if (!old || reducedMotion()) return this.show(tex);

    const oldToks = [...old.querySelectorAll<HTMLElement>("[data-tok]")];
    const oldIds = tokenIds(oldToks.map((t) => t.dataset.tok!));
    const oldRects = new Map(oldIds.map((id, i) => [id, oldToks[i].getBoundingClientRect()]));

    const next = this.layer(tex);
    next.classList.add("morph-in");
    this.root.append(next);
    this.current = next;

    const newToks = [...next.querySelectorAll<HTMLElement>("[data-tok]")];
    const newIds = tokenIds(newToks.map((t) => t.dataset.tok!));
    const diff = diffTokens(
      oldToks.map((t) => t.dataset.tok!),
      newToks.map((t) => t.dataset.tok!),
    );
    const persist = new Set(diff.persist);

    oldIds.forEach((id, i) => persist.has(id) && (oldToks[i].style.visibility = "hidden"));

    const animations: Animation[] = [];
    newIds.forEach((id, i) => {
      const el = newToks[i];
      if (persist.has(id)) {
        const from = oldRects.get(id)!;
        const to = el.getBoundingClientRect();
        el.classList.add("tok-persist");
        const sx = to.width ? from.width / to.width : 1;
        const sy = to.height ? from.height / to.height : 1;
        animations.push(
          el.animate(
            [
              {
                transform: `translate(${from.left - to.left}px, ${from.top - to.top}px) scale(${sx}, ${sy})`,
              },
              { transform: "none" },
            ],
            { duration, easing: "cubic-bezier(0.65, 0, 0.35, 1)", fill: "backwards" },
          ),
        );
      } else {
        el.classList.add("tok-enter");
        el.style.setProperty("--enter-duration", `${duration}ms`);
      }
    });

    animations.push(
      old.animate([{ opacity: 1 }, { opacity: 0 }], { duration: duration * 0.45, fill: "forwards" }),
    );
    old.classList.add("morph-out");

    // two frames so the transparent starting colour is committed before the transition
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    next.style.setProperty("--fade-duration", `${duration * 0.6}ms`);
    next.classList.remove("morph-in");

    await Promise.all(animations.map((a) => a.finished.catch(() => undefined)));
    old.remove();
    if (this.bus) applyFocus(next, this.bus.focused);
  }
}
