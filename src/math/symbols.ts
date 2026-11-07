/**
 * The symbol bus links every view of a formula: the rendered TeX, the notation table and the
 * 3D scene. Hovering τ in the equation lights τ's object in the scene; hovering an object in
 * the scene lights its symbol in the equation.
 */
export type FocusSource = "formula" | "notation" | "scene" | "probe";
type FocusListener = (key: string | null, source: FocusSource) => void;

export class SymbolBus {
  private listeners = new Set<FocusListener>();
  private current: string | null = null;

  get focused(): string | null {
    return this.current;
  }

  focus(key: string | null, source: FocusSource): void {
    if (key === this.current) return;
    this.current = key;
    for (const l of this.listeners) l(key, source);
  }

  onFocus(listener: FocusListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}

/** Highlight every element carrying `data-tok=key` (or data-sym=key) inside root. */
export function applyFocus(root: ParentNode, key: string | null): void {
  root
    .querySelectorAll<HTMLElement>("[data-tok].is-focus, [data-sym].is-focus")
    .forEach((el) => el.classList.remove("is-focus"));
  if (!key) return;
  root
    .querySelectorAll<HTMLElement>(`[data-tok="${CSS.escape(key)}"], [data-sym="${CSS.escape(key)}"]`)
    .forEach((el) => el.classList.add("is-focus"));
}
