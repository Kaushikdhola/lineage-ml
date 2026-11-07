import { FORMULAS } from "../atlas/formulas";
import { FAMILY_LABEL, type Formula } from "../atlas/schema";
import { esc } from "./dom";

/**
 * ⌘K palette: jump to any formula by name, family, idea or symbol meaning.
 * Ranking: name prefix, then name word prefix, then name substring, then everything else.
 */
export function rankFormulas(query: string, formulas: Formula[] = FORMULAS): Formula[] {
  const q = query.trim().toLowerCase();
  if (!q) return formulas.slice();
  const scored: { f: Formula; s: number }[] = [];
  for (const f of formulas) {
    const name = f.name.toLowerCase();
    const hay =
      `${f.id} ${FAMILY_LABEL[f.family]} ${f.idea} ${Object.values(f.symbols).join(" ")}`.toLowerCase();
    let s = -1;
    if (name.startsWith(q)) s = 4;
    else if (name.split(/[\s(–-]+/).some((w) => w.startsWith(q))) s = 3;
    else if (name.includes(q)) s = 2;
    else if (hay.includes(q)) s = 1;
    if (s > 0) scored.push({ f, s });
  }
  return scored.sort((a, b) => b.s - a.s || a.f.name.localeCompare(b.f.name)).map((x) => x.f);
}

export class SearchPalette {
  private dialog: HTMLDialogElement;
  private input: HTMLInputElement;
  private list: HTMLUListElement;
  private results: Formula[] = [];
  private cursor = 0;
  private onPick: (id: string) => void = () => {};

  constructor() {
    this.dialog = document.createElement("dialog");
    this.dialog.className = "search";
    this.dialog.setAttribute("aria-label", "Search formulas");
    this.dialog.innerHTML = `
      <form method="dialog">
        <input type="search" placeholder="Search ${FORMULAS.length} formulas: softmax, Bellman, √d_k, preference…" autocomplete="off" spellcheck="false" aria-controls="search-results" />
        <ul id="search-results" role="listbox"></ul>
        <p class="search-hint"><kbd>↑</kbd><kbd>↓</kbd> move <kbd>↵</kbd> open <kbd>Esc</kbd> close</p>
      </form>`;
    document.body.append(this.dialog);
    this.input = this.dialog.querySelector("input")!;
    this.list = this.dialog.querySelector("ul")!;

    this.input.addEventListener("input", () => this.update());
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const n = this.results.length;
        if (!n) return;
        this.cursor = (this.cursor + (e.key === "ArrowDown" ? 1 : -1) + n) % n;
        this.paint();
      }
      if (e.key === "Enter") {
        e.preventDefault();
        const f = this.results[this.cursor];
        if (f) this.pick(f.id);
      }
    });
    this.list.addEventListener("click", (e) => {
      const id = (e.target as HTMLElement).closest<HTMLElement>("[data-id]")?.dataset.id;
      if (id) this.pick(id);
    });
    this.dialog.addEventListener("click", (e) => {
      if (e.target === this.dialog) this.dialog.close();
    });
  }

  get isOpen(): boolean {
    return this.dialog.open;
  }

  open(onPick: (id: string) => void): void {
    this.onPick = onPick;
    this.input.value = "";
    this.update();
    if (!this.dialog.open) this.dialog.showModal();
    this.input.focus();
  }

  private pick(id: string): void {
    this.dialog.close();
    this.onPick(id);
  }

  private update(): void {
    this.results = rankFormulas(this.input.value);
    this.cursor = 0;
    this.paint();
  }

  private paint(): void {
    this.list.innerHTML = this.results.length
      ? this.results
          .map(
            (f, i) =>
              `<li role="option" data-id="${f.id}" aria-selected="${i === this.cursor}"><span class="r-name">${esc(f.name)}</span><span class="r-fam">${FAMILY_LABEL[f.family]}</span><span class="r-idea">${esc(f.idea)}</span></li>`,
          )
          .join("")
      : `<li class="r-empty">No formula matches “${esc(this.input.value)}”. Try a family name, like attention or evals.</li>`;
    this.list.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: "nearest" });
  }
}
