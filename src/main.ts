import "katex/dist/katex.min.css";
import "./ui/styles.css";
import "./ui/sky.css";
import "./ui/ablation.css";
import { FORMULAS } from "./atlas/formulas";
import { prefersReducedMotion } from "./ui/dom";
import { createLab } from "./ui/lab";
import { SearchPalette } from "./ui/search";
import type { SkyView } from "./ui/skyView";

/**
 * Router: two views, one page. The sky (home) and the lab (?f=<id>). Moving between them is a
 * continuous transition: the card's formula lifts into the lab's stage, and back.
 */
const app = document.querySelector<HTMLElement>("#app")!;
const labEl = document.createElement("div");
labEl.hidden = true;
app.append(labEl);

type ViewName = "sky" | "lab";
let view: ViewName = "sky";
let sky: SkyView | null = null;
let transitioning = false;
const search = new SearchPalette();
void Promise.all(
  ["20px KaTeX_Main", "italic 20px KaTeX_Math", "20px KaTeX_Size1", "20px KaTeX_Size2"].map((f) =>
    document.fonts?.load(f).catch(() => undefined),
  ),
);
const known = (id: string | null): id is string => !!id && FORMULAS.some((f) => f.id === id);

const lab = createLab(labEl, {
  navigate: (id) => {
    if (view === "lab")
      history.replaceState({ view: "lab", fromSky: history.state?.fromSky ?? false }, "", `?f=${id}`);
  },
  back: () => {
    if (history.state?.fromSky) history.back();
    else void toSky(true);
  },
  search: openSearch,
  ablation: (id, token) => {
    if (view !== "lab") return;
    history.replaceState(history.state, "", `?f=${id}${token ? `&ablate=${encodeURIComponent(token)}` : ""}`);
  },
});

async function ensureSky(): Promise<SkyView> {
  if (!sky) {
    const { SkyView } = await import("./ui/skyView");
    sky = new SkyView(app, { open: (id, from) => void toLab(id, from, true), search: openSearch });
    sky.root.hidden = true;
    (window as unknown as { __lineage: unknown }).__lineage = { sky };
  }
  return sky;
}

function openSearch(): void {
  search.open((id) => {
    if (view === "lab") lab.go(id);
    else void sky?.select(id);
  });
}

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    openSearch();
  } else if (e.key === "/" && !(e.target as HTMLElement).closest("input, textarea") && !search.isOpen) {
    e.preventDefault();
    openSearch();
  }
});

/** A fixed-position copy of a rendered formula that flies between two places. */
function flyFormula(from: Element, to: Element, duration: number): Promise<void> {
  const a = from.getBoundingClientRect();
  const b = to.getBoundingClientRect();
  const offscreen = b.bottom < 0 || b.top > innerHeight || a.bottom < 0 || a.top > innerHeight;
  if (prefersReducedMotion() || !a.width || !b.width || offscreen) return Promise.resolve();
  const ghost = document.createElement("div");
  ghost.className = "lift-ghost";
  const copy = from.cloneNode(true) as HTMLElement;
  copy.style.visibility = "visible";
  ghost.append(copy);
  Object.assign(ghost.style, {
    left: `${a.left}px`,
    top: `${a.top}px`,
    // .katex sets its own 1.21em, so the ghost takes the size of the element around it
    fontSize: getComputedStyle(from.parentElement ?? from).fontSize,
  });
  document.body.append(ghost);
  const k = b.width / a.width;
  const anim = ghost.animate(
    [
      { transform: "translate(0, 0) scale(1)" },
      { transform: `translate(${b.left - a.left}px, ${b.top - a.top}px) scale(${k})` },
    ],
    { duration, easing: "cubic-bezier(0.65, 0, 0.35, 1)", fill: "forwards" },
  );
  return anim.finished.then(
    () => ghost.remove(),
    () => ghost.remove(),
  );
}

const katexOf = (el: Element | null) => el?.querySelector(".katex") ?? null;

/** Sky → lab: the card's formula lifts off into the lab's stage. */
async function toLab(id: string, from: HTMLElement | null, push: boolean): Promise<void> {
  if (transitioning) return;
  transitioning = true;
  if (push) history.pushState({ view: "lab", fromSky: true }, "", `?f=${id}`);
  view = "lab";
  document.body.classList.remove("in-sky");
  lab.reset();
  labEl.hidden = false;
  window.scrollTo(0, 0);
  lab.go(id);
  lab.setActive(true);

  const target = katexOf(lab.stageEl.querySelector(".formula-layer"));
  const source = katexOf(from);
  if (sky && !sky.root.hidden && source && target) {
    labEl.classList.add("is-entering");
    sky.root.classList.add("is-leaving");
    (target as HTMLElement).style.visibility = "hidden";
    (source as HTMLElement).style.visibility = "hidden";
    await flyFormula(source, target, 950);
    (target as HTMLElement).style.visibility = "";
    (source as HTMLElement).style.visibility = "";
  }
  sky?.hide();
  sky?.root.classList.remove("is-leaving");
  labEl.classList.remove("is-entering");
  transitioning = false;
}

/** Lab → sky: the stage's formula drops back into the card, at its star. */
async function toSky(replace: boolean): Promise<void> {
  if (transitioning) return;
  transitioning = true;
  const id = lab.current();
  const s = await ensureSky();
  if (replace) history.replaceState({ view: "sky" }, "", location.pathname);
  view = "sky";
  lab.setActive(false);
  document.body.classList.add("in-sky");
  s.root.classList.add("is-arriving");
  s.show();
  if (id) s.arrive(id);
  else void s.deselect();

  const source = katexOf(lab.stageEl.querySelector(".formula-layer:last-child"));
  const target = katexOf(s.cardFormula());
  requestAnimationFrame(() => s.root.classList.remove("is-arriving"));
  if (source && target) {
    labEl.classList.add("is-leaving");
    (target as HTMLElement).style.visibility = "hidden";
    (source as HTMLElement).style.visibility = "hidden";
    await flyFormula(source, target, 850);
    (target as HTMLElement).style.visibility = "";
    (source as HTMLElement).style.visibility = "";
  }
  labEl.hidden = true;
  labEl.classList.remove("is-leaving");
  lab.reset();
  transitioning = false;
}

window.addEventListener("popstate", () => {
  const id = new URLSearchParams(location.search).get("f");
  if (known(id) && view === "sky") void toLab(id, sky?.cardFormula() ?? null, false);
  else if (known(id)) lab.go(id);
  else if (view === "lab") void toSky(false);
});

async function boot(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const id = params.get("f");
  if (known(id)) {
    view = "lab";
    history.replaceState({ view: "lab", fromSky: false }, "", `?f=${id}`);
    labEl.hidden = false;
    lab.go(id);
    lab.setActive(true);
    const token = params.get("ablate");
    if (token) lab.ablate(token);
    return;
  }
  const s = await ensureSky();
  view = "sky";
  document.body.classList.add("in-sky");
  history.replaceState({ view: "sky" }, "", location.pathname + (params.has("tour") ? "?tour=1" : ""));
  s.show();
  s.intro();
  if (params.has("tour")) setTimeout(() => void s.startTour(), prefersReducedMotion() ? 300 : 3400);
}

void boot();
