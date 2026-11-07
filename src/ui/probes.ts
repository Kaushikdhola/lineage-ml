/**
 * Probes: small live instruments, one per formula, that compute through src/core.
 * A probe never does its own math. Each returns a cleanup function.
 */
import * as core from "../core";
import type { SymbolBus } from "../math/symbols";
import { SoftmaxField } from "../scene/softmaxField";

export type Probe = (el: HTMLElement, bus: SymbolBus) => () => void;

const fmt = (x: number, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : "∞");
const pct = (x: number, d = 1) => `${(100 * x).toFixed(d)}%`;

function slider(label: string, sym: string, min: number, max: number, step: number, value: number) {
  const wrap = document.createElement("label");
  wrap.className = "ctl";
  wrap.dataset.sym = sym;
  wrap.innerHTML = `<span class="ctl-label">${label}</span><input type="range" min="${min}" max="${max}" step="${step}" value="${value}"><output></output>`;
  const input = wrap.querySelector("input")!;
  const out = wrap.querySelector("output")!;
  return {
    wrap,
    input,
    out,
    get value() {
      return Number(input.value);
    },
  };
}

function readout(rows: [string, string, string?][]): string {
  return `<dl class="readout">${rows.map(([k, v, sym]) => `<div${sym ? ` data-sym="${sym}"` : ""}><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>`;
}

function bindHover(el: HTMLElement, bus: SymbolBus): () => void {
  const over = (e: Event) => {
    const s = (e.target as HTMLElement).closest<HTMLElement>("[data-sym]");
    if (s) bus.focus(s.dataset.sym!, "probe");
  };
  const leave = () => bus.focus(null, "probe");
  el.addEventListener("pointerover", over);
  el.addEventListener("pointerleave", leave);
  return () => {
    el.removeEventListener("pointerover", over);
    el.removeEventListener("pointerleave", leave);
  };
}

const softmaxProbe =
  (withTemperature: boolean): Probe =>
  (el, bus) => {
    el.innerHTML = `<div class="scene" aria-label="Orbs: altitude is the logit, volume is the probability"></div><div class="ctls"></div><div class="out"></div>`;
    const logits = [2.0, 1.0, 0.2, -0.5, -1.2];
    const field = new SoftmaxField(el.querySelector(".scene")!, bus, logits.length);
    const ctls = el.querySelector(".ctls")!;
    const sliders = logits.map((z, i) => slider(`z${i + 1}`, "zi", -3, 3, 0.05, z));
    const tau = slider("τ", "tau", 0.1, 5, 0.05, 1);
    sliders.forEach((s) => ctls.append(s.wrap));
    if (withTemperature) ctls.append(tau.wrap);
    const update = () => {
      const z = sliders.map((s) => s.value);
      const t = withTemperature ? tau.value : 1;
      const p = core.softmax(z, t);
      sliders.forEach((s, i) => (s.out.textContent = fmt(z[i], 2)));
      tau.out.textContent = fmt(t, 2);
      field.update(p, z);
      el.querySelector(".out")!.innerHTML = readout([
        ["p", p.map((x) => fmt(x, 3)).join("  "), "softmax"],
        [
          "Σ p",
          fmt(
            p.reduce((s, x) => s + x, 0),
            6,
          ),
          "norm",
        ],
        ["entropy", `${fmt(core.entropy(p))} nats (max ${fmt(Math.log(p.length))})`],
      ]);
    };
    ctls.addEventListener("input", update);
    update();
    const unbind = bindHover(el, bus);
    return () => {
      unbind();
      field.dispose();
    };
  };

const sqrtDkProbe: Probe = (el, bus) => {
  el.innerHTML = `<p class="probe-lede">Random unit-variance queries and keys. Without scaling, softmax collapses onto one key as d_k grows.</p><div class="ctls"></div><div class="strips"></div><div class="out"></div>`;
  const dk = slider("log₂ d_k", "sqrtdk", 2, 10, 1, 6);
  el.querySelector(".ctls")!.append(dk.wrap);
  const update = () => {
    const d = 2 ** dk.value;
    dk.out.textContent = String(d);
    const q = core.randomMatrix(1, d, 7)[0];
    const K = core.randomMatrix(8, d, 11);
    const raw = core.softmax(K.map((k) => core.dot(q, k)));
    const scaled = core.softmax(K.map((k) => core.dot(q, k) / Math.sqrt(d)));
    const strip = (p: number[], label: string, sym?: string) =>
      `<div class="strip"${sym ? ` data-sym="${sym}"` : ""}><span>${label}</span>${p.map((x) => `<i style="--w:${x.toFixed(4)}" title="${fmt(x)}"></i>`).join("")}</div>`;
    el.querySelector(".strips")!.innerHTML = strip(raw, "QKᵀ") + strip(scaled, "QKᵀ/√d_k", "sqrtdk");
    const sr = core.softmaxSaturation(q, K, false);
    const ss = core.softmaxSaturation(q, K, true);
    el.querySelector(".out")!.innerHTML = readout([
      ["Var(q·k) theory", `${d} → 1 after scaling`, "sqrtdk"],
      ["max weight", `${pct(sr.maxProb)} → ${pct(ss.maxProb)}`],
      ["entropy", `${fmt(sr.entropy)} → ${fmt(ss.entropy)} nats`],
    ]);
  };
  el.addEventListener("input", update);
  update();
  return bindHover(el, bus);
};

const causalProbe: Probe = (el, bus) => {
  el.innerHTML = `<div class="ctls"><label class="ctl toggle" data-sym="M"><input type="checkbox" checked> <span class="ctl-label">causal mask M</span></label></div><div class="heat"></div><p class="probe-note">Rows are queries, columns are keys. Each row sums to 1.</p>`;
  const n = 6;
  const [Q, K, V] = [core.randomMatrix(n, 8, 3), core.randomMatrix(n, 8, 5), core.randomMatrix(n, 4, 9)];
  const box = el.querySelector<HTMLInputElement>("input")!;
  const update = () => {
    const { weights } = core.scaledDotProductAttention(Q, K, V, box.checked ? core.causalMask(n) : undefined);
    el.querySelector(".heat")!.innerHTML = weights
      .map(
        (r, i) =>
          `<div class="heat-row">${r.map((w, j) => `<i class="${j > i && box.checked ? "masked" : ""}" style="--w:${w.toFixed(4)}" title="${fmt(w)}"></i>`).join("")}</div>`,
      )
      .join("");
  };
  box.addEventListener("change", update);
  update();
  return bindHover(el, bus);
};

const passKProbe: Probe = (el, bus) => {
  el.innerHTML = `<p class="probe-lede">An agent ran each task n = 20 times. How often can you trust it?</p><div class="ctls"></div><svg class="curve" viewBox="0 0 300 120" preserveAspectRatio="none" aria-label="pass@k and pass^k as k grows"></svg><div class="legend"><span><i class="sw at"></i>pass@k: any of k tries works</span><span><i class="sw hat"></i>pass^k: all k tries work</span></div><div class="out"></div>`;
  const n = 20;
  const c = slider("successes c", "c", 0, n, 1, 14);
  const k = slider("attempts k", "k", 1, 10, 1, 4);
  el.querySelector(".ctls")!.append(c.wrap, k.wrap);
  const update = () => {
    c.out.textContent = `${c.value}/${n}`;
    k.out.textContent = String(k.value);
    const ks = Array.from({ length: 10 }, (_, i) => i + 1);
    const at = ks.map((kk) => core.passAtK(n, c.value, kk));
    const hat = ks.map((kk) => core.passHatK(n, c.value, kk));
    const pts = (ys: number[]) => ys.map((y, i) => `${(i / 9) * 300},${120 - y * 110 - 5}`).join(" ");
    el.querySelector(".curve")!.innerHTML =
      `<polyline class="l-at" points="${pts(at)}"/><polyline class="l-hat" points="${pts(hat)}"/><line class="k-mark" x1="${((k.value - 1) / 9) * 300}" x2="${((k.value - 1) / 9) * 300}" y1="0" y2="120"/>`;
    el.querySelector(".out")!.innerHTML = readout([
      ["pass@1", pct(c.value / n)],
      [`pass@${k.value} (any try works)`, pct(core.passAtK(n, c.value, k.value)), "k"],
      [`pass^${k.value} (every try works)`, pct(core.passHatK(n, c.value, k.value)), "c"],
    ]);
  };
  el.addEventListener("input", update);
  update();
  return bindHover(el, bus);
};

const bradleyTerryProbe: Probe = (el, bus) => {
  el.innerHTML = `<div class="ctls"></div><div class="out"></div>`;
  const bi = slider("β_i", "bi", -3, 3, 0.05, 0.6);
  const bj = slider("β_j", "bj", -3, 3, 0.05, -0.2);
  el.querySelector(".ctls")!.append(bi.wrap, bj.wrap);
  const update = () => {
    bi.out.textContent = fmt(bi.value, 2);
    bj.out.textContent = fmt(bj.value, 2);
    const p = core.bradleyTerryProb(bi.value, bj.value);
    const sm = core.softmax([bi.value, bj.value])[0];
    el.querySelector(".out")!.innerHTML = readout([
      ["P(i ≻ j)", pct(p, 2), "sigma"],
      ["softmax([β_i, β_j])₁", pct(sm, 2), "sigma"],
      ["Elo gap equivalent", `${fmt(((bi.value - bj.value) * 400) / Math.LN10, 0)} points`],
    ]);
  };
  el.addEventListener("input", update);
  update();
  return bindHover(el, bus);
};

const dpoProbe: Probe = (el, bus) => {
  el.innerHTML = `<div class="ctls"></div><div class="out"></div>`;
  const beta = slider("β", "beta", 0.01, 1, 0.01, 0.1);
  const margin = slider("log-ratio margin", "pi", -20, 20, 0.5, 5);
  el.querySelector(".ctls")!.append(beta.wrap, margin.wrap);
  const update = () => {
    beta.out.textContent = fmt(beta.value, 2);
    margin.out.textContent = fmt(margin.value, 1);
    const loss = core.dpoLoss(margin.value, 0, 0, 0, beta.value);
    el.querySelector(".out")!.innerHTML = readout([
      ["implied P(preferred ≻ rejected)", pct(core.sigmoid(beta.value * margin.value), 2), "sigma"],
      ["DPO loss", fmt(loss, 4), "L"],
    ]);
  };
  el.addEventListener("input", update);
  update();
  return bindHover(el, bus);
};

export const PROBES: Record<string, Probe> = {
  softmax: softmaxProbe(false),
  "softmax-temperature": softmaxProbe(true),
  "attention-scores": sqrtDkProbe,
  "scaled-scores": sqrtDkProbe,
  "scaled-dot-product-attention": sqrtDkProbe,
  "causal-attention": causalProbe,
  "pass-at-k": passKProbe,
  "pass-hat-k": passKProbe,
  "bradley-terry": bradleyTerryProbe,
  dpo: dpoProbe,
};
