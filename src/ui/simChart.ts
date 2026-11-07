import type { Band, SimReport } from "../sim";
import { esc } from "./dom";

/**
 * Draws a simulator's curves: mean lines with 95% bands, intact in the accent, ablated in amber
 * (the warning colour). Pure drawing: every number comes from the report.
 */
const W = 640;
const H = 250;
const M = { l: 46, r: 118, t: 14, b: 36 };

function niceStep(span: number, target: number): number {
  const raw = span / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const f = raw / mag;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
}

export function simChart(rep: SimReport): string {
  const { xs, intact, ablated } = rep.chart;
  const yMax = Math.max(...intact.hi, ...ablated.hi);
  const yMin = Math.min(0, ...intact.lo, ...ablated.lo);
  const step = niceStep(yMax - yMin, 4);
  const top = Math.ceil(yMax / step) * step;
  const bottom = Math.floor(yMin / step) * step;
  const x = (v: number) => M.l + ((v - xs[0]) / (xs[xs.length - 1] - xs[0])) * (W - M.l - M.r);
  const y = (v: number) => M.t + ((top - v) / (top - bottom)) * (H - M.t - M.b);
  const line = (vs: number[]) =>
    vs.map((v, i) => `${i ? "L" : "M"}${x(xs[i]).toFixed(1)} ${y(v).toFixed(1)}`).join("");
  const area = (b: Band) =>
    `${line(b.hi)}${b.lo
      .map((_, i) => b.lo.length - 1 - i)
      .map((i) => `L${x(xs[i]).toFixed(1)} ${y(b.lo[i]).toFixed(1)}`)
      .join("")}Z`;

  const yTicks: string[] = [];
  for (let v = bottom; v <= top + step / 2; v += step) {
    const yy = y(v).toFixed(1);
    yTicks.push(
      `<line class="grid" x1="${M.l}" x2="${W - M.r}" y1="${yy}" y2="${yy}"/><text class="tick" x="${M.l - 8}" y="${yy}" text-anchor="end" dominant-baseline="middle">${+v.toFixed(6)}</text>`,
    );
  }
  const xStep = niceStep(xs[xs.length - 1] - xs[0], 4);
  const xTicks: string[] = [];
  for (let v = 0; v <= xs[xs.length - 1]; v += xStep)
    xTicks.push(
      `<text class="tick" x="${x(v).toFixed(1)}" y="${H - M.b + 18}" text-anchor="middle">${v}</text>`,
    );

  const end = (b: Band, cls: string, label: string) => {
    const v = b.mean[b.mean.length - 1];
    return `<text class="end ${cls}" x="${W - M.r + 8}" y="${y(v).toFixed(1)}" dominant-baseline="middle">${esc(label)}</text>`;
  };
  // keep the two end labels from colliding
  const yi = y(intact.mean[intact.mean.length - 1]);
  const ya = y(ablated.mean[ablated.mean.length - 1]);
  const nudge = Math.abs(yi - ya) < 16 ? (16 - Math.abs(yi - ya)) / 2 : 0;

  const summary = `${rep.chart.title}: ${rep.labels.intact} ends at ${intact.mean.at(-1)!.toFixed(3)}, ${rep.labels.ablated} ends at ${ablated.mean.at(-1)!.toFixed(3)}, mean of ${rep.seeds} seeds.`;
  return `<svg class="sim-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(summary)}">
    ${yTicks.join("")}
    <line class="axis" x1="${M.l}" x2="${W - M.r}" y1="${y(bottom)}" y2="${y(bottom)}"/>
    ${xTicks.join("")}
    <text class="label" x="${M.l}" y="${H - 4}">${esc(rep.chart.xLabel)}</text>
    <text class="label" x="${M.l - 38}" y="${M.t - 2}" text-anchor="start">${esc(rep.chart.yLabel)}</text>
    <path class="band ablated" d="${area(ablated)}"/>
    <path class="band intact" d="${area(intact)}"/>
    <path class="mean ablated" d="${line(ablated.mean)}"/>
    <path class="mean intact" d="${line(intact.mean)}"/>
    <g transform="translate(0 ${yi < ya ? -nudge : nudge})">${end(intact, "intact", rep.labels.intact)}</g>
    <g transform="translate(0 ${ya <= yi ? -nudge : nudge})">${end(ablated, "ablated", rep.labels.ablated)}</g>
  </svg>`;
}
