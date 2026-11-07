import {
  addVec,
  colSlice,
  dot,
  hconcat,
  matmul,
  matvec,
  norm,
  shape,
  transpose,
  type Mat,
  type Vec,
} from "./linalg";
import { softmax } from "./softmax";

/** cos θ = a·b / (‖a‖ ‖b‖) */
export function cosineSimilarity(a: Vec, b: Vec): number {
  const d = norm(a) * norm(b);
  if (d === 0) throw new Error("cosineSimilarity: zero vector");
  return dot(a, b) / d;
}

/** Bahdanau et al. (2014) additive score: e = vᵀ tanh(W s + U h). */
export function additiveScore(s: Vec, h: Vec, W: Mat, U: Mat, v: Vec): number {
  const hidden = addVec(matvec(W, s), matvec(U, h)).map(Math.tanh);
  return dot(v, hidden);
}

/** Additive attention over encoder states H (one per row): weights α = softmax(e), context c = Σ α_j h_j. */
export function additiveAttention(
  s: Vec,
  H: Mat,
  W: Mat,
  U: Mat,
  v: Vec,
): { scores: Vec; weights: Vec; context: Vec } {
  const scores = H.map((h) => additiveScore(s, h, W, U, v));
  const weights = softmax(scores);
  const context = H[0].map((_, k) => H.reduce((acc, h, j) => acc + weights[j] * h[k], 0));
  return { scores, weights, context };
}

/** Luong et al. (2015) "dot" score: h_tᵀ h̄_s */
export function luongDotScore(ht: Vec, hs: Vec): number {
  return dot(ht, hs);
}

/** Luong et al. (2015) "general" score: h_tᵀ W_a h̄_s */
export function luongGeneralScore(ht: Vec, Wa: Mat, hs: Vec): number {
  return dot(ht, matvec(Wa, hs));
}

/** Raw scores S = Q Kᵀ */
export function attentionScores(Q: Mat, K: Mat): Mat {
  return matmul(Q, transpose(K));
}

/** Scaled scores S = Q Kᵀ / √d_k */
export function scaledScores(Q: Mat, K: Mat): Mat {
  const dk = shape(K)[1];
  const s = 1 / Math.sqrt(dk);
  return attentionScores(Q, K).map((r) => r.map((x) => x * s));
}

/** Additive causal mask M: 0 where key j ≤ query i, -∞ where j > i. */
export function causalMask(nq: number, nk: number = nq): Mat {
  return Array.from({ length: nq }, (_, i) => Array.from({ length: nk }, (_, j) => (j <= i ? 0 : -Infinity)));
}

export interface AttentionResult {
  scores: Mat; // Q Kᵀ / √d_k (+ M)
  weights: Mat; // row-wise softmax
  output: Mat; // weights · V
}

/** Vaswani et al. (2017): Attention(Q, K, V) = softmax(Q Kᵀ / √d_k + M) V */
export function scaledDotProductAttention(Q: Mat, K: Mat, V: Mat, mask?: Mat): AttentionResult {
  const [nk] = shape(K);
  if (shape(V)[0] !== nk) throw new Error("attention: K and V must have the same number of rows");
  let scores = scaledScores(Q, K);
  if (mask) scores = scores.map((r, i) => r.map((x, j) => x + mask[i][j]));
  const weights = scores.map((r) => softmax(r));
  return { scores, weights, output: matmul(weights, V) };
}

export interface MultiHeadWeights {
  WQ: Mat[]; // per head: d_model × d_k
  WK: Mat[];
  WV: Mat[]; // per head: d_model × d_v
  WO: Mat; // (h · d_v) × d_model
}

/** MultiHead(Q,K,V) = Concat(head_1..head_h) W^O, head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V). */
export function multiHeadAttention(
  Q: Mat,
  K: Mat,
  V: Mat,
  w: MultiHeadWeights,
  mask?: Mat,
): { heads: AttentionResult[]; output: Mat } {
  const heads = w.WQ.map((_, i) =>
    scaledDotProductAttention(matmul(Q, w.WQ[i]), matmul(K, w.WK[i]), matmul(V, w.WV[i]), mask),
  );
  const concat = hconcat(heads.map((h) => h.output));
  return { heads, output: matmul(concat, w.WO) };
}

/** Split a d_model-wide projection matrix into h equal column blocks (one per head). */
export function splitHeads(W: Mat, h: number): Mat[] {
  const cols = shape(W)[1];
  if (cols % h !== 0) throw new Error("splitHeads: width not divisible by number of heads");
  const w = cols / h;
  return Array.from({ length: h }, (_, i) => colSlice(W, i * w, (i + 1) * w));
}
