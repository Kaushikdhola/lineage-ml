"""
The oracle: independent reference implementations of every formula Lineage computes.

It generates inputs, computes expected outputs with NumPy / SciPy / statsmodels
(and PyTorch when installed), and writes tests/fixtures/core.json. The TypeScript
engine must reproduce these numbers (tests/core.test.ts).

    python python/oracle.py

Rules:
  * Implementations here must be INDEPENDENT of the TypeScript ones (use library
    routines or a different method, e.g. a linear solve instead of iteration).
  * Never edit the fixture by hand. If TS and Python disagree, find out why.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats
from scipy.spatial import distance
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "core.json"
rng = np.random.default_rng(20261107)


def enc(x):
    """JSON-safe encoding: -inf -> None, numpy -> lists."""
    if isinstance(x, np.ndarray):
        return enc(x.tolist())
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    if isinstance(x, dict):
        return {k: enc(v) for k, v in x.items()}
    if isinstance(x, (float, np.floating)):
        if np.isneginf(x):
            return None
        if not np.isfinite(x):
            raise ValueError(f"non-finite value {x}")
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def softmax_cases():
    zs = [
        [1.0, 2.0, 3.0],
        [1000.0, 1001.0, 1002.0],
        [-5.0, 0.0, 2.5, 7.0, -1.0],
        [0.3, -np.inf, 1.2, -np.inf],
        rng.normal(0, 3, 12).tolist(),
    ]
    cases = []
    for z in zs:
        z = np.array(z, dtype=float)
        for tau in [0.1, 0.5, 1.0, 3.0]:
            p = special.softmax(z / tau)
            cases.append({
                "z": z, "tau": tau, "p": p,
                "logp": np.where(p > 0, special.log_softmax(z / tau), -np.inf),
                "lse": special.logsumexp(z / tau),
                "entropy": stats.entropy(p),
            })
    xs = [-800.0, -50.0, -3.0, 0.0, 2.0, 50.0, 800.0]
    sig = [{"x": x, "sigmoid": special.expit(x), "logSigmoid": special.log_expit(x)} for x in xs]
    return {"softmax": cases, "sigmoid": sig}


def attention_cases():
    out = {}
    a, b = rng.normal(size=10), rng.normal(size=10)
    out["cosine"] = {"a": a, "b": b, "cos": 1 - distance.cosine(a, b)}

    Q, K, V = rng.normal(size=(4, 8)), rng.normal(size=(6, 8)), rng.normal(size=(6, 5))
    S = Q @ K.T
    Ss = S / np.sqrt(8)
    Wt = special.softmax(Ss, axis=1)
    out["sdpa"] = {"Q": Q, "K": K, "V": V, "scores": S, "scaled": Ss, "weights": Wt, "output": Wt @ V}

    Qc, Kc, Vc = rng.normal(size=(5, 8)), rng.normal(size=(5, 8)), rng.normal(size=(5, 4))
    mask = np.triu(np.full((5, 5), -np.inf), k=1)
    Sc = Qc @ Kc.T / np.sqrt(8) + mask
    Wc = special.softmax(Sc, axis=1)
    out["causal"] = {"Q": Qc, "K": Kc, "V": Vc, "mask": mask, "weights": Wc, "output": Wc @ Vc}

    X = rng.normal(size=(5, 16))
    WQ, WK, WV, WO = (rng.normal(scale=0.25, size=(16, 16)) for _ in range(4))
    h, dk = 4, 4
    # independent formulation: reshape into heads instead of slicing per head
    q = (X @ WQ).reshape(5, h, dk).transpose(1, 0, 2)
    k = (X @ WK).reshape(5, h, dk).transpose(1, 0, 2)
    v = (X @ WV).reshape(5, h, dk).transpose(1, 0, 2)
    att = special.softmax(q @ k.transpose(0, 2, 1) / np.sqrt(dk), axis=-1) @ v
    concat = att.transpose(1, 0, 2).reshape(5, h * dk)
    out["multihead"] = {"X": X, "WQ": WQ, "WK": WK, "WV": WV, "WO": WO, "h": h, "output": concat @ WO}

    s, H = rng.normal(size=6), rng.normal(size=(7, 5))
    W, U, vv = rng.normal(size=(4, 6)), rng.normal(size=(4, 5)), rng.normal(size=4)
    e = np.array([vv @ np.tanh(W @ s + U @ hj) for hj in H])
    alpha = special.softmax(e)
    out["additive"] = {"s": s, "H": H, "W": W, "U": U, "v": vv, "scores": e, "weights": alpha, "context": alpha @ H}

    ht, hs, Wa = rng.normal(size=6), rng.normal(size=6), rng.normal(size=(6, 6))
    out["luong"] = {"ht": ht, "hs": hs, "Wa": Wa, "dot": float(np.inner(ht, hs)), "general": float(ht @ Wa @ hs)}

    torch_check(out)
    return out


def torch_check(att):
    """Cross-check scaled dot-product attention against PyTorch when it is installed."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("  torch not installed: skipping PyTorch cross-check (pip install torch to enable)")
        return
    for name, causal in [("sdpa", False), ("causal", True)]:
        c = att[name]
        q, k, v = (torch.tensor(np.asarray(c[x]), dtype=torch.float64).unsqueeze(0) for x in "QKV")
        try:
            got = F.scaled_dot_product_attention(q, k, v, is_causal=causal).squeeze(0).numpy()
            tol = 1e-10
        except RuntimeError:
            q, k, v = q.float(), k.float(), v.float()
            got = F.scaled_dot_product_attention(q, k, v, is_causal=causal).squeeze(0).double().numpy()
            tol = 1e-5
        if not np.allclose(got, c["output"], atol=tol, rtol=0):
            raise SystemExit(f"PyTorch disagrees with the oracle on {name}")
    print(f"  ✓ PyTorch {torch.__version__} scaled_dot_product_attention agrees (plain + causal)")


def scaling_cases():
    out = {"moments": []}
    for d in [4, 16, 64]:
        qs, ks = rng.normal(size=(200, d)), rng.normal(size=(200, d))
        x = np.einsum("ij,ij->i", qs, ks)
        out["moments"].append({"d": d, "qs": qs, "ks": ks, "mean": x.mean(), "variance": x.var(ddof=1)})
    q, K = rng.normal(size=64), rng.normal(size=(10, 64))
    sat = {}
    for scaled in [False, True]:
        p = special.softmax(K @ q / (np.sqrt(64) if scaled else 1.0))
        sat["scaled" if scaled else "raw"] = {"entropy": stats.entropy(p), "maxProb": p.max()}
    out["saturation"] = {"q": q, "K": K, **sat}
    return out


def rl_cases():
    out = {}
    r = rng.normal(size=8)
    gamma = 0.9
    out["returns"] = {"rewards": r, "gamma": gamma,
                      "G": [sum(gamma ** k * r[t + k] for k in range(len(r) - t)) for t in range(len(r))]}

    S, A = 5, 3
    P = rng.random((S, A, S))
    P /= P.sum(axis=2, keepdims=True)
    R = rng.normal(size=(S, A))
    pi = rng.random((S, A))
    pi /= pi.sum(axis=1, keepdims=True)
    # independent method: exact linear solve V = (I - γ P_π)^-1 r_π
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    V_pi = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
    # value iteration reference
    V = np.zeros(S)
    for _ in range(100000):
        Qv = R + gamma * P @ V
        V_new = Qv.max(axis=1)
        if np.max(np.abs(V_new - V)) < 1e-13:
            V = V_new
            break
        V = V_new
    out["mdp"] = {"P": P, "R": R, "pi": pi, "gamma": gamma, "V_pi": V_pi, "V_star": V,
                  "policy": (R + gamma * P @ V).argmax(axis=1)}

    Q = rng.normal(size=(4, 3))
    s, a, rew, s2, alpha = 1, 2, 0.7, 3, 0.1
    Q2 = Q.copy()
    Q2[s, a] += alpha * (rew + gamma * Q[s2].max() - Q[s, a])
    Q3 = Q.copy()
    Q3[s, a] += alpha * (rew - Q[s, a])
    out["qlearning"] = {"Q": Q, "s": s, "a": a, "r": rew, "s2": s2, "alpha": alpha, "gamma": gamma,
                        "Q_next": Q2, "Q_terminal": Q3}

    theta, rewards = rng.normal(size=4), rng.normal(size=4)
    J = lambda th: float(special.softmax(th) @ rewards)  # noqa: E731
    h = 1e-6
    fd = np.array([(J(theta + h * e) - J(theta - h * e)) / (2 * h) for e in np.eye(4)])
    out["policyGradient"] = {"theta": theta, "rewards": rewards, "grad": fd,
                             "note": "central finite difference of J(θ)=Σ softmax(θ)·r"}

    ratios, adv, eps = rng.uniform(0.5, 1.5, 16), rng.normal(size=16), 0.2
    out["ppo"] = {"ratios": ratios, "advantages": adv, "eps": eps,
                  "terms": np.minimum(ratios * adv, np.clip(ratios, 1 - eps, 1 + eps) * adv),
                  "objective": np.minimum(ratios * adv, np.clip(ratios, 1 - eps, 1 + eps) * adv).mean()}
    out["kl"] = {"reward": 1.3, "logp": -2.1, "logref": -2.6, "beta": 0.1, "value": 1.3 - 0.1 * (-2.1 + 2.6)}
    dpo = []
    for lw, ll, rw, rl_, beta in [(-12.0, -15.0, -13.0, -14.0, 0.1), (-3.0, -2.0, -3.5, -1.0, 0.5), (-50.0, -400.0, -60.0, -60.0, 2.0)]:
        dpo.append({"logPw": lw, "logPl": ll, "refLogPw": rw, "refLogPl": rl_, "beta": beta,
                    "loss": -special.log_expit(beta * ((lw - rw) - (ll - rl_)))})
    out["dpo"] = dpo
    return out


def evals_cases():
    out = {}
    out["wilson"] = [{"x": x, "n": n, "ci": proportion_confint(x, n, alpha=0.05, method="wilson")}
                     for x, n in [(0, 10), (3, 10), (10, 10), (142, 198), (356, 500)]]
    out["mcnemar"] = [{"b": b, "c": c, "p": float(mcnemar([[0, b], [c, 0]], exact=True).pvalue)}
                      for b, c in [(0, 0), (5, 1), (12, 30), (40, 41), (3, 25)]]
    pk = []
    for n, c, k in [(10, 3, 1), (10, 3, 5), (200, 13, 10), (20, 20, 7), (50, 0, 4), (100, 60, 100)]:
        pk.append({"n": n, "c": c, "k": k,
                   "passAtK": 1 - special.comb(n - c, k, exact=True) / special.comb(n, k, exact=True),
                   "passHatK": special.comb(c, k, exact=True) / special.comb(n, k, exact=True)})
    out["passk"] = pk

    m = 5
    true_beta = rng.normal(size=m)
    wins = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            games = 40
            wi = rng.binomial(games, special.expit(true_beta[i] - true_beta[j]))
            wins[i, j], wins[j, i] = wi, games - wi

    def nll(b):
        d = b[:, None] - b[None, :]
        return -(wins * special.log_expit(d)).sum()

    res = optimize.minimize(nll, np.zeros(m), method="BFGS", options={"gtol": 1e-12, "maxiter": 10000})
    beta = res.x - res.x.mean()
    out["bradleyTerry"] = {"wins": wins, "beta": beta,
                           "probs": [[special.expit(beta[i] - beta[j]) for j in range(m)] for i in range(m)]}
    out["elo"] = [{"ra": ra, "rb": rb, "s": s, "k": k,
                   "expected": 1 / (1 + 10 ** ((rb - ra) / 400)),
                   "updated": ra + k * (s - 1 / (1 + 10 ** ((rb - ra) / 400)))}
                  for ra, rb, s, k in [(1500, 1500, 1, 32), (1600, 1400, 0, 32), (1200, 1800, 0.5, 16)]]
    return out


def normal_cases():
    zs = [-6.0, -3.0, -1.96, 0.0, 1.0, 2.5, 3.2, 8.0]
    ps = [1e-8, 0.025, 0.2, 0.5, 0.9, 0.975, 1 - 1e-8]
    return {"z": zs, "cdf": stats.norm.cdf(zs), "p": ps, "ppf": stats.norm.ppf(ps)}


def t_quantile(p, df):
    """scipy.stats.t.ppf polished by Newton steps on scipy's CDF.

    t.ppf alone is off by ~1e-11 relative at df = 1 (e.g. ppf(0.9, 1) vs the exact tan(0.4π)),
    which is less accurate than the TS implementation it is meant to check.
    """
    q = stats.t.ppf(p, df)
    for _ in range(3):
        q -= (stats.t.cdf(q, df) - p) / stats.t.pdf(q, df)
    return q


def stats_cases():
    """Student-t quantiles/CDF, log-gamma, incomplete beta and mean CIs, from SciPy."""
    out = {"tq": [], "tcdf": [], "lgamma": [], "betainc": [], "meanci": []}
    assert abs(t_quantile(0.9, 1) - np.tan(0.4 * np.pi)) < 1e-15 * 4
    for df in [1, 2, 3, 5, 10, 19, 30, 100]:
        for p in [0.6, 0.9, 0.95, 0.975, 0.995, 0.025]:
            out["tq"].append({"p": p, "df": df, "q": t_quantile(p, df)})
        for t in [-4.0, -1.3, 0.0, 0.7, 2.093, 6.0]:
            out["tcdf"].append({"t": t, "df": df, "cdf": stats.t.cdf(t, df)})
    for x in [0.1, 0.5, 1.0, 2.5, 9.5, 50.0, 171.3]:
        out["lgamma"].append({"x": x, "v": special.gammaln(x)})
    for x, a, b in [(0.2, 0.5, 9.5), (0.7, 2.0, 3.0), (0.99, 10.0, 0.5), (0.5, 50.0, 0.5), (0.01, 1.5, 1.5)]:
        out["betainc"].append({"x": x, "a": a, "b": b, "v": special.betainc(a, b, x)})
    for n in [2, 5, 20]:
        xs = rng.lognormal(0, 1, n)
        half = t_quantile(0.975, n - 1) * stats.sem(xs)
        lo, hi = xs.mean() - half, xs.mean() + half
        out["meanci"].append({"xs": xs, "mean": xs.mean(), "sd": xs.std(ddof=1), "lo": lo, "hi": hi})
    return out


# --- retrieval training (src/core/retrieval.ts) ---------------------------------------------

def mulberry32(seed):
    """Bit-exact port of src/core/random.ts (unsigned 32-bit arithmetic)."""
    M = 0xFFFFFFFF
    state = [seed & M]

    def imul(a, b):
        return (a * b) & M

    def nxt():
        state[0] = (state[0] + 0x6D2B79F5) & M
        t = state[0]
        t = imul(t ^ (t >> 15), t | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & M
        return ((t ^ (t >> 14)) & M) / 4294967296

    return nxt


def gaussian(rng):
    def g():
        u = 0.0
        while u == 0.0:
            u = rng()
        return np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * rng())

    return g


def retrieval_forward(W, K, V, X, targets, scale):
    """Vectorised NumPy forward pass: loss, attention weights."""
    S = scale * (X @ W.T) @ K.T
    A = special.softmax(S, axis=1)
    R = A @ V - V[targets]
    return 0.5 * (R ** 2).sum(1).mean(), A


def retrieval_cases():
    out = {}
    # 1. one step on a small random instance: loss and diagnostics directly, gradient by central
    #    finite differences (an independent method from the hand-derived backprop in TS)
    d, n, dv, B = 6, 5, 4, 3
    K, V = rng.normal(size=(n, d)), rng.normal(size=(n, dv))
    W, X = rng.normal(size=(d, d)) / np.sqrt(d), rng.normal(size=(B, d))
    targets = rng.integers(0, n, B)
    steps = []
    for scale in [1 / np.sqrt(d), 1.0, 3.0]:
        loss, A = retrieval_forward(W, K, V, X, targets, scale)
        h = 1e-6
        grad = np.zeros_like(W)
        for i in range(d):
            for j in range(d):
                Wp, Wm = W.copy(), W.copy()
                Wp[i, j] += h
                Wm[i, j] -= h
                grad[i, j] = (retrieval_forward(Wp, K, V, X, targets, scale)[0]
                              - retrieval_forward(Wm, K, V, X, targets, scale)[0]) / (2 * h)
        ent = np.mean([stats.entropy(a) for a in A])
        jac = np.mean([np.linalg.norm(np.diag(a) - np.outer(a, a), "fro") for a in A])
        steps.append({"scale": scale, "loss": loss, "gradFD": grad.ravel(), "entropy": ent, "jacobianNorm": jac})
    out["step"] = {"W": W.ravel(), "K": K, "V": V, "X": X, "targets": targets, "cases": steps}

    # 2. whole seeded runs, reproduced from the same PRNG: checks data order, init and SGD plumbing
    cfg = {"dk": 16, "nKeys": 8, "dv": 8, "batch": 4, "steps": 40, "noise": 0.5}
    runs = []
    for seed, scale, lr in [(0, 1 / 4, 0.3), (3, 1.0, 0.01), (7, 1 / 4, 0.05)]:
        r = mulberry32(seed)
        g = gaussian(r)
        Kr = np.array([[g() for _ in range(cfg["dk"])] for _ in range(cfg["nKeys"])])
        Vr = np.array([[g() for _ in range(cfg["dv"])] for _ in range(cfg["nKeys"])])
        Wr = np.array([g() / np.sqrt(cfg["dk"]) for _ in range(cfg["dk"] ** 2)]).reshape(cfg["dk"], cfg["dk"])
        losses = []
        for _ in range(cfg["steps"]):
            gb = gaussian(r)
            tg, Xb = [], []
            for _ in range(cfg["batch"]):
                t = int(np.floor(r() * cfg["nKeys"]))
                tg.append(t)
                Xb.append([Kr[t, i] + cfg["noise"] * gb() for i in range(cfg["dk"])])
            Xb, tg = np.array(Xb), np.array(tg)
            loss, A = retrieval_forward(Wr, Kr, Vr, Xb, tg, scale)
            losses.append(loss)
            G = (A @ Vr - Vr[tg]) @ Vr.T
            dS = A * (G - (A * G).sum(1, keepdims=True))
            Wr = Wr - lr * (scale * dS @ Kr).T @ Xb / cfg["batch"]
        runs.append({"seed": seed, "scale": scale, "lr": lr, "loss": losses})
    out["runs"] = {"cfg": cfg, "cases": runs}
    return out


if __name__ == "__main__":
    print("oracle: computing reference values")
    data = {
        "meta": {"numpy": np.__version__, "seed": 20261107},
        **softmax_cases(),
        "attention": attention_cases(),
        "scaling": scaling_cases(),
        "rl": rl_cases(),
        "evals": evals_cases(),
        "normal": normal_cases(),
        "stats": stats_cases(),
        "retrieval": retrieval_cases(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(enc(data)))
    print(f"oracle: wrote {OUT.relative_to(ROOT)}")

    import runpy

    runpy.run_path(str(Path(__file__).with_name("oracle_aggregation.py")), run_name="__main__")
