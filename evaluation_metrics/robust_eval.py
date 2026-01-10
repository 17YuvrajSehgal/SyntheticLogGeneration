#!/usr/bin/env python3
"""
Robustness evaluation (v2) for synthetic kernel-trace windows vs real baseline.

Auto-discovers:
  - outputs/art_outputs/*/generated_traces/ar/<bench>/**.npz
  - outputs/diffusion_outputs/*/generated_traces/discrete/<bench>/*.npz

Real baseline:
  dataset/window_shards/<bench>/<split>/*.npz

Adds beyond v1:
  - Per-position JSD drift for event and dt
  - Mutual information proxy MI(event,cpu) gap (cross-field semantics)
  - Conditional timing: weighted JSD over P(dt | event) for top-K events
  - Diversity/collapse metrics: entropy, effective #events, top10 mass
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np


# -----------------------------
# Basic probability helpers
# -----------------------------
def _safe_prob(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    s = float(x.sum())
    if s <= 0:
        return np.full_like(x, 1.0 / max(1, x.size), dtype=np.float64)
    return x / (s + eps)


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _safe_prob(p, eps)
    q = _safe_prob(q, eps)
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def js_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _safe_prob(p, eps)
    q = _safe_prob(q, eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, eps) + 0.5 * kl_div(q, m, eps)


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = _safe_prob(p, eps)
    return float(-np.sum(p * np.log(p + eps)))


def effective_num(p: np.ndarray, eps: float = 1e-12) -> float:
    # exp(entropy) = "effective number of categories"
    return float(np.exp(entropy(p, eps)))


# -----------------------------
# Counting helpers
# -----------------------------
def bincount_flat(arr: np.ndarray, size: int) -> np.ndarray:
    a = arr.reshape(-1).astype(np.int64, copy=False)
    return np.bincount(a, minlength=size)


def bincount_pairs(a: np.ndarray, b: np.ndarray, a_size: int, b_size: int) -> np.ndarray:
    aa = a.reshape(-1).astype(np.int64, copy=False)
    bb = b.reshape(-1).astype(np.int64, copy=False)
    idx = aa * b_size + bb
    return np.bincount(idx, minlength=a_size * b_size)


def event_bigrams(events: np.ndarray, num_events: int) -> np.ndarray:
    e = events.astype(np.int64, copy=False)
    prev = e[:, :-1].reshape(-1)
    nxt = e[:, 1:].reshape(-1)
    idx = prev * num_events + nxt
    return np.bincount(idx, minlength=num_events * num_events)


def run_length_stats(events: np.ndarray) -> Tuple[float, float]:
    e = events.astype(np.int64, copy=False)
    runs: List[int] = []
    for row in e:
        if row.size == 0:
            continue
        rlen = 1
        for i in range(1, row.size):
            if row[i] == row[i - 1]:
                rlen += 1
            else:
                runs.append(rlen)
                rlen = 1
        runs.append(rlen)
    if not runs:
        return 0.0, 0.0
    r = np.asarray(runs, dtype=np.float64)
    return float(r.mean()), float(np.percentile(r, 95))


def topk_overlap(p: np.ndarray, q: np.ndarray, k: int) -> float:
    p_idx = np.argsort(-p)[:k]
    q_idx = np.argsort(-q)[:k]
    return float(len(set(p_idx.tolist()) & set(q_idx.tolist()))) / float(k)


# -----------------------------
# Loading
# -----------------------------
def _load_npz(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    d.close()
    return out


# -----------------------------
# Memorization proxy (optional)
# -----------------------------
def hashed_ngrams(events: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=np.uint64)
    e = events.astype(np.uint64, copy=False)
    N, L = e.shape
    if L < k:
        return np.array([], dtype=np.uint64)

    P = np.uint64(1315423911)
    hashes = []
    for i in range(L - k + 1):
        w = e[:, i : i + k]
        h = np.zeros(N, dtype=np.uint64)
        for j in range(k):
            h = h * P + w[:, j]
        hashes.append(h)
    return np.concatenate(hashes, axis=0) if hashes else np.array([], dtype=np.uint64)


def ngram_overlap_rate(real_hashes: np.ndarray, synth_hashes: np.ndarray) -> float:
    if real_hashes.size == 0 or synth_hashes.size == 0:
        return 0.0
    real_set = np.unique(real_hashes)
    synth_u = np.unique(synth_hashes)
    real_set.sort()
    idx = np.searchsorted(real_set, synth_u)
    hit = (idx < real_set.size) & (real_set[idx] == synth_u)
    return float(hit.mean())


# -----------------------------
# Mutual information proxy
# -----------------------------
def mutual_information_from_joint(joint_flat: np.ndarray, a_size: int, b_size: int) -> float:
    p_ab = _safe_prob(joint_flat).reshape(a_size, b_size)
    p_a = p_ab.sum(axis=1, keepdims=True)
    p_b = p_ab.sum(axis=0, keepdims=True)
    eps = 1e-12
    mi = np.sum(p_ab * (np.log(p_ab + eps) - np.log(p_a + eps) - np.log(p_b + eps)))
    return float(mi)


# -----------------------------
# Conditional timing: P(dt | event) for top-K events
# -----------------------------
def conditional_dt_jsd(
    real_ev_dt: np.ndarray,
    synth_ev_dt: np.ndarray,
    num_events: int,
    num_dt: int,
    top_events: np.ndarray,
    weights: np.ndarray,
) -> float:
    # real_ev_dt and synth_ev_dt are flattened [E*D]
    R = real_ev_dt.reshape(num_events, num_dt).astype(np.float64, copy=False)
    S = synth_ev_dt.reshape(num_events, num_dt).astype(np.float64, copy=False)

    jsds = []
    wts = []
    for e, w in zip(top_events.tolist(), weights.tolist()):
        r = R[e]
        s = S[e]
        if r.sum() <= 0 or s.sum() <= 0:
            continue
        jsds.append(js_div(r, s))
        wts.append(w)
    if not jsds:
        return 0.0
    w = np.asarray(wts, dtype=np.float64)
    w = w / max(1e-12, w.sum())
    return float(np.sum(w * np.asarray(jsds, dtype=np.float64)))


# -----------------------------
# Per-position drift
# -----------------------------
def per_position_jsd(real_tokens: np.ndarray, synth_tokens: np.ndarray, vocab: int) -> Tuple[float, float]:
    """
    Returns (mean_jsd, p95_jsd) across positions t.
    real_tokens/synth_tokens: (N,L)
    """
    L = real_tokens.shape[1]
    jsds = []
    for t in range(L):
        r = np.bincount(real_tokens[:, t].astype(np.int64), minlength=vocab)
        s = np.bincount(synth_tokens[:, t].astype(np.int64), minlength=vocab)
        jsds.append(js_div(r, s))
    jsds = np.asarray(jsds, dtype=np.float64)
    return float(jsds.mean()), float(np.percentile(jsds, 95))


# -----------------------------
# Real baseline
# -----------------------------
@dataclass
class RealBaseline:
    event_hist: np.ndarray
    dt_hist: np.ndarray
    cpu_hist: np.ndarray
    ev_cpu_hist: np.ndarray     # (E*C)
    ev_dt_hist: np.ndarray      # (E*D)
    bigram_hist: np.ndarray     # (E*E)

    dt0_rate: float
    cpu_switch_rate: float
    run_mean: float
    run_p95: float

    unique_event_frac: float
    top10_mass: float
    top50_mass: float
    event_entropy: float
    event_effnum: float
    mi_event_cpu: float


def build_real_baseline(
    real_glob: str,
    num_events: int,
    num_dt_buckets: int,
    num_cpus: int,
    max_shards: int,
    max_windows_per_shard: int,
) -> Tuple[RealBaseline, np.ndarray, np.ndarray]:
    paths = sorted(glob.glob(real_glob))
    if not paths:
        raise FileNotFoundError(f"No real shards match glob: {real_glob}")
    paths = paths[:max_shards]

    E, D, C = num_events, num_dt_buckets, num_cpus
    ev_hist = np.zeros(E, dtype=np.int64)
    dt_hist = np.zeros(D, dtype=np.int64)
    cpu_hist = np.zeros(C, dtype=np.int64)
    ev_cpu = np.zeros(E * C, dtype=np.int64)
    ev_dt = np.zeros(E * D, dtype=np.int64)
    bigr = np.zeros(E * E, dtype=np.int64)

    dt0_cnt = 0
    tok_cnt = 0
    cpu_sw_cnt = 0
    cpu_tr_cnt = 0

    all_events_for_cov: List[np.ndarray] = []
    run_means: List[float] = []
    run_p95s: List[float] = []

    # for per-position drift: sample a capped subset across shards
    real_ev_rows: List[np.ndarray] = []
    real_dt_rows: List[np.ndarray] = []

    for p in paths:
        d = _load_npz(p)
        ev = d["event"][:max_windows_per_shard]
        dt = d["dt"][:max_windows_per_shard]
        cpu = d["cpu"][:max_windows_per_shard]

        ev_hist += bincount_flat(ev, E)
        dt_hist += bincount_flat(dt, D)
        cpu_hist += bincount_flat(cpu, C)

        ev_cpu += bincount_pairs(ev, cpu, E, C)
        ev_dt += bincount_pairs(ev, dt, E, D)
        bigr += event_bigrams(ev, E)

        dt0_cnt += int((dt == 0).sum())
        tok_cnt += int(dt.size)
        cpu_sw_cnt += int((cpu[:, 1:] != cpu[:, :-1]).sum())
        cpu_tr_cnt += int(cpu[:, 1:].size)

        all_events_for_cov.append(ev.reshape(-1))
        m, p95 = run_length_stats(ev)
        run_means.append(m)
        run_p95s.append(p95)

        # keep small slice for per-position baseline
        take = min(5000, ev.shape[0])
        real_ev_rows.append(ev[:take])
        real_dt_rows.append(dt[:take])

    ev_p = _safe_prob(ev_hist)
    top10 = np.argsort(-ev_p)[:10]
    top50 = np.argsort(-ev_p)[:50]
    top10_mass = float(ev_p[top10].sum())
    top50_mass = float(ev_p[top50].sum())

    all_ev = np.concatenate(all_events_for_cov) if all_events_for_cov else np.array([], dtype=np.int64)
    unique_event_frac = float(np.unique(all_ev).size) / float(E) if E > 0 else 0.0

    ev_ent = entropy(ev_hist)
    ev_eff = effective_num(ev_hist)

    mi_ec = mutual_information_from_joint(ev_cpu, E, C)

    real_ev_pos = np.concatenate(real_ev_rows, axis=0) if real_ev_rows else np.zeros((1, 1), dtype=np.int32)
    real_dt_pos = np.concatenate(real_dt_rows, axis=0) if real_dt_rows else np.zeros((1, 1), dtype=np.uint8)

    rb = RealBaseline(
        event_hist=ev_hist,
        dt_hist=dt_hist,
        cpu_hist=cpu_hist,
        ev_cpu_hist=ev_cpu,
        ev_dt_hist=ev_dt,
        bigram_hist=bigr,
        dt0_rate=float(dt0_cnt) / float(max(1, tok_cnt)),
        cpu_switch_rate=float(cpu_sw_cnt) / float(max(1, cpu_tr_cnt)),
        run_mean=float(np.mean(run_means)) if run_means else 0.0,
        run_p95=float(np.mean(run_p95s)) if run_p95s else 0.0,
        unique_event_frac=unique_event_frac,
        top10_mass=top10_mass,
        top50_mass=top50_mass,
        event_entropy=ev_ent,
        event_effnum=ev_eff,
        mi_event_cpu=mi_ec,
    )
    return rb, real_ev_pos, real_dt_pos


# -----------------------------
# Synth evaluation
# -----------------------------
@dataclass
class SynthMetrics:
    path: str
    model_type: str     # "ar" or "diffusion"
    run_name: str
    variant: str        # temp/mode folder for AR, or T for diffusion (best-effort parse)

    num_windows: int
    seq_len: int

    # Marginals
    js_event: float
    js_dt: float
    js_cpu: float

    # Joints / transitions
    js_event_cpu: float
    js_event_dt: float
    js_event_bigram: float

    # New: per-position drift
    pos_js_event_mean: float
    pos_js_event_p95: float
    pos_js_dt_mean: float
    pos_js_dt_p95: float

    # New: semantics proxy
    mi_event_cpu: float
    mi_event_cpu_gap: float

    # New: conditional timing realism
    js_dt_given_event_topk: float

    # Diversity / collapse
    unique_event_frac: float
    event_entropy: float
    event_effnum: float
    top10_mass: float
    top50_overlap: float

    # Sanity
    dt0_rate: float
    cpu_switch_rate: float
    run_mean: float
    run_p95: float

    # Optional memorization proxy
    ngram_k: int
    ngram_overlap: float

    # Composite score
    robust_score: float


def infer_meta(path: str) -> Tuple[str, str, str]:
    p = path.replace("\\", "/")

    # AR: .../outputs/art_outputs/<run>/generated_traces/ar/<bench>/<Tfolder>/file.npz
    if "/outputs/art_outputs/" in p and "/generated_traces/ar/" in p:
        after = p.split("/outputs/art_outputs/")[1]
        run = after.split("/")[0]
        # variant folder if present (T0p7/T1p0/T1p3)
        m = re.search(r"/generated_traces/ar/[^/]+/([^/]+)/", p)
        variant = m.group(1) if m else "unknown"
        return "ar", run, variant

    # Diffusion: .../outputs/diffusion_outputs/<run>/generated_traces/discrete/<bench>/file.npz
    if "/outputs/diffusion_outputs/" in p and "/generated_traces/discrete/" in p:
        after = p.split("/outputs/diffusion_outputs/")[1]
        run = after.split("/")[0]
        # try parse T from filename
        m = re.search(r"_T(\d+)", os.path.basename(p))
        variant = f"T{m.group(1)}" if m else "unknown"
        return "diffusion", run, variant

    return "unknown", "unknown", "unknown"


def find_synth_files(repo: str, bench: str) -> List[str]:
    pats = [
        os.path.join(repo, "outputs", "art_outputs", "*", "generated_traces", "ar", bench, "**", "*.npz"),
        os.path.join(repo, "outputs", "diffusion_outputs", "*", "generated_traces", "discrete", bench, "*.npz"),
    ]
    out: List[str] = []
    for pat in pats:
        out.extend(glob.glob(pat, recursive=True))
    return sorted(set(out))


def eval_one_synth(
    synth_path: str,
    real: RealBaseline,
    real_ev_pos: np.ndarray,
    real_dt_pos: np.ndarray,
    num_events: int,
    num_dt_buckets: int,
    num_cpus: int,
    real_top50_idx: np.ndarray,
    real_event_prob: np.ndarray,
    topk_events_for_cond: np.ndarray,
    topk_weights: np.ndarray,
    real_ngram_hashes: np.ndarray,
    ngram_k: int,
) -> SynthMetrics:
    d = _load_npz(synth_path)
    ev = d["event"]
    dt = d["dt"]
    cpu = d["cpu"]

    N, L = ev.shape
    E, D, C = num_events, num_dt_buckets, num_cpus

    s_ev = bincount_flat(ev, E)
    s_dt = bincount_flat(dt, D)
    s_cpu = bincount_flat(cpu, C)

    js_e = js_div(real.event_hist, s_ev)
    js_d = js_div(real.dt_hist, s_dt)
    js_c = js_div(real.cpu_hist, s_cpu)

    s_ev_cpu = bincount_pairs(ev, cpu, E, C)
    s_ev_dt = bincount_pairs(ev, dt, E, D)
    js_ev_cpu = js_div(real.ev_cpu_hist, s_ev_cpu)
    js_ev_dt = js_div(real.ev_dt_hist, s_ev_dt)

    s_bigr = event_bigrams(ev, E)
    js_bigr = js_div(real.bigram_hist, s_bigr)

    # per-position drift: use min rows to keep comparable
    take = min(real_ev_pos.shape[0], ev.shape[0], 5000)
    if take >= 2 and real_ev_pos.shape[1] == L:
        pos_e_mean, pos_e_p95 = per_position_jsd(real_ev_pos[:take], ev[:take], E)
        pos_d_mean, pos_d_p95 = per_position_jsd(real_dt_pos[:take], dt[:take], D)
    else:
        pos_e_mean = pos_e_p95 = 0.0
        pos_d_mean = pos_d_p95 = 0.0

    # MI(event,cpu)
    mi_ec = mutual_information_from_joint(s_ev_cpu, E, C)
    mi_gap = abs(mi_ec - real.mi_event_cpu)

    # conditional timing P(dt|event) for top-K events
    js_dt_given_ev = conditional_dt_jsd(
        real.ev_dt_hist, s_ev_dt, E, D, topk_events_for_cond, topk_weights
    )

    # diversity / collapse
    ev_flat = ev.reshape(-1).astype(np.int64, copy=False)
    unique_event_frac = float(np.unique(ev_flat).size) / float(E) if E > 0 else 0.0
    ev_ent = entropy(s_ev)
    ev_eff = effective_num(s_ev)

    s_ev_p = _safe_prob(s_ev)
    top10 = np.argsort(-s_ev_p)[:10]
    top10_mass = float(s_ev_p[top10].sum())
    top50_ov = topk_overlap(real_event_prob, s_ev_p, k=50)

    # sanity
    dt0_rate = float((dt == 0).sum()) / float(max(1, dt.size))
    cpu_switch_rate = float((cpu[:, 1:] != cpu[:, :-1]).sum()) / float(max(1, cpu[:, 1:].size))
    run_m, run_p95 = run_length_stats(ev)

    # optional ngram overlap
    if ngram_k > 0:
        s_hash = hashed_ngrams(ev, ngram_k)
        ov = ngram_overlap_rate(real_ngram_hashes, s_hash)
    else:
        ov = 0.0

    model_type, run_name, variant = infer_meta(synth_path)

    # composite robustness score (lower = better)
    # Weighted to reward matching *structure* (joints/conditional) not just marginals.
    robust_score = (
        1.0 * js_e
        + 0.7 * js_d
        + 0.5 * js_c
        + 1.2 * js_ev_cpu
        + 1.2 * js_ev_dt
        + 1.0 * js_dt_given_ev
        + 1.0 * js_bigr
        + 0.5 * pos_e_mean
        + 0.3 * pos_d_mean
        + 0.2 * mi_gap
    )

    return SynthMetrics(
        path=synth_path,
        model_type=model_type,
        run_name=run_name,
        variant=variant,
        num_windows=int(N),
        seq_len=int(L),
        js_event=float(js_e),
        js_dt=float(js_d),
        js_cpu=float(js_c),
        js_event_cpu=float(js_ev_cpu),
        js_event_dt=float(js_ev_dt),
        js_event_bigram=float(js_bigr),
        pos_js_event_mean=float(pos_e_mean),
        pos_js_event_p95=float(pos_e_p95),
        pos_js_dt_mean=float(pos_d_mean),
        pos_js_dt_p95=float(pos_d_p95),
        mi_event_cpu=float(mi_ec),
        mi_event_cpu_gap=float(mi_gap),
        js_dt_given_event_topk=float(js_dt_given_ev),
        unique_event_frac=float(unique_event_frac),
        event_entropy=float(ev_ent),
        event_effnum=float(ev_eff),
        top10_mass=float(top10_mass),
        top50_overlap=float(top50_ov),
        dt0_rate=float(dt0_rate),
        cpu_switch_rate=float(cpu_switch_rate),
        run_mean=float(run_m),
        run_p95=float(run_p95),
        ngram_k=int(ngram_k),
        ngram_overlap=float(ov),
        robust_score=float(robust_score),
    )


def write_csv(path: str, rows: List[Dict]) -> None:
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--bench", default="compress-gzip")
    ap.add_argument("--split", default="train")
    ap.add_argument("--num_events", type=int, default=380)
    ap.add_argument("--num_dt_buckets", type=int, default=256)
    ap.add_argument("--num_cpus", type=int, default=4)
    ap.add_argument("--max_shards", type=int, default=50)
    ap.add_argument("--max_windows_per_shard", type=int, default=100000)
    ap.add_argument("--cond_topk_events", type=int, default=50, help="top-K events for P(dt|event) metric")
    ap.add_argument("--ngram_k", type=int, default=0)
    ap.add_argument("--out_dir", default="results/robustness_eval_v2")
    ap.add_argument("--synth_glob", default="")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    bench = args.bench
    out_dir = os.path.join(repo, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    real_glob = os.path.join(repo, "dataset", "window_shards", bench, args.split, "*.npz")
    print("[REAL] glob:", real_glob)

    real, real_ev_pos, real_dt_pos = build_real_baseline(
        real_glob=real_glob,
        num_events=args.num_events,
        num_dt_buckets=args.num_dt_buckets,
        num_cpus=args.num_cpus,
        max_shards=args.max_shards,
        max_windows_per_shard=args.max_windows_per_shard,
    )

    # precompute top-k indices for overlaps / conditional dt
    real_event_prob = _safe_prob(real.event_hist)
    real_top50_idx = np.argsort(-real_event_prob)[:50]

    topk = min(args.cond_topk_events, args.num_events)
    topk_events = np.argsort(-real_event_prob)[:topk]
    topk_weights = real_event_prob[topk_events]
    topk_weights = topk_weights / max(1e-12, topk_weights.sum())

    # real ngram hashes (optional)
    if args.ngram_k > 0:
        paths = sorted(glob.glob(real_glob))[: args.max_shards]
        hashes = []
        for p in paths:
            d = _load_npz(p)
            ev = d["event"][: args.max_windows_per_shard]
            hashes.append(hashed_ngrams(ev, args.ngram_k))
        real_ngram_hashes = np.concatenate(hashes) if hashes else np.array([], dtype=np.uint64)
        print(f"[REAL] ngram_k={args.ngram_k} hashes:", real_ngram_hashes.size)
    else:
        real_ngram_hashes = np.array([], dtype=np.uint64)

    # synth discovery
    if args.synth_glob:
        synth_files = sorted(glob.glob(args.synth_glob, recursive=True))
    else:
        synth_files = find_synth_files(repo, bench)

    if not synth_files:
        raise FileNotFoundError("No synthetic .npz files found. Check outputs/ or use --synth_glob")

    print("[SYNTH] files:", len(synth_files))

    results: List[SynthMetrics] = []
    for i, sp in enumerate(synth_files, 1):
        print(f"[{i}/{len(synth_files)}] eval {sp}")
        results.append(
            eval_one_synth(
                synth_path=sp,
                real=real,
                real_ev_pos=real_ev_pos,
                real_dt_pos=real_dt_pos,
                num_events=args.num_events,
                num_dt_buckets=args.num_dt_buckets,
                num_cpus=args.num_cpus,
                real_top50_idx=real_top50_idx,
                real_event_prob=real_event_prob,
                topk_events_for_cond=topk_events,
                topk_weights=topk_weights,
                real_ngram_hashes=real_ngram_hashes,
                ngram_k=args.ngram_k,
            )
        )

    ranked = sorted(results, key=lambda x: x.robust_score)

    csv_path = os.path.join(out_dir, f"robust_eval_v2_{bench}_{args.split}.csv")
    json_path = os.path.join(out_dir, f"robust_eval_v2_{bench}_{args.split}.json")

    rows = [asdict(r) for r in results]
    write_csv(csv_path, rows)

    payload = {
        "bench": bench,
        "split": args.split,
        "score_definition": "weighted robust_score (see code)",
        "real_baseline": asdict(real),
        "top_ranked": [
            {"path": r.path, "model_type": r.model_type, "run": r.run_name, "variant": r.variant, "robust_score": r.robust_score}
            for r in ranked[:15]
        ],
        "results": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print()
    print("[WROTE]", csv_path)
    print("[WROTE]", json_path)
    print()
    print("[TOP 15] (lower robust_score is better):")
    for j, r in enumerate(ranked[:15], 1):
        print(
            f"{j:2d}. score={r.robust_score:.6f}  model={r.model_type:9s} run={r.run_name:10s} var={r.variant:6s}  {os.path.basename(r.path)}"
        )


if __name__ == "__main__":
    main()