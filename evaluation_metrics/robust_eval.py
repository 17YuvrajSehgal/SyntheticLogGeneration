#!/usr/bin/env python3
"""
Robust evaluation of synthetic kernel-trace windows against real aggregate baseline.

Outputs:
  - CSV summary (one row per synthetic file)
  - JSON detailed report

Metrics:
  - Marginal KL + JSD: event/dt/cpu
  - Joint JSD: (event,cpu) and (event,dt)
  - Transition JSD: event bigrams
  - Sanity signals: unique coverage, top-k overlap, dt=0 rate, cpu switch rate, run-length stats
  - Optional memorization proxy: n-gram overlap rate vs real (hashed) (set --ngram_k > 0)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np


# -----------------------------
# Math helpers
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

def bincount_flat(arr: np.ndarray, size: int) -> np.ndarray:
    a = arr.reshape(-1).astype(np.int64, copy=False)
    return np.bincount(a, minlength=size)

def bincount_pairs(a: np.ndarray, b: np.ndarray, a_size: int, b_size: int) -> np.ndarray:
    # flatten 2D windows to 1D tokens
    aa = a.reshape(-1).astype(np.int64, copy=False)
    bb = b.reshape(-1).astype(np.int64, copy=False)
    idx = aa * b_size + bb
    out = np.bincount(idx, minlength=a_size * b_size)
    return out  # 1D of length a_size*b_size

def event_bigrams(events: np.ndarray, num_events: int) -> np.ndarray:
    # events shape: (N, L)
    e = events.astype(np.int64, copy=False)
    prev = e[:, :-1].reshape(-1)
    nxt = e[:, 1:].reshape(-1)
    idx = prev * num_events + nxt
    return np.bincount(idx, minlength=num_events * num_events)


def topk_overlap(p: np.ndarray, q: np.ndarray, k: int) -> float:
    # overlap between top-k indices (by probability)
    p_idx = np.argsort(-p)[:k]
    q_idx = np.argsort(-q)[:k]
    return float(len(set(p_idx.tolist()) & set(q_idx.tolist()))) / float(k)

def run_length_stats(events: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean and 95th percentile of run-lengths of identical consecutive events.
    """
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
    runs_arr = np.asarray(runs, dtype=np.float64)
    return float(runs_arr.mean()), float(np.percentile(runs_arr, 95))


# -----------------------------
# Real baseline builder
# -----------------------------
@dataclass
class RealBaseline:
    event_hist: np.ndarray
    dt_hist: np.ndarray
    cpu_hist: np.ndarray
    ev_cpu_hist: np.ndarray          # (E*C) flattened
    ev_dt_hist: np.ndarray           # (E*D) flattened
    bigram_hist: np.ndarray          # (E*E) flattened
    dt0_rate: float
    cpu_switch_rate: float
    run_mean: float
    run_p95: float
    unique_event_frac: float
    top50_mass: float

def _load_npz(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    d.close()
    return out

def build_real_baseline(
    real_glob: str,
    num_events: int,
    num_dt_buckets: int,
    num_cpus: int,
    max_shards: int,
    max_windows_per_shard: int,
) -> RealBaseline:
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

    # run-length / unique coverage
    all_events_for_cov: List[np.ndarray] = []
    run_means: List[float] = []
    run_p95s: List[float] = []

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

        # sanity rates
        dt0_cnt += int((dt == 0).sum())
        tok_cnt += int(dt.size)

        cpu_sw_cnt += int((cpu[:, 1:] != cpu[:, :-1]).sum())
        cpu_tr_cnt += int(cpu[:, 1:].size)

        all_events_for_cov.append(ev.reshape(-1))

        m, p95 = run_length_stats(ev)
        run_means.append(m)
        run_p95s.append(p95)

    ev_p = _safe_prob(ev_hist)
    top50 = np.argsort(-ev_p)[:50]
    top50_mass = float(ev_p[top50].sum())

    all_ev = np.concatenate(all_events_for_cov) if all_events_for_cov else np.array([], dtype=np.int64)
    unique_event_frac = float(np.unique(all_ev).size) / float(E) if E > 0 else 0.0

    return RealBaseline(
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
        top50_mass=top50_mass,
    )


# -----------------------------
# Optional memorization proxy
# -----------------------------
def hashed_ngrams(events: np.ndarray, k: int) -> np.ndarray:
    """
    Hash n-grams from each row; returns uint64 array of hashes.
    This is a proxy for overlap/memorization (NOT a formal privacy guarantee).
    """
    if k <= 0:
        return np.array([], dtype=np.uint64)
    e = events.astype(np.uint64, copy=False)
    N, L = e.shape
    if L < k:
        return np.array([], dtype=np.uint64)

    # rolling hash: h = (((x1 * P) + x2)*P + x3) ...
    P = np.uint64(1315423911)  # fixed prime-ish constant
    hashes = []
    for i in range(L - k + 1):
        window = e[:, i:i+k]  # (N,k)
        h = np.zeros(N, dtype=np.uint64)
        for j in range(k):
            h = h * P + window[:, j]
        hashes.append(h)
    return np.concatenate(hashes, axis=0) if hashes else np.array([], dtype=np.uint64)

def ngram_overlap_rate(real_hashes: np.ndarray, synth_hashes: np.ndarray) -> float:
    if real_hashes.size == 0 or synth_hashes.size == 0:
        return 0.0
    real_set = np.unique(real_hashes)
    synth_u = np.unique(synth_hashes)
    # membership test via searchsorted
    real_set.sort()
    idx = np.searchsorted(real_set, synth_u)
    hit = (idx < real_set.size) & (real_set[idx] == synth_u)
    return float(hit.mean())


# -----------------------------
# Per-synth evaluation
# -----------------------------
@dataclass
class SynthMetrics:
    path: str
    model_type: str  # "ar" or "diffusion"
    run_name: str

    num_windows: int
    seq_len: int

    # Marginals
    kl_event: float
    kl_dt: float
    kl_cpu: float
    js_event: float
    js_dt: float
    js_cpu: float

    # Joints
    js_event_cpu: float
    js_event_dt: float

    # Transitions
    js_event_bigram: float

    # Sanity signals
    dt0_rate: float
    cpu_switch_rate: float
    run_mean: float
    run_p95: float
    unique_event_frac: float
    top50_mass: float
    top50_overlap: float

    # Optional memorization proxy
    ngram_k: int
    ngram_overlap: float

def infer_meta(path: str) -> Tuple[str, str]:
    """
    Returns (model_type, run_name) inferred from path.
    """
    p = path.replace("\\", "/")
    if "/outputs/art_outputs/" in p and "/generated_traces/ar/" in p:
        # .../outputs/art_outputs/<run>/generated_traces/ar/...
        parts = p.split("/outputs/art_outputs/")[1].split("/")
        return "ar", parts[0]
    if "/outputs/diffusion_outputs/" in p and "/generated_traces/discrete/" in p:
        parts = p.split("/outputs/diffusion_outputs/")[1].split("/")
        return "diffusion", parts[0]
    return "unknown", "unknown"

def eval_one_synth(
    synth_path: str,
    real: RealBaseline,
    num_events: int,
    num_dt_buckets: int,
    num_cpus: int,
    real_topk_idx: np.ndarray,
    real_event_prob: np.ndarray,
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

    # Marginal KL/JSD (RealAgg || Synth) and JSD
    kl_e = kl_div(real.event_hist, s_ev)
    kl_d = kl_div(real.dt_hist, s_dt)
    kl_c = kl_div(real.cpu_hist, s_cpu)

    js_e = js_div(real.event_hist, s_ev)
    js_d = js_div(real.dt_hist, s_dt)
    js_c = js_div(real.cpu_hist, s_cpu)

    # Joint JSD
    s_ev_cpu = bincount_pairs(ev, cpu, E, C)
    s_ev_dt = bincount_pairs(ev, dt, E, D)
    js_ev_cpu = js_div(real.ev_cpu_hist, s_ev_cpu)
    js_ev_dt = js_div(real.ev_dt_hist, s_ev_dt)

    # Transition JSD
    s_bigr = event_bigrams(ev, E)
    js_bigr = js_div(real.bigram_hist, s_bigr)

    # Sanity signals
    dt0_rate = float((dt == 0).sum()) / float(max(1, dt.size))
    cpu_switch_rate = float((cpu[:, 1:] != cpu[:, :-1]).sum()) / float(max(1, cpu[:, 1:].size))
    run_m, run_p95 = run_length_stats(ev)

    ev_flat = ev.reshape(-1).astype(np.int64, copy=False)
    unique_event_frac = float(np.unique(ev_flat).size) / float(E) if E > 0 else 0.0

    s_ev_p = _safe_prob(s_ev)
    top50_mass = float(s_ev_p[real_topk_idx].sum())
    top50_ov = topk_overlap(real_event_prob, s_ev_p, k=50)

    # Optional n-gram overlap
    if ngram_k > 0:
        s_hash = hashed_ngrams(ev, ngram_k)
        ov = ngram_overlap_rate(real_ngram_hashes, s_hash)
    else:
        ov = 0.0

    model_type, run_name = infer_meta(synth_path)

    return SynthMetrics(
        path=synth_path,
        model_type=model_type,
        run_name=run_name,
        num_windows=int(N),
        seq_len=int(L),
        kl_event=float(kl_e),
        kl_dt=float(kl_d),
        kl_cpu=float(kl_c),
        js_event=float(js_e),
        js_dt=float(js_d),
        js_cpu=float(js_c),
        js_event_cpu=float(js_ev_cpu),
        js_event_dt=float(js_ev_dt),
        js_event_bigram=float(js_bigr),
        dt0_rate=float(dt0_rate),
        cpu_switch_rate=float(cpu_switch_rate),
        run_mean=float(run_m),
        run_p95=float(run_p95),
        unique_event_frac=float(unique_event_frac),
        top50_mass=float(top50_mass),
        top50_overlap=float(top50_ov),
        ngram_k=int(ngram_k),
        ngram_overlap=float(ov),
    )


def find_synth_files(repo: str, bench: str) -> List[str]:
    pats = [
        os.path.join(repo, "outputs", "art_outputs", "*", "generated_traces", "ar", bench, "**", "*.npz"),
        os.path.join(repo, "outputs", "diffusion_outputs", "*", "generated_traces", "discrete", bench, "*.npz"),
    ]
    out: List[str] = []
    for pat in pats:
        out.extend(glob.glob(pat, recursive=True))
    out = sorted(set(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--bench", default="compress-gzip")
    ap.add_argument("--split", default="train", choices=["train", "testing", "test", "val", "validation"])
    ap.add_argument("--num_events", type=int, default=380)
    ap.add_argument("--num_dt_buckets", type=int, default=256)
    ap.add_argument("--num_cpus", type=int, default=4)

    ap.add_argument("--max_shards", type=int, default=50)
    ap.add_argument("--max_windows_per_shard", type=int, default=100000)  # set lower if you want faster baseline

    ap.add_argument("--ngram_k", type=int, default=0, help="0 disables; try 8 as a memorization proxy")
    ap.add_argument("--out_dir", default="results/robustness_eval")
    ap.add_argument("--synth_glob", default="", help="optional override glob for synth files")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    bench = args.bench

    os.makedirs(os.path.join(repo, args.out_dir), exist_ok=True)

    real_glob = os.path.join(repo, "dataset", "window_shards", bench, args.split, "*.npz")
    print("[REAL] glob:", real_glob)

    real = build_real_baseline(
        real_glob=real_glob,
        num_events=args.num_events,
        num_dt_buckets=args.num_dt_buckets,
        num_cpus=args.num_cpus,
        max_shards=args.max_shards,
        max_windows_per_shard=args.max_windows_per_shard,
    )

    # Precompute for top-k overlap
    real_event_prob = _safe_prob(real.event_hist)
    real_topk_idx = np.argsort(-real_event_prob)[:50]

    # Optional: build real ngram hashes for overlap proxy
    if args.ngram_k > 0:
        # build hashes from the same real shards used in baseline (bounded)
        paths = sorted(glob.glob(real_glob))[: args.max_shards]
        real_hashes_all = []
        for p in paths:
            d = _load_npz(p)
            ev = d["event"][: args.max_windows_per_shard]
            real_hashes_all.append(hashed_ngrams(ev, args.ngram_k))
        real_ngram_hashes = np.concatenate(real_hashes_all) if real_hashes_all else np.array([], dtype=np.uint64)
        print(f"[REAL] ngram_k={args.ngram_k} hashes:", real_ngram_hashes.size)
    else:
        real_ngram_hashes = np.array([], dtype=np.uint64)

    # Find synth
    if args.synth_glob:
        synth_files = sorted(glob.glob(args.synth_glob, recursive=True))
    else:
        synth_files = find_synth_files(repo, bench)

    if not synth_files:
        raise FileNotFoundError("No synthetic .npz files found. Check your outputs directory or --synth_glob.")

    print("[SYNTH] files:", len(synth_files))

    results: List[SynthMetrics] = []
    for i, sp in enumerate(synth_files, 1):
        print(f"[{i}/{len(synth_files)}] eval {sp}")
        m = eval_one_synth(
            synth_path=sp,
            real=real,
            num_events=args.num_events,
            num_dt_buckets=args.num_dt_buckets,
            num_cpus=args.num_cpus,
            real_topk_idx=real_topk_idx,
            real_event_prob=real_event_prob,
            real_ngram_hashes=real_ngram_hashes,
            ngram_k=args.ngram_k,
        )
        results.append(m)

    # Sort by a simple composite "robustness distance" (lower is better)
    def score(x: SynthMetrics) -> float:
        return (
            x.js_event
            + x.js_dt
            + x.js_cpu
            + x.js_event_cpu
            + x.js_event_dt
            + x.js_event_bigram
        )

    ranked = sorted(results, key=score)

    # Write CSV
    csv_path = os.path.join(repo, args.out_dir, f"robust_eval_{bench}_{args.split}.csv")
    cols = list(asdict(results[0]).keys())
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            row = asdict(r)
            f.write(",".join(str(row[c]) for c in cols) + "\n")

    # Write JSON
    json_path = os.path.join(repo, args.out_dir, f"robust_eval_{bench}_{args.split}.json")
    payload = {
        "bench": bench,
        "split": args.split,
        "real_baseline": asdict(real),
        "score_definition": "js_event + js_dt + js_cpu + js_event_cpu + js_event_dt + js_event_bigram",
        "top_ranked": [{"path": r.path, "score": score(r)} for r in ranked[:10]],
        "results": [asdict(r) for r in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print()
    print("[WROTE]", csv_path)
    print("[WROTE]", json_path)
    print()
    print("[TOP 10] (lower composite score is better):")
    for j, r in enumerate(ranked[:10], 1):
        print(f"{j:2d}. score={score(r):.6f}  model={r.model_type:9s} run={r.run_name:10s}  {os.path.basename(r.path)}")


if __name__ == "__main__":
    main()