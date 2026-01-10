# python -u evaluation_metrics/transition_eval.py   --real_glob "dataset/window_shards/compress-gzip/train/*.npz"   --synth outputs/diffusion_outputs/generated_traces/compress-gzip/synth_100k_step50k_repaired.npz   --max_real_shards 50   --num_events 384   --num_dt_buckets 256   --num_cpus 4   --out_dir "evaluation_results/repairedSynth_vs_real"

# python -u evaluation_metrics/transition_eval.py   --real_glob "dataset/window_shards/compress-gzip/train/*.npz"   --synth outputs/diffusion_outputs/generated_traces/compress-gzip/synth_100k_step50k.npz   --max_real_shards 50   --num_events 384   --num_dt_buckets 256   --num_cpus 4   --out_dir "evaluation_results/originalSynth_vs_real"

import argparse, glob
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

EPS = 1e-12

def safe_normalize_rows(mat):
    # mat: [A,B] counts
    row_sums = mat.sum(axis=1, keepdims=True).astype(np.float64)
    return (mat.astype(np.float64) + EPS) / (row_sums + EPS * mat.shape[1])

def safe_normalize(vec):
    s = vec.sum().astype(np.float64)
    return (vec.astype(np.float64) + EPS) / (s + EPS * len(vec))

def kl_div(p, q):
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    return float(np.sum(p * (np.log(p + EPS) - np.log(q + EPS))))

def l1_dist(p, q):
    return float(np.sum(np.abs(p - q)))

def frob_norm(p, q):
    d = p - q
    return float(np.sqrt(np.sum(d * d)))

def cosine_sim(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def accumulate_from_arrays(event, dt, cpu,
                           num_events, num_dt, num_cpus,
                           out_marg_event, out_marg_dt, out_marg_cpu,
                           out_bigram_event, out_bigram_cpu,
                           out_joint_event_cpu):
    # event, dt, cpu: [B,L]
    # Flatten & clip to valid ranges
    e_flat = np.clip(event.reshape(-1).astype(np.int64), 0, num_events - 1)
    d_flat = np.clip(dt.reshape(-1).astype(np.int64),    0, num_dt - 1)
    c_flat = np.clip(cpu.reshape(-1).astype(np.int64),   0, num_cpus - 1)

    # ---- Marginals (FIX) ----
    out_marg_event += np.bincount(e_flat, minlength=num_events)
    out_marg_dt    += np.bincount(d_flat, minlength=num_dt)
    out_marg_cpu   += np.bincount(c_flat, minlength=num_cpus)

    # ---- Event bigrams ----
    e0 = np.clip(event[:, :-1].reshape(-1).astype(np.int64), 0, num_events - 1)
    e1 = np.clip(event[:,  1:].reshape(-1).astype(np.int64), 0, num_events - 1)
    idx = e0 * num_events + e1
    out_bigram_event.reshape(-1)[:] += np.bincount(idx, minlength=num_events * num_events)

    # ---- CPU bigrams ----
    c0 = np.clip(cpu[:, :-1].reshape(-1).astype(np.int64), 0, num_cpus - 1)
    c1 = np.clip(cpu[:,  1:].reshape(-1).astype(np.int64), 0, num_cpus - 1)
    idxc = c0 * num_cpus + c1
    out_bigram_cpu.reshape(-1)[:] += np.bincount(idxc, minlength=num_cpus * num_cpus)

    # ---- Joint P(event, cpu) ----
    ec_idx = e_flat * num_cpus + c_flat
    out_joint_event_cpu.reshape(-1)[:] += np.bincount(ec_idx, minlength=num_events * num_cpus)

def iter_real_shards(real_glob, max_shards=None):
    paths = sorted(glob.glob(real_glob))
    if not paths:
        raise FileNotFoundError(f"No shards match: {real_glob}")
    if max_shards is not None:
        paths = paths[:max_shards]
    for p in paths:
        with np.load(p) as d:
            yield p, d["event"], d["dt"], d["cpu"]

def load_synth_npz(path):
    with np.load(path) as d:
        return d["event"], d["dt"], d["cpu"]

# --- PLOTTING HELPERS ---
def plot_transition_heatmap(mat_real, mat_synth, title, out_path, top_k=20):
    """
    Plot heatmaps of the top-k most frequent real transitions.
    """
    # Simply sum rows to get marginals, then pick top events
    # Or just use the global marginals if passed.
    # Let's just use row sums of mat_real
    row_sums = mat_real.sum(axis=1)
    top_indices = np.argsort(row_sums)[::-1][:top_k]
    
    # Subset matrices
    sub_real = mat_real[top_indices][:, top_indices]
    sub_synth = mat_synth[top_indices][:, top_indices]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Use log scale for heatmaps usually better for long-tailed logs
    # But these are probabilities? If probabilities, raw is fine. 
    # If they are counts, log. These are probabilities (normalized).
    
    sns.heatmap(sub_real, ax=axes[0], cmap="viridis", cbar=True, vmin=0, vmax=1)
    axes[0].set_title(f"Real {title} (Top {top_k})")
    
    sns.heatmap(sub_synth, ax=axes[1], cmap="viridis", cbar=True, vmin=0, vmax=1)
    axes[1].set_title(f"Synth {title} (Top {top_k})")
    
    diff = np.abs(sub_real - sub_synth)
    sns.heatmap(diff, ax=axes[2], cmap="Reds", cbar=True, vmin=0, vmax=0.5)
    axes[2].set_title(f"Difference (L1 Error)")
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_marginal_comparison(p_real, p_synth, curr_name, out_path, top_k=50):
    """
    Bar plot of top-k items.
    """
    indices = np.argsort(p_real)[::-1][:top_k]
    
    vals_real = p_real[indices]
    vals_synth = p_synth[indices]
    
    labels = indices # just IDs
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - width/2, vals_real, width, label='Real')
    ax.bar(x + width/2, vals_synth, width, label='Synth')
    
    ax.set_ylabel('Probability')
    ax.set_title(f'{curr_name} Marginal Distribution (Top {top_k})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_glob", required=True, help='e.g. "window_shards/compress-gzip/train/*.npz"')
    ap.add_argument("--synth", required=True, help="synthetic .npz from sample_synthetic")
    ap.add_argument("--max_real_shards", type=int, default=50, help="how many real shards to aggregate")
    
    ap.add_argument("--num_events", type=int, required=True)
    ap.add_argument("--num_dt_buckets", type=int, default=256)
    ap.add_argument("--num_cpus", type=int, default=4)
    
    ap.add_argument("--topk_events_dt", type=int, default=30, help="top-K events for P(dt|event) analysis")
    ap.add_argument("--min_occ_dt", type=int, default=5000, help="min occurrences to include event in dt|event stats")
    
    ap.add_argument("--out_dir", type=str, default="evaluation_results", help="Directory to save plots and json")

    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)

    E = args.num_events
    D = args.num_dt_buckets
    C = args.num_cpus

    # --- PASS 1: aggregate real counts (marginals + bigrams + joint) ---
    real_marg_e = np.zeros(E, dtype=np.int64)
    real_marg_d = np.zeros(D, dtype=np.int64)
    real_marg_c = np.zeros(C, dtype=np.int64)
    real_bigram_e = np.zeros((E, E), dtype=np.int64)
    real_bigram_c = np.zeros((C, C), dtype=np.int64)
    real_joint_ec = np.zeros((E, C), dtype=np.int64)

    used = 0
    for _, ev, dt, cpu in iter_real_shards(args.real_glob, args.max_real_shards):
        accumulate_from_arrays(ev, dt, cpu, E, D, C,
                               real_marg_e, real_marg_d, real_marg_c,
                               real_bigram_e, real_bigram_c,
                               real_joint_ec)
        used += 1

    # --- SYNTH aggregate counts ---
    synth_marg_e = np.zeros(E, dtype=np.int64)
    synth_marg_d = np.zeros(D, dtype=np.int64)
    synth_marg_c = np.zeros(C, dtype=np.int64)
    synth_bigram_e = np.zeros((E, E), dtype=np.int64)
    synth_bigram_c = np.zeros((C, C), dtype=np.int64)
    synth_joint_ec = np.zeros((E, C), dtype=np.int64)

    sev, sdt, scpu = load_synth_npz(args.synth)

    # Optional warnings
    if scpu.max() >= C or scpu.min() < 0:
        print(f"[WARN] Synth cpu out of range: min={int(scpu.min())}, max={int(scpu.max())}, expected [0,{C-1}]. (Clipped in eval)")
    if sev.max() >= E or sev.min() < 0:
        print(f"[WARN] Synth event out of range: min={int(sev.min())}, max={int(sev.max())}, expected [0,{E-1}]. (Clipped in eval)")
    if sdt.max() >= D or sdt.min() < 0:
        print(f"[WARN] Synth dt out of range: min={int(sdt.min())}, max={int(sdt.max())}, expected [0,{D-1}]. (Clipped in eval)")

    accumulate_from_arrays(sev, sdt, scpu, E, D, C,
                           synth_marg_e, synth_marg_d, synth_marg_c,
                           synth_bigram_e, synth_bigram_c,
                           synth_joint_ec)

    # --- Convert to probabilities ---
    p_re = safe_normalize(real_marg_e)
    p_se = safe_normalize(synth_marg_e)
    p_rd = safe_normalize(real_marg_d)
    p_sd = safe_normalize(synth_marg_d)
    p_rc = safe_normalize(real_marg_c)
    p_sc = safe_normalize(synth_marg_c)

    P_re = safe_normalize_rows(real_bigram_e)   # P(next_event | event)
    P_se = safe_normalize_rows(synth_bigram_e)

    P_rc = safe_normalize_rows(real_bigram_c)   # P(next_cpu | cpu)
    P_sc = safe_normalize_rows(synth_bigram_c)

    p_rec = safe_normalize(real_joint_ec.reshape(-1)).reshape(E, C)
    p_sec = safe_normalize(synth_joint_ec.reshape(-1)).reshape(E, C)

    # --- Metrics ---
    results = {}
    
    print("Real shards used:", used)
    print("Synth windows:", sev.shape[0], "seq_len:", sev.shape[1])
    
    results["metrics_kl"] = {
        "event_marginal": kl_div(p_re, p_se),
        "dt_marginal": kl_div(p_rd, p_sd),
        "cpu_marginal": kl_div(p_rc, p_sc)
    }

    print("\n=== Marginal KL (RealAgg || Synth) ===")
    print("Event KL:", results["metrics_kl"]["event_marginal"])
    print("DT KL   :", results["metrics_kl"]["dt_marginal"])
    print("CPU KL  :", results["metrics_kl"]["cpu_marginal"])
    
    # Plot Marginals
    plot_marginal_comparison(p_re, p_se, "Event", os.path.join(args.out_dir, "event_marginal_top50.png"), top_k=50)
    plot_marginal_comparison(p_rd, p_sd, "DT", os.path.join(args.out_dir, "dt_marginal_top50.png"), top_k=50)
    # CPU is small enough to plot all
    plot_marginal_comparison(p_rc, p_sc, "CPU", os.path.join(args.out_dir, "cpu_marginal.png"), top_k=C)


    # ---- Event transition metrics ----
    row_kls = np.zeros(E, dtype=np.float64)
    for e in range(E):
        row_kls[e] = kl_div(P_re[e], P_se[e])

    valid_rows = (real_bigram_e.sum(axis=1) > 0)
    weighted_event_bigram_kl = float(np.sum(p_re[valid_rows] * row_kls[valid_rows]))
    
    results["transition_event"] = {
        "weighted_row_kl": weighted_event_bigram_kl,
        "l1_distance": l1_dist(P_re, P_se),
        "frobenius_norm": frob_norm(P_re, P_se),
        "cosine_sim": cosine_sim(P_re, P_se)
    }

    print("\n=== Event Transition Matrix: P(e_{t+1} | e_t) ===")
    print("Weighted row KL (real-supported rows only):", weighted_event_bigram_kl)
    print("L1 distance     :", results["transition_event"]["l1_distance"])
    print("Frobenius norm   :", results["transition_event"]["frobenius_norm"])
    print("Cosine similarity:", results["transition_event"]["cosine_sim"])
    
    # Plot Heatmap
    plot_transition_heatmap(P_re, P_se, "Event Transition", os.path.join(args.out_dir, "event_transition_heatmap.png"), top_k=20)

    # ---- CPU transition metrics ----
    cpu_row_kls = np.zeros(C, dtype=np.float64)
    for c in range(C):
        cpu_row_kls[c] = kl_div(P_rc[c], P_sc[c])
    weighted_cpu_bigram_kl = float(np.sum(p_rc * cpu_row_kls))
    
    results["transition_cpu"] = {
        "weighted_row_kl": weighted_cpu_bigram_kl,
        "l1_distance": l1_dist(P_rc, P_sc),
        "frobenius_norm": frob_norm(P_rc, P_sc),
        "cosine_sim": cosine_sim(P_rc, P_sc)
    }

    print("\n=== CPU Transition Matrix: P(c_{t+1} | c_t) ===")
    print("Weighted row KL:", weighted_cpu_bigram_kl)
    print("L1 distance     :", results["transition_cpu"]["l1_distance"])
    print("Frobenius norm  :", results["transition_cpu"]["frobenius_norm"])
    print("Cosine similarity:", results["transition_cpu"]["cosine_sim"])

    # ---- Joint metrics ----
    results["joint_event_cpu"] = {
        "kl": kl_div(p_rec, p_sec),
        "l1_distance": l1_dist(p_rec, p_sec),
        "cosine_sim": cosine_sim(p_rec, p_sec)
    }
    
    print("\n=== Joint Event-CPU: P(event, cpu) ===")
    print("KL (Real||Synth):", results["joint_event_cpu"]["kl"])
    print("L1 distance     :", results["joint_event_cpu"]["l1_distance"])
    print("Cosine similarity:", results["joint_event_cpu"]["cosine_sim"])

    # --- Timing conditioned on event ---
    top = np.argsort(real_marg_e)[::-1]
    candidates = [int(e) for e in top if real_marg_e[e] >= args.min_occ_dt]
    candidates = candidates[:args.topk_events_dt]

    print("\n=== Timing conditioned on event: P(dt | event=e) ===")
    print(f"Using topK={args.topk_events_dt}, min_occ={args.min_occ_dt}, actual_used={len(candidates)}")

    real_dt_given_e  = {e: np.zeros(D, dtype=np.int64) for e in candidates}
    synth_dt_given_e = {e: np.zeros(D, dtype=np.int64) for e in candidates}

    # Real dt|event
    for _, ev, dt, _ in iter_real_shards(args.real_glob, args.max_real_shards):
        flat_e = np.clip(ev.reshape(-1).astype(np.int64), 0, E - 1)
        flat_d = np.clip(dt.reshape(-1).astype(np.int64), 0, D - 1)
        for e in candidates:
            m = (flat_e == e)
            if m.any():
                real_dt_given_e[e] += np.bincount(flat_d[m], minlength=D)

    # Synth dt|event
    flat_se = np.clip(sev.reshape(-1).astype(np.int64), 0, E - 1)
    flat_sd = np.clip(sdt.reshape(-1).astype(np.int64), 0, D - 1)
    for e in candidates:
        m = (flat_se == e)
        if m.any():
            synth_dt_given_e[e] += np.bincount(flat_sd[m], minlength=D)

    per_event = []
    for e in candidates:
        pr = safe_normalize(real_dt_given_e[e])
        ps = safe_normalize(synth_dt_given_e[e])
        per_event.append((kl_div(pr, ps), e, int(real_dt_given_e[e].sum()), int(synth_dt_given_e[e].sum())))

    if per_event:
        per_event.sort(reverse=True)
        kls = np.array([x[0] for x in per_event], dtype=np.float64)
        print("Mean KL(dt|e):", float(kls.mean()))
        print("Median KL(dt|e):", float(np.median(kls)))
        print("\nWorst 10 events by KL(dt|e):")
        worst_list = []
        for k, e, nr, ns in per_event[:10]:
            print(f"  event={e:3d}  KL={k:.4f}  real_occ={nr}  synth_occ={ns}")
            worst_list.append({"event": e, "kl": k, "real_occ": nr, "synth_occ": ns})
        
        results["dt_conditional"] = {
            "mean_kl": float(kls.mean()),
            "median_kl": float(np.median(kls)),
            "worst_10": worst_list
        }
    else:
        print("No events met the occurrence threshold; lower --min_occ_dt.")
        results["dt_conditional"] = {}

    # ---- Most mismatched event transition rows (real_count>0 only) ----
    print("\n=== Most mismatched event transition rows (by KL, real_count>0) ===")
    worst = np.argsort(row_kls)[::-1]
    shown = 0
    worst_rows = []
    for e in worst:
        if real_marg_e[e] == 0:
            continue
        print(f"  event={int(e):3d} row_KL={float(row_kls[e]):.4f}  real_count={int(real_marg_e[e])}  synth_count={int(synth_marg_e[e])}")
        worst_rows.append({
            "event": int(e),
            "row_kl": float(row_kls[e]),
            "real_count": int(real_marg_e[e]),
            "synth_count": int(synth_marg_e[e])
        })
        shown += 1
        if shown >= 10:
            break
    results["worst_transition_rows"] = worst_rows
            
    # Save JSON
    json_path = os.path.join(args.out_dir, "metrics.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Saved results to {args.out_dir}")

if __name__ == "__main__":
    main()