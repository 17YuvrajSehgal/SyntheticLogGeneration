import argparse, glob
import numpy as np

EPS = 1e-12

def safe_normalize_rows(mat):
    # mat: [A,B] counts
    row_sums = mat.sum(axis=1, keepdims=True).astype(np.float64)
    return (mat.astype(np.float64) + EPS) / (row_sums + EPS * mat.shape[1])

def safe_normalize(vec):
    s = vec.sum().astype(np.float64)
    return (vec.astype(np.float64) + EPS) / (s + EPS * len(vec))

def kl_div(p, q):
    # p,q already probability vectors or matrices (same shape)
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
    # Marginals
    cpu_flat = cpu.reshape(-1).astype(np.int64)
    cpu_flat = np.clip(cpu_flat, 0, num_cpus - 1)
    out_marg_cpu += np.bincount(cpu_flat, minlength=num_cpus)
    # Bigram transitions along time dimension
    e0 = event[:, :-1].reshape(-1)
    e1 = event[:, 1: ].reshape(-1)
    idx = e0 * num_events + e1
    out_bigram_event.reshape(-1)[:] += np.bincount(idx, minlength=num_events * num_events)

    c0 = np.clip(cpu[:, :-1], 0, num_cpus - 1).reshape(-1).astype(np.int64)
    c1 = np.clip(cpu[:, 1:], 0, num_cpus - 1).reshape(-1).astype(np.int64)
    idxc = c0 * num_cpus + c1
    out_bigram_cpu.reshape(-1)[:] += np.bincount(idxc, minlength=num_cpus * num_cpus)

    # Joint P(event, cpu) at same position
    cpu_flat = np.clip(cpu.reshape(-1), 0, num_cpus - 1).astype(np.int64)
    ec_idx = event.reshape(-1).astype(np.int64) * num_cpus + cpu_flat

    out_joint_event_cpu.reshape(-1)[:] += np.bincount(ec_idx, minlength=num_events * num_cpus)

def iter_real_shards(real_glob, max_shards=None):
    paths = sorted(glob.glob(real_glob))
    if not paths:
        raise FileNotFoundError(f"No shards match: {real_glob}")
    if max_shards is not None:
        paths = paths[:max_shards]
    for p in paths:
        d = np.load(p)
        yield p, d["event"], d["dt"], d["cpu"]
        d.close()

def load_synth_npz(path):
    d = np.load(path)
    event, dt, cpu = d["event"], d["dt"], d["cpu"]
    d.close()
    return event, dt, cpu

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

    args = ap.parse_args()

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

    # Optional sanity warning (does not change results; just informs you)
    if scpu.max() >= C or scpu.min() < 0:
        print(
            f"[WARN] Synth cpu out of range: min={int(scpu.min())}, max={int(scpu.max())}, expected [0, {C - 1}]. Clipping during evaluation.")
    if sev.max() >= E or sev.min() < 0:
        print(f"[WARN] Synth event out of range: min={int(sev.min())}, max={int(sev.max())}, expected [0, {E - 1}].")
    if sdt.max() >= D or sdt.min() < 0:
        print(f"[WARN] Synth dt out of range: min={int(sdt.min())}, max={int(sdt.max())}, expected [0, {D - 1}].")

    accumulate_from_arrays(sev, sdt, scpu, E, D, C,
                           synth_marg_e, synth_marg_d, synth_marg_c,
                           synth_bigram_e, synth_bigram_c,
                           synth_joint_ec)

    # --- Convert to probabilities ---
    # Marginals
    p_re = safe_normalize(real_marg_e)
    p_se = safe_normalize(synth_marg_e)
    p_rd = safe_normalize(real_marg_d)
    p_sd = safe_normalize(synth_marg_d)
    p_rc = safe_normalize(real_marg_c)
    p_sc = safe_normalize(synth_marg_c)

    # Conditionals
    P_re = safe_normalize_rows(real_bigram_e)   # P(next_event | event)
    P_se = safe_normalize_rows(synth_bigram_e)

    P_rc = safe_normalize_rows(real_bigram_c)   # P(next_cpu | cpu)
    P_sc = safe_normalize_rows(synth_bigram_c)

    # Joint event-cpu
    p_rec = safe_normalize(real_joint_ec.reshape(-1)).reshape(E, C)
    p_sec = safe_normalize(synth_joint_ec.reshape(-1)).reshape(E, C)

    # --- Metrics ---
    print("Real shards used:", used)
    print("Synth windows:", sev.shape[0], "seq_len:", sev.shape[1])

    print("\n=== Marginal KL (RealAgg || Synth) ===")
    print("Event KL:", kl_div(p_re, p_se))
    print("DT KL   :", kl_div(p_rd, p_sd))
    print("CPU KL  :", kl_div(p_rc, p_sc))

    # Event transition matrix metrics
    # Weighted KL over rows: sum_e p_real(e) * KL(P_real(.|e) || P_synth(.|e))
    row_kls = np.zeros(E, dtype=np.float64)
    for e in range(E):
        row_kls[e] = kl_div(P_re[e], P_se[e])
    weighted_event_bigram_kl = float(np.sum(p_re * row_kls))

    print("\n=== Event Transition Matrix: P(e_{t+1} | e_t) ===")
    print("Weighted row KL:", weighted_event_bigram_kl)
    print("L1 distance     :", l1_dist(P_re, P_se))
    print("Frobenius norm   :", frob_norm(P_re, P_se))
    print("Cosine similarity:", cosine_sim(P_re, P_se))

    # CPU transition metrics
    cpu_row_kls = np.zeros(C, dtype=np.float64)
    for c in range(C):
        cpu_row_kls[c] = kl_div(P_rc[c], P_sc[c])
    weighted_cpu_bigram_kl = float(np.sum(p_rc * cpu_row_kls))

    print("\n=== CPU Transition Matrix: P(c_{t+1} | c_t) ===")
    print("Weighted row KL:", weighted_cpu_bigram_kl)
    print("L1 distance     :", l1_dist(P_rc, P_sc))
    print("Frobenius norm  :", frob_norm(P_rc, P_sc))
    print("Cosine similarity:", cosine_sim(P_rc, P_sc))

    # Event-CPU joint metrics
    print("\n=== Joint Event-CPU: P(event, cpu) ===")
    print("KL (Real||Synth):", kl_div(p_rec, p_sec))
    print("L1 distance     :", l1_dist(p_rec, p_sec))
    print("Cosine similarity:", cosine_sim(p_rec, p_sec))

    # --- Timing conditioned on event (top-K by real freq, but require min occ) ---
    # Select candidate events by real marginal frequency
    top = np.argsort(real_marg_e)[::-1]
    candidates = [int(e) for e in top if real_marg_e[e] >= args.min_occ_dt]
    candidates = candidates[:args.topk_events_dt]

    print("\n=== Timing conditioned on event: P(dt | event=e) ===")
    print(f"Using topK={args.topk_events_dt}, min_occ={args.min_occ_dt}, actual_used={len(candidates)}")

    # Build dt hist per event for real + synth for selected events
    real_dt_given_e = {e: np.zeros(D, dtype=np.int64) for e in candidates}
    synth_dt_given_e = {e: np.zeros(D, dtype=np.int64) for e in candidates}

    # Pass over real shards again for dt|event
    for _, ev, dt, _ in iter_real_shards(args.real_glob, args.max_real_shards):
        flat_e = ev.reshape(-1)
        flat_d = dt.reshape(-1).astype(np.int64)
        for e in candidates:
            m = (flat_e == e)
            if m.any():
                real_dt_given_e[e] += np.bincount(flat_d[m], minlength=D)

    # Synth
    flat_se = sev.reshape(-1)
    flat_sd = sdt.reshape(-1).astype(np.int64)
    for e in candidates:
        m = (flat_se == e)
        if m.any():
            synth_dt_given_e[e] += np.bincount(flat_sd[m], minlength=D)

    # Compute per-event KL + summary
    per_event = []
    for e in candidates:
        pr = safe_normalize(real_dt_given_e[e])
        ps = safe_normalize(synth_dt_given_e[e])
        k = kl_div(pr, ps)
        per_event.append((k, e, int(real_dt_given_e[e].sum()), int(synth_dt_given_e[e].sum())))

    if per_event:
        per_event.sort(reverse=True)  # worst first
        kls = np.array([x[0] for x in per_event], dtype=np.float64)
        print("Mean KL(dt|e):", float(kls.mean()))
        print("Median KL(dt|e):", float(np.median(kls)))
        print("\nWorst 10 events by KL(dt|e):")
        for k, e, nr, ns in per_event[:10]:
            print(f"  event={e:3d}  KL={k:.4f}  real_occ={nr}  synth_occ={ns}")
    else:
        print("No events met the occurrence threshold; lower --min_occ_dt.")

    # Optional: show most mismatched event transition rows (by KL)
    print("\n=== Most mismatched event transition rows (by KL) ===")
    worst_rows = np.argsort(row_kls)[::-1][:10]
    for e in worst_rows:
        print(f"  event={int(e):3d} row_KL={float(row_kls[e]):.4f}  real_count={int(real_marg_e[e])}  synth_count={int(synth_marg_e[e])}")

if __name__ == "__main__":
    main()
