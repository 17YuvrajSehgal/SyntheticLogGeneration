#!/usr/bin/env python3
"""
Build a global vocabulary (event_name -> event_id) from LTTng text traces.

Example input line:
[09:00:05.211376257] (+?.?????????) flutin kmem_kmalloc: { cpu_id = 0 }, { ... }

We extract the event name as the token right before the ':' (e.g., kmem_kmalloc).

Outputs (in --out_dir):
- vocab.json           : {event_name: event_id}
- id_to_event.json     : {event_id: event_name}
- vocab_stats.json     : counts and metadata
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


# Matches "... <event_name>: { cpu_id = ... }"
# We rely on "flutin <event>:" pattern seen in your sample.
EVENT_RE = re.compile(r"\]\s+\([^)]+\)\s+\S+\s+(?P<event>[^:\s]+)\s*:")


def iter_trace_files(root: Path, pattern: str) -> Iterable[Path]:
    """Yield files under root matching glob pattern (recursive)."""
    yield from root.rglob(pattern)


def extract_event_name(line: str) -> str | None:
    """Return event name if line matches expected format, else None."""
    m = EVENT_RE.search(line)
    if not m:
        return None
    return m.group("event")


def build_vocab(files: list[Path], max_lines_per_file: int | None = None) -> Counter:
    """
    Scan files and count event occurrences.
    max_lines_per_file can be used for quick tests (None = scan entire file).
    """
    counts = Counter()

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if max_lines_per_file is not None and i >= max_lines_per_file:
                        break
                    ev = extract_event_name(line)
                    if ev:
                        counts[ev] += 1
        except Exception as e:
            print(f"[WARN] Failed reading {fp}: {e}")

    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing txt traces (e.g., /home/yuvraj17/scratch/txt_traces_all_benchmarks)",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="*-all-events-run*.txt",
        help="Glob pattern to match trace files (recursive). Default matches all-events runs.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="metadata",
        help="Directory to write vocab outputs (default: ./metadata)",
    )
    ap.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Only keep events that appear at least this many times.",
    )
    ap.add_argument(
        "--max_lines_per_file",
        type=int,
        default=None,
        help="For debugging: only scan first N lines per file.",
    )
    ap.add_argument(
        "--sort",
        type=str,
        choices=["freq", "alpha"],
        default="freq",
        help="How to order event IDs: by frequency (desc) or alphabetically.",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(iter_trace_files(root, args.glob))
    if not files:
        raise SystemExit(f"No files found under {root} matching pattern: {args.glob}")

    print(f"[INFO] Found {len(files)} files. Scanning...")

    counts = build_vocab(files, max_lines_per_file=args.max_lines_per_file)

    # Filter
    counts = Counter({k: v for k, v in counts.items() if v >= args.min_count})

    if not counts:
        raise SystemExit("No events extracted. Check your input format or regex.")

    # Sort events
    if args.sort == "freq":
        # Most frequent gets smallest ID (common in NLP)
        ordered = [ev for ev, _ in counts.most_common()]
    else:
        ordered = sorted(counts.keys())

    vocab = {ev: idx for idx, ev in enumerate(ordered)}
    id_to_event = {str(idx): ev for ev, idx in vocab.items()}

    vocab_path = out_dir / "vocab.json"
    id_to_event_path = out_dir / "id_to_event.json"
    stats_path = out_dir / "vocab_stats.json"

    vocab_path.write_text(json.dumps(vocab, indent=2, sort_keys=False))
    id_to_event_path.write_text(json.dumps(id_to_event, indent=2, sort_keys=False))

    stats = {
        "root": str(root),
        "glob": args.glob,
        "num_files": len(files),
        "num_unique_events": len(vocab),
        "min_count": args.min_count,
        "sort": args.sort,
        "total_events_counted": int(sum(counts.values())),
        "top_50": counts.most_common(50),
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"[OK] Wrote:")
    print(f"  - {vocab_path}")
    print(f"  - {id_to_event_path}")
    print(f"  - {stats_path}")
    print(f"[INFO] Unique events: {len(vocab)} | Total counted: {sum(counts.values())}")


if __name__ == "__main__":
    main()