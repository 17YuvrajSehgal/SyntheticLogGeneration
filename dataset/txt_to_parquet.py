#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# Example line:
# [19:13:13.855396359] (+0.000002664) flutin kmem_kfree: { cpu_id = 0 }, { call_site = ... }
LINE_RE = re.compile(
    r'^\[(?P<ts>[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+)\]\s+'
    r'\(\+(?P<delta>[0-9]*\.?[0-9]+)\)\s+'
    r'(?P<host>\S+)\s+'
    r'(?P<event>[^:]+):\s+'
    r'(?P<rest>.*)$'
)

CPU_RE = re.compile(r'\{\s*cpu_id\s*=\s*(?P<cpu>\d+)\s*\}')

# Filename patterns you likely have:
# ramspeed-all-events-run0.txt
# compress-gzip-all-events-run12.txt
FNAME_RE = re.compile(r'^(?P<dataset>.+?)-all-events-(?P<run>run\d+)\.txt$')


def infer_dataset_run(filename: str, dataset_arg: str | None, run_arg: str | None):
    if dataset_arg and run_arg:
        return dataset_arg, run_arg

    m = FNAME_RE.match(filename)
    if m:
        return m.group("dataset"), m.group("run")

    # fallback: dataset from parent dir name, run unknown
    dataset = dataset_arg or Path(filename).stem
    run_id = run_arg or "run0"
    return dataset, run_id


def write_parquet_stream(
    input_path: Path,
    output_path: Path,
    dataset: str,
    run_id: str,
    batch_size: int = 200_000,
):
    # Arrow schema (stable types)
    schema = pa.schema([
        ("dataset", pa.string()),
        ("run_id", pa.string()),
        ("line_idx", pa.int64()),
        ("ts_str", pa.string()),
        ("delta_s", pa.float64()),
        ("t_rel_ns", pa.int64()),
        ("event_name", pa.string()),
        ("cpu_id", pa.int32()),
        ("raw_line", pa.string()),
    ])

    writer = pq.ParquetWriter(
        output_path.as_posix(),
        schema=schema,
        compression="zstd",   # great compression; change to "snappy" if needed
        use_dictionary=True,
    )

    # Buffers for one batch
    cols = {name: [] for name in schema.names}

    t_rel_ns = 0
    line_idx = 0
    parsed = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw_line = raw.rstrip("\n")

            m = LINE_RE.match(raw_line)
            if not m:
                skipped += 1
                continue

            ts_str = m.group("ts")
            delta_s = float(m.group("delta"))
            event_name = m.group("event").strip()
            rest = m.group("rest")

            # Prefer cpu_id from the first { cpu_id = ... } block
            cpu_m = CPU_RE.search(rest)
            cpu_id = int(cpu_m.group("cpu")) if cpu_m else -1

            # cumulative monotonic time in ns
            # (use rounding to nearest ns for stability)
            t_rel_ns += int(round(delta_s * 1_000_000_000))

            # append to batch buffers
            cols["dataset"].append(dataset)
            cols["run_id"].append(run_id)
            cols["line_idx"].append(line_idx)
            cols["ts_str"].append(ts_str)
            cols["delta_s"].append(delta_s)
            cols["t_rel_ns"].append(t_rel_ns)
            cols["event_name"].append(event_name)
            cols["cpu_id"].append(cpu_id)
            cols["raw_line"].append(raw_line)

            line_idx += 1
            parsed += 1

            if parsed % batch_size == 0:
                table = pa.Table.from_pydict(cols, schema=schema)
                writer.write_table(table)
                # reset buffers
                cols = {name: [] for name in schema.names}

    # flush remainder
    if cols["line_idx"]:
        table = pa.Table.from_pydict(cols, schema=schema)
        writer.write_table(table)

    writer.close()

    print(f"[OK] {input_path.name} -> {output_path}")
    print(f"     parsed={parsed}, skipped={skipped}, last_t_rel_ns={t_rel_ns}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input .txt trace file (babeltrace2 output)")
    ap.add_argument("--output", required=True, help="Output .parquet file path")
    ap.add_argument("--dataset", default=None, help="Override dataset name (otherwise inferred from filename)")
    ap.add_argument("--run-id", default=None, help="Override run_id like run0 (otherwise inferred from filename)")
    ap.add_argument("--batch-size", type=int, default=200_000, help="Rows per Parquet write batch")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset, run_id = infer_dataset_run(input_path.name, args.dataset, args.run_id)

    write_parquet_stream(
        input_path=input_path,
        output_path=output_path,
        dataset=dataset,
        run_id=run_id,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()