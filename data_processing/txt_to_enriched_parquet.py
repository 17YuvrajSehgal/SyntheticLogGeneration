#!/usr/bin/env python3
import argparse
import os
import re
import json
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Any, List

# --- Regex Setup ---

# Line Parser: [TIMESTAMP] (+DELTA) hostname event_name: { payload }
# Updated to optionally match (? ...) for delta
LINE_RE = re.compile(
    r'^\[(?P<ts>[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+)\]\s+'
    r'\(\+(?P<delta>[^)]+)\)\s+' # Capture everything inside (+...)
    r'(?P<host>\S+)\s+'
    r'(?P<event>[^:]+):\s+'
    r'(?P<rest>.*)$'
)

# Key-Value Parser (Same as before)
KV_RE = re.compile(
    r'(?P<key>[a-zA-Z0-9_]+)\s*=\s*'
    r'(?:'
    r'(?P<qval>"(?:[^"\\]|\\.)*")|'  # Quoted string
    r'(?P<sval>[^,{}]+)'             # Simple value (up to comma or brace)
    r')'
)

def parse_payload_to_dict(payload_str: str) -> Dict[str, Any]:
    """
    Parses " { key = val, key2 = val2 }, { ctx = val }" string into a flat dict.
    Returns strings as values mostly, does basic int conversion.
    """
    data = {}
    
    # Simple approach: find all matches of KV_RE
    for m in KV_RE.finditer(payload_str):
        k = m.group("key")
        if m.group("qval"):
            # It's a quoted string, strip quotes
            v = m.group("qval")[1:-1]
        else:
            # It's a simple value, strip whitespace
            v = m.group("sval").strip()
            
        # Try converting to number if possible
        if v.isdigit():
            v = int(v)
        elif v.startswith("0x"):
            try:
                v = int(v, 16)
            except:
                pass
        
        data[k] = v
        
    return data

def load_vocab(vocab_path: str) -> Dict[str, int]:
    print(f"[INFO] Loading vocab from {vocab_path}...")
    with open(vocab_path, 'r') as f:
        return json.load(f)

def write_parquet_stream(
    input_path: Path,
    output_path: Path,
    vocab: Dict[str, int],
    batch_size: int = 200_000,
):
    # --- Schema Definition ---
    # Core
    fields = [
        ("line_idx", pa.int64()),
        ("ts_str", pa.string()),
        ("delta_s", pa.float64()),
        ("t_rel_ns", pa.int64()),
        ("event_name", pa.string()),
        ("event_id", pa.int32()),  # NEW: event_id
        ("cpu_id", pa.int32()),
    ]
    
    # First-Class Columns (Sparse / Nullable)
    first_class_cols = [
        ("tid", pa.int64()),
        ("pid", pa.int64()),
        ("comm", pa.string()),
        ("ret", pa.int64()),
        ("fd", pa.int64()),
        ("filename", pa.string()),
        ("flags", pa.string()), 
        ("state", pa.int64()),
    ]
    fields.extend(first_class_cols)
    
    # Catch-All
    fields.append(("extra_params", pa.string())) # JSON string
    
    schema = pa.schema(fields)

    print(f"[INFO] Writing to {output_path} with schema: {schema.names}")

    writer = pq.ParquetWriter(
        output_path.as_posix(),
        schema=schema,
        compression="zstd",
        use_dictionary=True,
    )

    # Buffers
    cols = {name: [] for name in schema.names}

    t_rel_ns = 0
    line_idx = 0
    parsed = 0
    skipped = 0
    
    fc_keys = set(n for n, _ in first_class_cols)

    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw_line = raw.rstrip("\n")

            m = LINE_RE.match(raw_line)
            if not m:
                skipped += 1
                continue

            ts_str = m.group("ts")
            
            # 1. Delta handling
            raw_delta = m.group("delta")
            if "?" in raw_delta:
                delta_s = 0.0
            else:
                try:
                    delta_s = float(raw_delta)
                except ValueError:
                    delta_s = 0.0

            event_name = m.group("event").strip()
            rest = m.group("rest") 

            # 2. Event ID
            event_id = vocab.get(event_name, -1)

            # 3. Payload
            all_params = parse_payload_to_dict(rest)
            
            # 4. CPU
            cpu_val = all_params.get("cpu_id", -1)
            try:
                cpu_id = int(cpu_val)
            except:
                cpu_id = -1
                
            # 5. Monotonic time
            t_rel_ns += int(round(delta_s * 1_000_000_000))

            # 6. Fill Core Columns
            cols["line_idx"].append(line_idx)
            cols["ts_str"].append(ts_str)
            cols["delta_s"].append(delta_s)
            cols["t_rel_ns"].append(t_rel_ns)
            cols["event_name"].append(event_name)
            cols["event_id"].append(event_id)
            cols["cpu_id"].append(cpu_id)
            
            # 7. Fill First-Class Columns
            extras = {}
            for fc_key, fc_type in first_class_cols:
                val = all_params.get(fc_key)
                if val is not None:
                    if fc_type == pa.int64():
                        if not isinstance(val, int):
                             try:
                                 val = int(val)
                             except:
                                 val = None 
                    elif fc_type == pa.string():
                         if not isinstance(val, str):
                             val = str(val)
                cols[fc_key].append(val)
                
            for k, v in all_params.items():
                if k not in fc_keys and k != "cpu_id":
                    extras[k] = v
                    
            cols["extra_params"].append(json.dumps(extras) if extras else None)

            line_idx += 1
            parsed += 1

            if parsed % batch_size == 0:
                table = pa.Table.from_pydict(cols, schema=schema)
                writer.write_table(table)
                cols = {name: [] for name in schema.names}
                print(f"   ... processed {parsed} lines", end='\r')

    if cols["line_idx"]:
        table = pa.Table.from_pydict(cols, schema=schema)
        writer.write_table(table)

    writer.close()

    print(f"\n[OK] {input_path.name} -> {output_path}")
    print(f"     parsed={parsed}, skipped={skipped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input .txt trace file")
    ap.add_argument("--output", required=True, help="Output .parquet file path")
    ap.add_argument("--vocab", required=True, help="Path to vocab.json")
    ap.add_argument("--batch-size", type=int, default=200_000)
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    vocab = load_vocab(args.vocab)

    write_parquet_stream(
        input_path=input_path,
        output_path=output_path,
        vocab=vocab,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()


# python synthetic_data_processing/txt_to_enriched_parquet.py --input dataset/txt_traces/compress-gzip-all-events-run0.txt --output dataset/parquet_test/test2.parquet --vocab dataset/metadata_all_events/vocab.json