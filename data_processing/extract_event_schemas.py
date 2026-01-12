# python synthetic_data_processing/extract_event_schemas.py --dir <path_to_txt_traces_root> --out dataset/event_schemas.json
import argparse
import json
import re
import sys
import os
import glob
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Regex to capture the event name and the payload part
# Example line: [08:46:31.491351270] (+0.000000632) fluti power_cpu_idle: { cpu_id = 2 }, { state = 4294967295, cpu_id = 2 }

# Pre-compiled regex for performance
# Matches " identifier =" patterns.
KEY_PATTERN = re.compile(r'\b([a-zA-Z0-9_]+)\s*=')

def parse_line(line: str) -> Tuple[str, List[str]]:
    """
    Returns (event_name, list_of_keys).
    """
    line = line.strip()
    if not line:
        return None, []
    
    # 1. Find the split point between metadata and payload
    parts = line.split(": {", 1) # simple heuristic split
    if len(parts) < 2:
        return None, []
        
    prefix = parts[0]
    # Event name is the last token in the prefix
    event_name = prefix.split()[-1]
    
    # Payload is the rest of the line (re-adding the open brace we consumed)
    payload_str = "{ " + parts[1]
    
    # 2. Extract keys
    keys = KEY_PATTERN.findall(payload_str)
    
    return event_name, keys

def process_file(filepath: str, schema_stats: Dict):
    print(f"[INFO] Reading {filepath}...")
    try:
        count = 0
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if not line.strip(): 
                    continue
                
                # Skip comments or headers if any
                if line.startswith("#") or line.startswith("==="):
                    continue
                    
                e_name, keys = parse_line(line)
                if e_name:
                    # canonicalize keys (sort them) to form a signature
                    signature = tuple(sorted(list(set(keys))))
                    schema_stats[e_name][signature] += 1
                
                count += 1
                if count % 500000 == 0:
                    print(f"   ... {count} lines", end='\r')
                    
    except Exception as e:
        print(f"[WARN] Failed to read {filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract event parameter schemas from LTTng text traces recursively.")
    parser.add_argument("--dir", required=True, help="Root directory containing subfolders with .txt traces")
    parser.add_argument("--out", required=True, help="Path to output JSON registry")
    args = parser.parse_args()
    
    # Structure: event_name -> dict( frozen_set_of_keys -> count )
    schema_stats = defaultdict(lambda: defaultdict(int))
    
    # Find all .txt files
    print(f"[INFO] Scanning directory: {args.dir}")
    trace_files = []
    
    for root, dirs, files in os.walk(args.dir):
        # Exclude parquet directories
        if "parquet" in root.lower():
            continue
            
        for file in files:
            if file.endswith(".txt"):
                trace_files.append(os.path.join(root, file))
    
    if not trace_files:
        print(f"[ERROR] No .txt files found in {args.dir}")
        sys.exit(1)
        
    print(f"[INFO] Found {len(trace_files)} text trace files.")
    
    # Process all files
    for idx, trace_path in enumerate(trace_files):
        print(f"[{idx+1}/{len(trace_files)}] Processing file...")
        process_file(trace_path, schema_stats)
        
    print(f"\n[INFO] Parsing complete. Discovered {len(schema_stats)} unique event types.")
    
    # Format output
    output_registry = {}
    
    for event, variants in schema_stats.items():
        variant_list = []
        # Sort variants by count descending
        sorted_vars = sorted(variants.items(), key=lambda x: x[1], reverse=True)
        
        for idx, (sig, count) in enumerate(sorted_vars):
            variant_list.append({
                "signature_id": idx,
                "fields": list(sig),
                "count": count
            })
            
        output_registry[event] = variant_list
        
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(output_registry, f, indent=2)
        
    print(f"[INFO] Registry saved to {args.out}")

if __name__ == "__main__":
    main()
