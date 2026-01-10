#python synthetic_data_processing/extract_event_schemas.py --trace dataset/txt_traces/compress-gzip-all-events-run0.txt --out dataset/metadata_all_events/event_schemas.json
import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Regex to capture the event name and the payload part
# Example line: [08:46:31.491351270] (+0.000000632) fluti power_cpu_idle: { cpu_id = 2 }, { state = 4294967295, cpu_id = 2 }
# We want "power_cpu_idle" and the content inside all {...} blocks.
# Note: Some lines have multiple {} blocks or nested ones.
# Simplified approach:
# 1. Split by ": " to separate metadata from content.
# 2. The part before ": " contains the event name (last word).
# 3. The part after ": " contains the payload.

def parse_line(line: str) -> Tuple[str, List[str]]:
    """
    Returns (event_name, list_of_keys).
    """
    line = line.strip()
    if not line:
        return None, []
    
    # 1. Find the split point between metadata and payload
    # Look for the first colon that follows the timestamp/hostname pattern
    # Actually, LTTng text output is usually:
    # [TIMESTAMP] (DELTA) hostname event_name: { payload }
    
    parts = line.split(": {", 1) # simple heuristic split
    if len(parts) < 2:
        return None, []
        
    prefix = parts[0]
    # Event name is the last token in the prefix
    event_name = prefix.split()[-1]
    
    # Payload is the rest of the line (re-adding the open brace we consumed)
    payload_str = "{ " + parts[1]
    
    # 2. Extract keys
    # Keys are usually "key ="
    # We can use regex to find all " identifier =" patterns.
    # This might find false positives inside strings, but for LTTng traces it is usually safe enough 
    # as keys are not quoted, and string values are quoted.
    
    # Regex: Matches word boundary, identifier, space*, equals
    # We ignore what's on the right side of equals.
    keys = re.findall(r'\b([a-zA-Z0-9_]+)\s*=', payload_str)
    
    return event_name, keys

def main():
    parser = argparse.ArgumentParser(description="Extract event parameter schemas from LTTng text traces.")
    parser.add_argument("--trace", required=True, help="Path to input .txt trace file")
    parser.add_argument("--out", required=True, help="Path to output JSON registry")
    args = parser.parse_args()
    
    # Structure: event_name -> dict( frozen_set_of_keys -> count )
    schema_stats = defaultdict(lambda: defaultdict(int))
    
    print(f"[INFO] Reading {args.trace}...")
    
    count = 0
    with open(args.trace, 'r', encoding='utf-8', errors='replace') as f:
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
            if count % 100000 == 0:
                print(f"... processed {count} lines", end='\r')
                
    print(f"\n[INFO] simple parsing complete. Discovered {len(schema_stats)} event types.")
    
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
        
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(output_registry, f, indent=2)
        
    print(f"[INFO] Registry saved to {args.out}")

if __name__ == "__main__":
    main()
