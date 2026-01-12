import argparse
import os
import subprocess
import json
import sys

def run_command(cmd):
    print(f"[CMD] {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"[ERR] Command failed with code {ret}")
        sys.exit(ret)

def main():
    parser = argparse.ArgumentParser(description="Synthetic Log Repair Pipeline")
    parser.add_argument("--trace", required=True, help="Original synthetic traces (.npz)")
    parser.add_argument("--constraints", required=True, help="Constraints JSON")
    parser.add_argument("--output-dir", default="repaired_traces", help="Output directory for repaired traces")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paths
    report_pre = os.path.join(args.output_dir, "validity_pre_repair.json")
    repaired_trace = os.path.join(args.output_dir, "repaired_traces.npz")
    report_post = os.path.join(args.output_dir, "validity_post_repair.json")
    
    # 1. Pre-Repair Validation
    print("="*60)
    print("STEP 1: Validating Original Traces")
    print("="*60)
    run_command(f"python synthetic_log_gen/validate.py --trace {args.trace} --constraints {args.constraints} --output {report_pre}")
    
    # 2. Repair
    print("\n" + "="*60)
    print("STEP 2: Running Post-Hoc Repair")
    print("="*60)
    run_command(f"python synthetic_log_gen/repair.py --trace {args.trace} --constraints {args.constraints} --output {repaired_trace}")
    
    # 3. Post-Repair Validation
    print("\n" + "="*60)
    print("STEP 3: Validating Repaired Traces")
    print("="*60)
    run_command(f"python synthetic_log_gen/validate.py --trace {repaired_trace} --constraints {args.constraints} --output {report_post}")
    
    # 4. Comparative Report
    print("\n" + "="*60)
    print("STEP 4: IMPROVEMENT REPORT")
    print("="*60)
    
    with open(report_pre) as f: pre = json.load(f)
    with open(report_post) as f: post = json.load(f)
    
    score_pre = pre["validity_score"]
    score_post = post["validity_score"]
    
    print(f"{'Metric':<25} | {'Pre-Repair':<10} | {'Post-Repair':<10} | {'Improvement':<10}")
    print("-" * 65)
    
    metrics = ["transitions", "timing", "cpu_global", "cpu_local"]
    for m in metrics:
        v1 = score_pre.get(m, 0.0)
        v2 = score_post.get(m, 0.0)
        imp = v2 - v1
        print(f"{m:<25} | {v1:>9.2f}% | {v2:>9.2f}% | {imp:>+9.2f}%")
        
    print("-" * 65)

if __name__ == "__main__":
    main()

# python run_repair_pipeline.py --trace generated_traces/context_4096_gen.npz --constraints dataset/constraints_universal.json
