import os
import shutil
import glob
from tqdm import tqdm

def organize_dataset(root_dir):
    """
    Moves files from:
       root/train/bench-runX.npz
    To:
       root/bench/train/bench-runX.npz
    """
    splits = ["train", "val", "test"]
    
    for split in splits:
        source_dir = os.path.join(root_dir, split)
        if not os.path.exists(source_dir):
            print(f"Skipping {source_dir} (not found)")
            continue
            
        print(f"Processing {split}...")
        files = glob.glob(os.path.join(source_dir, "*.npz"))
        
        for fpath in tqdm(files):
            filename = os.path.basename(fpath)
            
            # Heuristic: Benchmark name is the prefix before "-all-events"
            # e.g. "ffmpeg-all-events-run20..." -> "ffmpeg"
            # e.g. "compress-gzip-all-events..." -> "compress-gzip"
            
            if "-all-events" in filename:
                bench_name = filename.split("-all-events")[0]
            else:
                # Fallback: take first part before hyphen?
                # or just use "unknown"
                bench_name = filename.split("-")[0]
            
            # Create target dir: root/bench/split/
            target_dir = os.path.join(root_dir, bench_name, split)
            os.makedirs(target_dir, exist_ok=True)
            
            # Move
            shutil.move(fpath, os.path.join(target_dir, filename))
            
    # Cleanup empty split dirs
    for split in splits:
        try:
            os.rmdir(os.path.join(root_dir, split))
        except:
            pass # Not empty

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="*", help="List of root directory to organize")
    args = parser.parse_args()

    # Default to SCRATCH directories if no args provided
    if not args.dirs:
        scratch = os.environ.get("SCRATCH")
        if scratch:
            print(f"Detected SCRATCH: {scratch}")
            targets = [
                "windowed_npz_256", 
                "windowed_npz_1024", 
                "windowed_npz_4096"
            ]
            args.dirs = [os.path.join(scratch, t) for t in targets]
        else:
            print("No arguments provided and SCRATCH not set.")
            print("Usage: python organize_shards.py [dir1] [dir2] ...")
            exit(1)
    
    for d in args.dirs:
        print(f"Organizing {d}...")
        if os.path.exists(d):
            organize_dataset(d)
        else:
            print(f"Directory not found: {d}")
        print("Done.")
