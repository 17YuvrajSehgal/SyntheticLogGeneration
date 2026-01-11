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
    # Organize all 3 resolutions
    roots = [
        "dataset/window_shards", # The symlinked or main one?
        # User mentioned these scratch paths, but likely mapped to dataset/window_shards locally or we run this on cluster.
        # Let's assume user puts path as arg or we verify.
    ]
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="List of root directory to organize (e.g. dataset/window_shards)")
    args = parser.parse_args()
    
    for d in args.dirs:
        print(f"Organizing {d}...")
        organize_dataset(d)
        print("Done.")
