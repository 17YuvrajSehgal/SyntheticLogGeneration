# ============================================================
# STEP 1: Prepare real train/test data
# ============================================================
python experiments_downstream/prepare_data.py --real-glob "/scratch/yuvraj17/window_shards/windowed_npz_256/scimark2/train/*.npz" --benchmark scimark2 --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data"

# ============================================================
# STEP 2: Generate 10k synthetic data
# ============================================================
python sample_diffusion.py --ckpt "$SCRATCH/SyntheticLogGeneration/logs_tensorboard/improved_baseline_scimark2_256/ckpt_epoch_19.pt" --out "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/synthetic_raw_256_10k.npz" --num-samples 10000 --seq-len 256 --batch-size 64 --d-model 256 --nhead 4 --num-layers 4 --use-ddim --ddim-steps 50

# ============================================================
# STEP 3: Create combined (50% real + 50% synthetic no repaired) dataset
# ============================================================
python experiments_downstream/combine_datasets.py --real-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/real_train.npz" --synthetic-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/synthetic_raw_256_10k.npz" --output "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/combined_real_synthetic_norepair_256_50_50.npz" --ratio 0.5

# ============================================================
# STEP 4: Repair synthetic data
# ============================================================
python synthetic_log_gen/repair.py --trace "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/synthetic_raw_256_10k.npz" --constraints dataset/constraints_universal.json --output "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/synthetic_repaired_256_10k.npz"

# ============================================================
# STEP 5: Create combined (50% real + 50% synthetic) dataset
# ============================================================
python experiments_downstream/combine_datasets.py --real-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/real_train.npz" --synthetic-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/synthetic_repaired_256_10k.npz" --output "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/combined_real_synthetic_repaired_256_50_50.npz" --ratio 0.5

# ============================================================
# STEP 6: Train on real data (baseline)
# ============================================================
python experiments_downstream/models/train_predictor.py --train-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/real_train.npz" --test-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/real_test.npz" --run-name "real_baseline_scimark2_256" --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/results" --seq-len 128 --batch-size 64 --epochs 20

# ============================================================
# STEP 7: Train only on synthetic data
# ============================================================
python experiments_downstream/models/train_predictor.py --train-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/synthetic_raw_256_10k.npz" --test-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/real_test.npz" --run-name "synthetic_data_only_scimark2_256" --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/results" --seq-len 128 --batch-size 64 --epochs 20

# ============================================================
# STEP 8: Train on combined (hybrid) data repaired
# ============================================================
python experiments_downstream/models/train_predictor.py --train-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/combined_real_synthetic_repaired_256_50_50.npz" --test-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/real_test.npz" --run-name "combined_50_50_scimark2_256" --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/results" --seq-len 128 --batch-size 64 --epochs 20


# ============================================================
# STEP 9: Train on combined (hybrid) data no repair
# ============================================================
python experiments_downstream/models/train_predictor.py --train-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/combined_real_synthetic_norepair_256_50_50.npz" --test-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/real_test.npz" --run-name "combined_50_50_norepair_scimark2_256" --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/results" --seq-len 128 --batch-size 64 --epochs 20