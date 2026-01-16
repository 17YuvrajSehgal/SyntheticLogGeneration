# ============================================================
# STEP 1: Prepare real train/test data
# ============================================================
python experiments_downstream/prepare_data.py --real-glob "/scratch/yuvraj17/windowed_npz_256/scimark2/train/*.npz" --benchmark scimark2 --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data"

# ============================================================
# STEP 2: Generate 10k synthetic data
# ============================================================
python sample_diffusion.py --ckpt "$SCRATCH/SyntheticLogGeneration/experiments_results/scimark2/exp_context_256/exp_context_256_scimark2/exp_context_256_scimark2/ckpt_epoch_19.pt" --out "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/synthetic_raw_256_10k.npz" --num-samples 10000 --seq-len 256 --batch-size 64 --use-ddim --ddim-steps 50

# ============================================================
# STEP 3: Repair synthetic data
# ============================================================
python synthetic_log_gen/repair.py --trace "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/synthetic_raw_256_10k.npz" --constraints dataset/constraints_universal.json --output "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/synthetic_repaired_256_10k.npz"

# ============================================================
# STEP 4: Create combined (50% real + 50% synthetic) dataset
# ============================================================
python experiments_downstream/combine_datasets.py --real-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/real_train.npz" --synthetic-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/synthetic_repaired_256_10k.npz" --output "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/combined_real_synthetic_256_50_50.npz" --ratio 0.5

# ============================================================
# STEP 5: Train on real data (baseline)
# ============================================================
python experiments_downstream/models/train_predictor.py --train-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/real_train.npz" --test-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/real_test.npz" --run-name "real_baseline_scimark2_256" --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/results" --seq-len 128 --batch-size 64 --epochs 20

# ============================================================
# STEP 6: Train only on synthetic data
# ============================================================
python experiments_downstream/models/train_predictor.py --train-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/synthetic_repaired_256_10k.npz" --test-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/real_test.npz" --run-name "synthetic_only_scimark2_256" --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/results" --seq-len 128 --batch-size 64 --epochs 20 --patience 5

# ============================================================
# STEP 7: Train on combined (hybrid) data
# ============================================================
python experiments_downstream/models/train_predictor.py --train-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/combined_real_synthetic_256_50_50.npz" --test-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/data/real_test.npz" --run-name "combined_50_50_scimark2_256" --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/scimark2/256/results" --seq-len 128 --batch-size 64 --epochs 20 --patience 5