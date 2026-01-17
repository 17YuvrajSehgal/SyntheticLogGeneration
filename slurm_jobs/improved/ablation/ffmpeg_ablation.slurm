#Phase 1: Training Diffusion Models ✅ (DONE)
#Phase 2: Generate Synthetic Data from Each Model
python sample_diffusion.py --ckpt "$SCRATCH/SyntheticLogGeneration/experiments_results/ffmpeg-ablation/exp_ablation_base_ffmpeg/ckpt_epoch_19.pt" --out "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ffmpeg-ablation/ffmpeg_base_10k.npz" --num-samples 10000 --seq-len 1024 --d-model 512 --nhead 8 --num-layers 8 --use-ddim --ddim-steps 50

#Phase 3: Train Downstream Predictors on Real Data (NEW APPROACH)
python experiments_downstream/prepare_data.py --real-glob "$SCRATCH/window_shards/windowed_npz_1024/ffmpeg/train/*.npz" --benchmark ffmpeg --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation/ffmpeg/data"

#Phase 4: Train Downstream Predictors
python experiments_downstream/models/train_predictor.py --train-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation/ffmpeg/data/real_train.npz" --test-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation/ffmpeg/data/real_test.npz" --channels event --run-name ablation_ffmpeg_event_only --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation/ffmpeg/results" --seq-len 128 --batch-size 64 --epochs 20

python experiments_downstream/models/train_predictor.py --train-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation/ffmpeg/data/real_train.npz" --test-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation/ffmpeg/data/real_test.npz" --channels event dt --run-name ablation_ffmpeg_base --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation/ffmpeg/results" --seq-len 128 --batch-size 64 --epochs 20

python experiments_downstream/models/train_predictor.py --train-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation/ffmpeg/data/real_train.npz" --test-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation/ffmpeg/data/real_test.npz" --channels event dt cpu tid --run-name ablation_ffmpeg_system --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation/ffmpeg/results" --seq-len 128 --batch-size 64 --epochs 20

python experiments_downstream/models/train_predictor.py --train-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation/ffmpeg/data/real_train.npz" --test-data "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation/ffmpeg/data/real_test.npz" --channels event dt cpu tid comm ret --run-name ablation_ffmpeg_full --output-dir "$SCRATCH/SyntheticLogGeneration/experiments_downstream_results/ablation/ffmpeg/results" --seq-len 128 --batch-size 64 --epochs 20