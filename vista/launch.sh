#!/bin/bash
# Train VisTA-BAGEL on one node with 8 GPUs: 16,384 steps from the official UniREdit-BAGEL weights.
#
#   MODEL_PATH    directory of ByteDance-Seed/BAGEL-7B-MoT (configs, tokenizer, VAE, ViT)
#   INIT_WEIGHTS  model.safetensors of the official UniREdit-BAGEL checkpoint (maplebb/UniREdit-Bagel-bf16)
#   VISTA_DATA    training data folder (default: ../data/train_examples; ../data/train after downloading, see the README)
#   OUT           results folder
set -euo pipefail
cd "$(dirname "$0")"
: "${MODEL_PATH:?}" "${INIT_WEIGHTS:?}" "${OUT:?}"
export PYTHONPATH="$PWD/../bagel:$PWD:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 WANDB_MODE=offline OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS=${STEPS:-16384}
mkdir -p "$OUT/warm_start"
ln -sfn "$(readlink -f "$INIT_WEIGHTS")" "$OUT/warm_start/model.safetensors"
python -m torch.distributed.run --standalone --nproc_per_node=8 run_training.py \
  --hint mixed_hint --milestones "$STEPS" \
  --dataset_config_file configs/data.yaml --model_path "$MODEL_PATH" \
  --resume_from "$OUT/warm_start" --finetune_from_hf True --resume_model_only True \
  --finetune_from_ema False --auto_resume False --copy_init_moe False \
  --layer_module Qwen2MoTDecoderLayer --max_latent_size 64 --use_flex True \
  --visual_gen True --visual_und True --freeze_vae True --freeze_vit True --freeze_llm False --freeze_und False \
  --text_cond_dropout_prob 0 --vae_cond_dropout_prob 0 --vit_cond_dropout_prob 0 \
  --lr 2e-6 --lr_scheduler constant --warmup_steps 8 --total_steps "$STEPS" \
  --ce_weight 1 --mse_weight 1 --ce_loss_reweighting False --ema 0.9999 \
  --expected_num_tokens 1 --max_num_tokens 12288 --max_num_tokens_per_sample 8192 \
  --prefer_buffer_before 1 --max_buffer_size 16 --num_workers 1 --prefetch_factor 1 \
  --num_shard 8 --num_replicate 1 --sharding_strategy FULL_SHARD \
  --data_seed 20260918 --global_seed 20260918 --log_every 1 --save_every "$STEPS" \
  --eval_every "$STEPS" --checkpoint_keep 1 --save_final_checkpoint True \
  --checkpoint_dir "$OUT/checkpoints" --results_dir "$OUT" \
  --wandb_project vista --wandb_name vista-bagel --wandb_offline True 2>&1 | tee "$OUT/console.log"
