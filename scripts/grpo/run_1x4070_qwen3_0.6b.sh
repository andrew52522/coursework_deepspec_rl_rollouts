#!/usr/bin/env bash
set -euo pipefail

# This script is intended to run INSIDE the coursework CUDA container,
# with verl mounted at /workspace/verl.

MODEL_PATH="${MODEL_PATH:-/model}"
TRAIN_FILE="${TRAIN_FILE:-/data/gsm8k/train.parquet}"
TEST_FILE="${TEST_FILE:-/data/gsm8k/test.parquet}"

echo "=== NATIVE GRPO 1x4070 CONFIG ==="
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "TEST_FILE=${TEST_FILE}"
echo "GRPO_RUN_CONFIG train_batch_size=4 rollout_n=16 ppo_mini_batch_size=2 ppo_epochs=1 max_prompt_length=256 max_response_length=512 max_model_len=768 ppo_max_token_len_per_gpu=1024 dataloader_num_workers=0"
echo "VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER-<UNSET>}"
echo "VERL_FORCE_SHM_WEIGHT_TRANSFER=${VERL_FORCE_SHM_WEIGHT_TRANSFER-<UNSET>}"

uv run \
  --frozen \
  --all-packages \
  --extra vllm \
  --extra fsdp \
  python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.train_batch_size=4 \
  data.dataloader_num_workers=0 \
  data.max_prompt_length=256 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_model_len=768 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=1024 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1024 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.logger='["console"]' \
  trainer.project_name=coursework_native_grpo \
  trainer.experiment_name=qwen3_0_6b_1x4070_native_smoke \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.resume_mode=disable \
  'ray_kwargs.ray_init.runtime_env.py_executable=uv -v run --frozen --all-packages --extra vllm --extra fsdp' \
  "$@"
