set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
#export VLLM_ATTENTION_BACKEND=XFORMERS

export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export WORLD_SIZE=${WORLD_SIZE}
export RANK=${RANK}

current_time=$(date +"%Y%m%d_%H%M%S")
math_train_path=/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/datasets/math-verl/train.parquet
math_test_path=/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/datasets/math-verl/test.parquet
gsm8k_train_path=/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/datasets/gsm8k-verl/train.parquet
gsm8k_test_path=/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/datasets/gsm8k-verl/test.parquet
export RUN_NAME=verl-wreinforce-plus-plus-math_${current_time}
export PROJECT_NAME=verl-wreinforce-plus-plus-math
export OUTPUT_PATH=/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/exp_output
export HDFS_LOG_PATH=$OUTPUT_PATH/log/$RUN_NAME
export HDFS_CHECKPOINT_PATH=$OUTPUT_PATH/model_output/$RUN_NAME
export TENSORBOARD_DIR=$OUTPUT_PATH/tensorboard_log/$RUN_NAME

LOG_FILE_PATH="$HDFS_LOG_PATH/$RUN_NAME.log"
touch "$LOG_FILE_PATH"

train_files="['$math_train_path']"
test_files="['$gsm8k_test_path']"
Model_PATH_or_NAME=/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2.5-Math-7B

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 3))
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 5))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=psr_nsr \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$Model_PATH_or_NAME \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_ppo_max_token_len \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$infer_ppo_max_token_len \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$infer_ppo_max_token_len \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=recipe/reward_function/simplerl_math.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH \
    trainer.total_epochs=20  2>&1 | tee -a $LOG_FILE_PATH

