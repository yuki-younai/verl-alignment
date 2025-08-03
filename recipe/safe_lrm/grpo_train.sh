set -x

export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export WORLD_SIZE=${WORLD_SIZE}
export RANK=${RANK}
export MY_PORT=8464

echo 'Acquire::http::Proxy "http://10.20.112.35:3143";' > /etc/apt/apt.conf.d/proxy
sudo apt install iputils-ping -y
sudo apt-get install dnsutils -y
sudo apt-get install gawk -y

ping_result=$(ping -c 1 $MASTER_ADDR)  # 捕获输出到变量
echo ping result: "$ping_result"  # 打印变量内容
#ping_result="PING jo-dbehyvqzqt4so34i-worker-0 (172.27.34.143) 56(84) bytes of data."
export HEAD_NODE_ADDRESS=$(echo "$ping_result" | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | head -n 1)
echo Master Ipv4 Address: $HEAD_NODE_ADDRESS


if [ "$RANK" -eq 0 ]; then
    echo "本机 $MASTER_ADDR 是Head节点, 第 $RANK 个,启动Head节点..."
    ray start --head --port=$MY_PORT 
else
    echo "本机 $LOCAL_IP 是Worker节点, 第 $RANK 个,连接到Head节点 $MASTER_ADDR..."
    ray start --block --address=$HEAD_NODE_ADDRESS:$MY_PORT 
fi
#Train
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/word2number-1.1.zip
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/latex2sympy2_extended-1.10.1-py3-none-any.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/math_verify-0.7.0-py3-none-any.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/antlr4-python3-runtime-4.7.2.tar.gz
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/latex2sympy2-1.9.1-py3-none-any.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/antlr4-python3-runtime-4.9.3.tar.gz


current_time=$(date +"%Y%m%d_%H%M%S")
#train_path=/mnt/public/gpfs-jd/code/guoweiyang/LRM/datasets/math-verl/train.parquet
#test_path=/mnt/public/gpfs-jd/code/guoweiyang/LRM/datasets/math-verl/test.parquet
train_path=/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/datasets/dapo17k-verl/train.parquet
test_path=/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/datasets/eval_datasets/aime2024-32/aime-2024.parquet
train_files="['$train_path']"
test_files="['$test_path']"

export RUN_NAME=verl-grpo-train-${current_time}
export PROJECT_NAME=verl-grpo-train
export OUTPUT_PATH=/mnt/public/gpfs-jd/code/guoweiyang/LRM/exp_out/$RUN_NAME
export HDFS_LOG_PATH=$OUTPUT_PATH/log
export HDFS_CHECKPOINT_PATH=$OUTPUT_PATH/model_output
export TENSORBOARD_DIR=$OUTPUT_PATH/tensorboard_log

if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir -p "$OUTPUT_PATH"
    echo "目录 $OUTPUT_PATH 已创建"
else
    echo "目录 $OUTPUT_PATH 已存在"
fi

SCRIPT_NAME=$(basename "$0")
DESTINATION_PATH="$OUTPUT_PATH/$SCRIPT_NAME"
cp "$0" "$DESTINATION_PATH"

loss_agg_mode="token-mean"

micro_batch_size=8
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 6))
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * $micro_batch_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * $micro_batch_size))
Model_PATH_or_NAME=/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B-Instruct


if [ "$RANK" -eq 0 ]; then
    python -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=512 \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=$Model_PATH_or_NAME \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
        actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.2 \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.rollout.n=8 \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        custom_reward_function.path=examples/reward_function/math_reward.py \
        custom_reward_function.name=compute_score_math \
        actor_rollout_ref.actor.checkpoint.save_contents=['model'] \
        trainer.logger=['console','tensorboard'] \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$RUN_NAME \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=$WORLD_SIZE \
        trainer.save_freq=100 \
        trainer.test_freq=10 \
        trainer.default_local_dir=$HDFS_CHECKPOINT_PATH \
        trainer.total_epochs=20
fi



