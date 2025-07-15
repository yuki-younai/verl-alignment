set -x

#export VLLM_ATTENTION_BACKEND=XFORMERS
#export CUDA_LAUNCH_BLOCKING=1
#export NCCL_CROSS_NIC=2


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

current_time=$(date +"%Y%m%d_%H%M%S")
gsm8k_train_path=/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/datasets/gsm8k-verl/train.parquet
gsm8k_test_path=/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/datasets/gsm8k-verl/test.parquet
export RUN_NAME=verl-grpo-multi-node-${current_time}
export PROJECT_NAME=verl-grpo-multi-node
export OUTPUT_PATH=/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/exp_output/$RUN_NAME
export HDFS_LOG_PATH=$OUTPUT_PATH/log
export HDFS_CHECKPOINT_PATH=$OUTPUT_PATH/model_output
export TENSORBOARD_DIR=$OUTPUT_PATH/tensorboard_log

SCRIPT_NAME=$(basename "$0")
DESTINATION_PATH="$OUTPUT_PATH/$SCRIPT_NAME"
cp "$0" "$DESTINATION_PATH"


train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

if [ "$RANK" -eq 0 ]; then
    python -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=512 \
        data.max_prompt_length=1024 \
        data.max_response_length=2048 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2.5-Math-7B \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio=0.2 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        custom_reward_function.path=examples/reward_function/simplerl_math.py \
        custom_reward_function.name=compute_score \
        trainer.critic_warmup=0 \
        trainer.logger=['console','tensorboard'] \
        trainer.project_name='verl_grpo_example_gsm8k_math' \
        trainer.experiment_name='qwen2_7b_function_rm' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=$WORLD_SIZE \
        trainer.save_freq=300 \
        trainer.test_freq=10 \
        trainer.default_local_dir=$HDFS_CHECKPOINT_PATH \
        trainer.total_epochs=15 
fi
