set -x

export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export WORLD_SIZE=${WORLD_SIZE}
export RANK=${RANK}
export MY_PORT=8469

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
#Mol2caption or Caption2mol
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/whl_files/selfies-2.2.0-py3-none-any.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/whl_files/rdkit-2025.3.3-cp310-cp310-manylinux_2_28_x86_64.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/whl_files/nltk-3.9.1-py3-none-any.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/whl_files/rapidfuzz-3.13.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/whl_files/levenshtein-0.27.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/whl_files/rouge_score-0.1.2.tar.gz


current_time=$(date +"%Y%m%d_%H%M%S")
export RUN_NAME=verl-qwen2moe-science_${current_time}
export PROJECT_NAME=verl-qwen2moe-science
export OUTPUT_PATH=/mnt/public/gpfs-jd/model/RL_Model/$RUN_NAME
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


math_path="/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/math/dapo10k"
physics_path="/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/physics-full-17k"
stem_path="/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/science/webinstruct-split-13k"
code_path="/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/code/leetcode2k"
chemistry_path="/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/chemistry-split-8k"
caption2mol_path="/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/caption2mol-split-6k"
mol2caption_path="/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/mol2caption-split-6k"

train_math_path="${math_path}/train.parquet"
train_physics_path="${physics_path}/train.parquet"
train_stem_path="${stem_path}/train.parquet"
train_code_path="${code_path}/train.parquet"
train_chemistry_path="${chemistry_path}/train.parquet"
train_caption2mol_path="${caption2mol_path}/train.parquet"
train_mol2caption_path="${mol2caption_path}/train.parquet"

test_math_path="${math_path}/test.parquet"
test_physics_path="${physics_path}/test.parquet"
test_stem_path="${stem_path}/test.parquet"
test_chemistry_path="${chemistry_path}/test.parquet"
test_caption2mol_path="${caption2mol_path}/test.parquet"
test_mol2caption_path="${mol2caption_path}/test.parquet"

train_files="['$train_math_path','$train_physics_path', '$train_stem_path', '$train_code_path', '$train_chemistry_path', '$train_caption2mol_path', '$train_mol2caption_path']"
test_files="['$test_math_path', '$test_physics_path', '$test_chemistry_path', '$test_caption2mol_path', '$test_mol2caption_path']"

### Separated Clip Epsilons (-> Clip-Higher)
clip_ratio_low=0.2
clip_ratio_high=0.28

### Dynamic Sampling (with Group Filtering)
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

### Flexible Loss Aggregation Mode (-> Token-level Loss)
loss_agg_mode="token-mean"

### Overlong Reward Shaping
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=1.0

fsdp_size=-1
# Algorithm
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 6))
micro_batch_size=4
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * $micro_batch_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * $micro_batch_size))


HF_E8_T50=/mnt/public/gpfs-jd/model/BASE_LLM/Experiments/mcore2hf/Qwen2.5-7B-1S-NumExpert8-TP1-PP2-EP4-SI18944-EI18944-TOP1-NSHARD0-NCSHARD1-SEI_copy-REI_copy-RI_normal-Data01_1-02_1-03_1-04_1-05_1-06_1-08_1-09_1-10_1-11_1-12_1-13_1-17_4-Train50B-WMUP0.5B-LR1e-5-MIN_LR1e-7_untie
HF_E64_T50=/mnt/public/gpfs-jd/model/BASE_LLM//mcore2hf/Qwen2.5-7B-1S-NumExpert64-TP1-PP2-EP8-SI18944-EI2368-TOP8-NSHARD8-NCSHARD1-SEI_copy-REI_nv_shard-RI_nv_shard-Data01_1-02_1-03_1-04_1-05_1-06_1-08_1-09_1-10_1-11_1-12_1-13_1-17_4-19_1-Train50B-WMUP0.5B-LR1e-5-MIN_LR1e-7_untie
HF_E64_T100=/mnt/public/gpfs-jd/model/BASE_LLM//mcore2hf/Qwen2.5-7B-1S-NumExpert64-TP1-PP2-EP8-SI18944-EI2368-TOP8-NSHARD8-NCSHARD1-SEI_copy-REI_nv_shard-RI_nv_shard-Data01_1-02_1-03_1-04_1-05_1-06_1-08_1-09_1-10_1-11_1-12_1-13_1-17_4-19_1-Train100B-WMUP1B-LR1e-5-MIN_LR1e-7_untie
HF_E64_T200=/mnt/public/gpfs-jd/model/BASE_LLM//mcore2hf/Qwen2.5-7B-1S-NumExpert64-TP1-PP2-EP8-SI18944-EI2368-TOP8-NSHARD8-NCSHARD1-SEI_copy-REI_nv_shard-RI_nv_shard-Data01_1-02_1-03_1-04_1-05_1-06_1-08_1-09_1-10_1-11_1-12_1-13_1-17_4-19_1-Train200B-WMUP2B-LR1e-5-MIN_LR1e-7_untie

MCORE_E8_T50=/mnt/public/gpfs-jd/model/BASE_LLM/Experiments/verlmcore/Qwen2.5-7B-1S-NumExpert8-TP1-PP2-EP4-SI18944-EI18944-TOP1-NSHARD0-NCSHARD1-SEI_copy-REI_copy-RI_normal-Data01_1-02_1-03_1-04_1-05_1-06_1-08_1-09_1-10_1-11_1-12_1-13_1-17_4-Train50B-WMUP0.5B-LR1e-5-MIN_LR1e-7_untie/dist_ckpt
MCORE_E64_T50=/mnt/public/gpfs-jd/model/BASE_LLM/verlmcore/Qwen2.5-7B-1S-NumExpert64-TP1-PP2-EP8-SI18944-EI2368-TOP8-NSHARD8-NCSHARD1-SEI_copy-REI_nv_shard-RI_nv_shard-Data01_1-02_1-03_1-04_1-05_1-06_1-08_1-09_1-10_1-11_1-12_1-13_1-17_4-19_1-Train50B-WMUP0.5B-LR1e-5-MIN_LR1e-7_untie
MCORE_E64_T100=/mnt/public/gpfs-jd/model/BASE_LLM/verlmcore/Qwen2.5-7B-1S-NumExpert64-TP1-PP2-EP8-SI18944-EI2368-TOP8-NSHARD8-NCSHARD1-SEI_copy-REI_nv_shard-RI_nv_shard-Data01_1-02_1-03_1-04_1-05_1-06_1-08_1-09_1-10_1-11_1-12_1-13_1-17_4-19_1-Train100B-WMUP1B-LR1e-5-MIN_LR1e-7_untie
MCORE_E64_T200=/mnt/public/gpfs-jd/model/BASE_LLM/verlmcore/Qwen2.5-7B-1S-NumExpert64-TP1-PP2-EP8-SI18944-EI2368-TOP8-NSHARD8-NCSHARD1-SEI_copy-REI_nv_shard-RI_nv_shard-Data01_1-02_1-03_1-04_1-05_1-06_1-08_1-09_1-10_1-11_1-12_1-13_1-17_4-19_1-Train200B-WMUP2B-LR1e-5-MIN_LR1e-7_untie

HF_MODEL_PATH=$HF_E64_T50
DIST_CKPT_PATH=$MCORE_E64_T50

if [ "$RANK" -eq 0 ]; then
    python3 -m recipe.science_rl.main_dapo \
        algorithm.adv_estimator=grpo \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=512 \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=$HF_MODEL_PATH \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.override_config.moe_config.freeze_moe_router=False \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
        actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
        actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=4 \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
        actor_rollout_ref.actor.megatron.expert_model_parallel_size=8 \
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
        actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
        actor_rollout_ref.rollout.n=8 \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        ++algorithm.filter_groups.enable=${enable_filter_groups} \
        ++algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
        ++algorithm.filter_groups.metric=${filter_groups_metric} \
        custom_reward_function.path=examples/reward_function/sciencerl_reward.py \
        custom_reward_function.name=compute_score_science \
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=4 \
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
        actor_rollout_ref.ref.megatron.expert_model_parallel_size=8 \
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
        actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
        reward_model.reward_manager=dapo \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
        +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
        +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
        actor_rollout_ref.actor.checkpoint.save_contents=['model'] \
        trainer.logger=['console','tensorboard'] \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$RUN_NAME \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=$WORLD_SIZE \
        trainer.test_freq=20 \
        trainer.save_freq=300 \
        trainer.default_local_dir=$HDFS_CHECKPOINT_PATH \
        trainer.total_epochs=5
fi


