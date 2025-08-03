export CUDA_VISIBLE_DEVICES=0

#Train
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/word2number-1.1.zip
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/latex2sympy2_extended-1.10.1-py3-none-any.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/math_verify-0.7.0-py3-none-any.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/antlr4-python3-runtime-4.7.2.tar.gz
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/latex2sympy2-1.9.1-py3-none-any.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/antlr4-python3-runtime-4.9.3.tar.gz
MODEL_SVAE_PATH=/mnt/public/gpfs-jd/code/guoweiyang/LRM/exp_out/verl-grpo-train-20250722_144845/model_output/global_step_330/actor
HF_SAVE_PATH="${MODEL_SVAE_PATH%/*}"

python -m verl.model_merger merge \
    --backend fsdp  \
    --local_dir $MODEL_SVAE_PATH \
    --target_dir $HF_SAVE_PATH/hf_model

# MODEL_SVAE_PATH=/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/exp_output/verl-grpo-physics-singlenode_20250626_022933/model_output/global_step_120/actor
# HF_SAVE_PATH="${MODEL_SVAE_PATH%/*}"

# echo $HF_SAVE_PATH
# python scripts/model_merger.py merge \
#     --backend fsdp  \
#     --local_dir $MODEL_SVAE_PATH \
#     --target_dir $HF_SAVE_PATH/hf_model
# # fsdp 
# # megatron
