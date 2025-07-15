

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
VERL_PATH=${CURRENT_DIR}
echo $VERL_PATH
export PYTHONPATH=${VERL_PATH}:${VERL_PATH}/verl:$PYTHONPATH

HF_MODEL_PATH=/mnt/public/gpfs-jd/model/BASE_LLM//mcore2hf/Qwen2.5-7B-1S-NumExpert64-TP1-PP2-EP8-SI18944-EI2368-TOP8-NSHARD8-NCSHARD1-SEI_copy-REI_nv_shard-RI_nv_shard-Data01_1-02_1-03_1-04_1-05_1-06_1-08_1-09_1-10_1-11_1-12_1-13_1-17_4-19_1-Train200B-WMUP2B-LR1e-5-MIN_LR1e-7_untie
basename=$(basename "$HF_MODEL_PATH}")
DIST_CKPT_PATH=/mnt/public/gpfs-jd/model/BASE_LLM/verlmcore


python scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH/$basename


















