export CUDA_VISIBLE_DEVICES=0,1,2,3
#Best
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/word2number-1.1.zip
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/antlr4_python3_runtime-4.13.2-py3-none-any.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/latex2sympy2_extended-1.0.6-py3-none-any.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/whl_files/math_verify-0.5.2-py3-none-any.whl
#Mol2caption or Caption2mol
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/whl_files/selfies-2.2.0-py3-none-any.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/whl_files/rdkit-2025.3.3-cp310-cp310-manylinux_2_28_x86_64.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/whl_files/nltk-3.9.1-py3-none-any.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/whl_files/rapidfuzz-3.13.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/whl_files/levenshtein-0.27.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/whl_files/rouge_score-0.1.2.tar.gz
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/whl_files/langdetect-1.0.9.tar.gz
pip install --no-index /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/whl_files/immutabledict-4.2.1-py3-none-any.whl

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
VERL_PATH=${CURRENT_DIR}
echo $VERL_PATH
export PYTHONPATH=${VERL_PATH}:${VERL_PATH}/examples:$PYTHONPATH

#/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2.5-Math-7B no qwen25_math
#/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B no_box
#/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B-Instruct default
current_time=$(date +"%Y%m%d_%H%M%S")
HF_SAVE_PATH=$(dirname "$MODEL_PATH")
echo $HF_SAVE_PATH

#Math: "aime24" "amc23" "math500" "minerva" "olympiad_bench" "gsm8k"
#stem: "gpqa-d" "supergpqa" 
#General: "mmlu_pro"
#Code: "humaneval"
#Science: "science_physics" "science_chemistry" "science_mol2caption" "science_caption2mol"

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l 2>/dev/null || echo 0)

MODEL_PATH=${1:-"/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B-Instruct"}  
TASKS_LIST=${2:-"aime24 amc23 math500 minerva olympiad_bench gsm8k"}
TEMPLATE=${3:-"default"}
basename=$(basename "${MODEL_PATH}")


echo "================================================================="
echo "启动任务: $TASKS_LIST "
echo "模型地址: $MODEL_PATH 模板: $TEMPLATE"
echo "================================================================="

python scripts/model_eval.py --model_path $MODEL_PATH \
                    --verify_model_path /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/datasets/general-verifiy \
                    --dataset_name /mnt/public/gpfs-jd/data/RL_Data/eval_data \
                    --output_path  "/mnt/public/gpfs-jd/data/RL_Data/eval_results/${basename}_${current_time}" \
                    --template $TEMPLATE \
                    --tasks $TASKS_LIST\
                    --temperature 0.0 \
                    --num_generation 1 \
                    --repeat_num 3 \
                    --tensor_parallel_size 1


