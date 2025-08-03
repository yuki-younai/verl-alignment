https://huggingface.co/datasets/yukiyounai/easy-eval-datasets









## Evaluation

#math:"aime24 amc23 math500 minerva olympiad_bench gsm8k cmath"
#mcq: "gpqa-d supergpqa mmlu_pro mmlu"
#science: "science_physics science_chemistry science_mol2caption science_caption2mol"
#code: "humaneval humaneval_plus mbpp mbppplus leetcode2k"

cd /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/verl-main-alignment
bash eval_qwen2.sh \
    /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/exp_output/verl-dapo_20250711_091844/model_output/global_step_610/hf_model \
    "humaneval humaneval_plus mbpp mbppplus leetcode2k" \
    "default" 













