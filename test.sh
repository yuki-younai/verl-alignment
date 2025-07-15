


#math:"aime24 amc23 math500 minerva olympiad_bench gsm8k cmath"
#mcq: "gpqa-d supergpqa mmlu_pro mmlu"
#science: "science_physics science_chemistry science_mol2caption science_caption2mol"
#code: "humaneval humaneval_plus mbpp mbppplus leetcode2k"

cd /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/verl-main-alignment
bash eval_qwen2.sh \
    /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/exp_output/verl-dapo_20250711_091844/model_output/global_step_610/hf_model \
    "humaneval humaneval_plus mbpp mbppplus leetcode2k" \
    "default" 

cd /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/verl-main-alignment
bash eval_qwen2.sh \
    /mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B \
    "gpqa-d supergpqa mmlu_pro mmlu" \
    "no_box" 


HF_E8_T50=/mnt/public/gpfs-jd/model/BASE_LLM/Experiments/mcore2hf/Qwen2.5-7B-1S-NumExpert8-TP1-PP2-EP4-SI18944-EI18944-TOP1-NSHARD0-NCSHARD1-SEI_copy-REI_copy-RI_normal-Data01_1-02_1-03_1-04_1-05_1-06_1-08_1-09_1-10_1-11_1-12_1-13_1-17_4-Train50B-WMUP0.5B-LR1e-5-MIN_LR1e-7_untie
HF_E64_T50=/mnt/public/gpfs-jd/model/BASE_LLM//mcore2hf/Qwen2.5-7B-1S-NumExpert64-TP1-PP2-EP8-SI18944-EI2368-TOP8-NSHARD8-NCSHARD1-SEI_copy-REI_nv_shard-RI_nv_shard-Data01_1-02_1-03_1-04_1-05_1-06_1-08_1-09_1-10_1-11_1-12_1-13_1-17_4-19_1-Train50B-WMUP0.5B-LR1e-5-MIN_LR1e-7_untie
HF_E64_T100=/mnt/public/gpfs-jd/model/BASE_LLM//mcore2hf/Qwen2.5-7B-1S-NumExpert64-TP1-PP2-EP8-SI18944-EI2368-TOP8-NSHARD8-NCSHARD1-SEI_copy-REI_nv_shard-RI_nv_shard-Data01_1-02_1-03_1-04_1-05_1-06_1-08_1-09_1-10_1-11_1-12_1-13_1-17_4-19_1-Train100B-WMUP1B-LR1e-5-MIN_LR1e-7_untie
HF_E64_T200=/mnt/public/gpfs-jd/model/BASE_LLM//mcore2hf/Qwen2.5-7B-1S-NumExpert64-TP1-PP2-EP8-SI18944-EI2368-TOP8-NSHARD8-NCSHARD1-SEI_copy-REI_nv_shard-RI_nv_shard-Data01_1-02_1-03_1-04_1-05_1-06_1-08_1-09_1-10_1-11_1-12_1-13_1-17_4-19_1-Train200B-WMUP2B-LR1e-5-MIN_LR1e-7_untie

cd /mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/verl-main-alignment
bash eval_qwen2moe.sh \
    /mnt/public/gpfs-jd/model/RL_Model/verl-qwen2moe-science_20250710_161712/model_output/global_step_425/hf_model \
    "humaneval humaneval_plus mbpp mbppplus leetcode2k" \
    "default" 

