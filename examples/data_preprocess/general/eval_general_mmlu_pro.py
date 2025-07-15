
from datasets import load_dataset
import random
import re
import pandas
from datasets import Dataset, DatasetDict, Features, Value
#https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/
dataset = load_dataset('/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/datasets/eval_datasets/mmlu-pro')['test']
def form_options(options: list):
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}\n'
    return option_str

instruction_following = "Please only provide the letter of the answer('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J') in the box. Only put the option letter in the box, e.g. \\boxed{A}. There is only one correct answer."

def process_fn(example, idx):
    question = example['question'] + '\n' + form_options(example['options']) + '\n'
    question = question + instruction_following
    data = {
        "data_source": "TIGER-Lab/MMLU-Pro",
        "prompt": [
            {"role": "user", "content": question}
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": example["answer"]
        },
        "extra_info": {
            "split": "train",
            "index": 0,
        },
    }
    return data


dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

dataset.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/mmlu_pro")






















