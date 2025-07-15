
from datasets import load_dataset
import random
import re
import pandas
from datasets import Dataset, DatasetDict, Features, Value
#https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024
dataset = load_dataset('/mnt/public/gpfs-jd/data/RL_Data/origin/Evaluation_data/aime2024-32')['train']

instruction_following =  "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
def process_fn(example, idx):

    data = {
        "data_source": "math",
        "prompt": [
            {"role": "user", "content": example['extra_info']["raw_problem"]+instruction_following}
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": example["reward_model"]["ground_truth"]
        },
        "extra_info": {
            "split": "train",
            "index": idx,
            "reference": "None",
            "original_prompt": example['extra_info']["raw_problem"],
            "prefix": "None",
        },
    }
    return data

dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

print(dataset[0])

#dataset.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/aime24")
import os
local_dir="/mnt/public/gpfs-jd/data/RL_Data/RL_Train/math/dapo17k"
dataset.to_parquet(os.path.join(local_dir, "test.parquet"))




















