
from datasets import load_dataset
import random
import re
import pandas
from datasets import Dataset, DatasetDict, Features, Value
#https://huggingface.co/datasets/HuggingFaceH4/MATH-500
dataset = load_dataset('/mnt/public/gpfs-jd/data/RL_Data/origin/Evaluation_data/math500')['test']

def process_fn(example, idx):

    data = {
        "data_source": "HuggingFaceH4/MATH-500",
        "prompt": [
            {"role": "user", "content": example["problem"]}
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": example["answer"]
        },
        "extra_info": {
            "split": "train",
            "index": idx,
            "solution": example["solution"]
        },
    }
    return data

dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

print(dataset[0])

dataset.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/math500")






















