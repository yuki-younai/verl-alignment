
from datasets import load_dataset
import random
import re
import pandas
from datasets import Dataset, DatasetDict, Features, Value
#https://huggingface.co/datasets/weitianwen/cmath
dataset = load_dataset('/mnt/public/gpfs-jd/data/RL_Data/origin/cmath')['test']

def process_fn(example, idx):

    data = {
        "data_source": "weitianwen/cmath",
        "prompt": [
            {"role": "user", "content": example["question"]}
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": example["golden"]
        },
        "extra_info": {
            "split": "train",
            "index": idx
        },
    }
    return data

dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

print(dataset)
print(dataset[0])

dataset.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/cmath")






















