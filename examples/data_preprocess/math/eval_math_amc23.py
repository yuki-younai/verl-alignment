
from datasets import load_dataset
import random
import re
import pandas
from datasets import Dataset, DatasetDict, Features, Value
#https://huggingface.co/datasets/zwhe99/amc23
dataset = load_dataset('/mnt/public/gpfs-jd/data/RL_Data/origin/Evaluation_data/amc23')['train']

def process_fn(example, idx):

    data = {
        "data_source": "zwhe99/amc23",
        "prompt": [
            {"role": "user", "content": example["question"]}
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": example["answer"]
        },
        "extra_info": {
            "split": "train",
            "index": idx
        },
    }
    return data

dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

print(dataset[0])

dataset.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/amc23")






















