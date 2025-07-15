from datasets import load_dataset
import random
import re
import pandas
from datasets import Dataset, DatasetDict, Features, Value
#https://huggingface.co/datasets/openai/gsm8k
dataset = load_dataset('/mnt/public/gpfs-jd/data/RL_Data/origin/Evaluation_data/gsm8k')['test']

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def process_fn(example, idx):
    answer_raw = example["answer"]
    solution = extract_solution(answer_raw)
    data = {
        "data_source": "openai/gsm8k",
        "prompt": [
            {"role": "user", "content": example["question"]}
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            "split": "train",
            "index": idx,
            "solution": answer_raw
        },
    }
    return data

dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

print(dataset[0])

dataset.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/gsm8k")

