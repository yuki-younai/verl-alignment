import argparse
import json
import os
import datasets
import transformers
from datasets import load_dataset
from tqdm import tqdm



non_numeric_answer = 0
non_arithmetic_question = 0

PromptTemplate = """{{context}}"""



def process_fn(example, idx):
    prompt = example["prompt"]
    instruction_id_list = example["instruction_id_list"]
    kwargs = example["kwargs"]

    data = {
        "data_source": "google/IFEval",
        "prompt": [
            {
                "role": "user",
                "content": PromptTemplate.replace("{{context}}", prompt)
            }
        ],
        "ability": "ood",
        "reward_model": {
            "style": "rule",
            "ground_truth": kwargs,
        },
        "extra_info": {
            "instruction_id_list": instruction_id_list,
            "prompt": prompt,
        }
    }

    if idx == 0 or idx == 1:
        print("\n" + "=" * 10 + f" {idx}" + "=" * 10)
        print(data)

    return data



if __name__ == "__main__":
    
    #https://huggingface.co/datasets/google/IFEval
    dataset = load_dataset("/mnt/public/gpfs-jd/data/RL_Data/origin/ifeval")["train"] 
    # Process datasets
    dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)

    dataset.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/ifeval")

