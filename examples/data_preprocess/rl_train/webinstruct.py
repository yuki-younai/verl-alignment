"""Downloads, processes, and saves math datasets."""

import argparse
import os
from typing import Any, Dict
import datasets



def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:

    answer_type = example["answer_type"]
    if answer_type=="Multiple Choice":
        instruction_following =  "Only put the option letter in the box, e.g. \\boxed{A}. There is only one correct answer.\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    elif answer_type=="Boolean":
        instruction_following =  "\nPlease reason step by step, and put your final answer within \\boxed{{}}.(True or False)\n\n"
    else:
        instruction_following =  "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    
    answer = example["answer"]
    if answer=="Yes" or answer=="yes":
        answer="True"
    elif answer=="No" or answer=="no":
        answer="False"
    
    data = {
        "data_source": "stem",
        "prompt": [
            {"role": "user", "content": example["question"]+instruction_following},
        ],
        "ability": "stem",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": "train",
            "index": idx,
            "reference": answer,
            "original_prompt": example["question"],
            "prefix": answer_type,
        },
    }

    return data


def filter_fn(example):
    answer_type = example['answer_type']
    answer = example["answer"]
    if answer_type=="Boolean":
        if answer not in ["True", "true", "False", "false"]:
            return False
    if answer_type=="Integer":
        if len(answer)>10:
            return False
    if answer_type=="Multiple Choice":
        if len(answer)>=2:
            return False
        
    return answer_type == "Multiple Choice" or answer_type=="Boolean" or  answer_type=="Integer"

if __name__ == "__main__":
    """Main script execution: parse args, load, process, and save datasets."""

    train_data_source = "/mnt/public/gpfs-jd/data/RL_Data/origin/WebInstruct-verified"
    #https://huggingface.co/datasets/TIGER-Lab/WebInstruct-verified
    train_dataset = datasets.load_dataset(train_data_source, trust_remote_code=True, split="train")
    filtered_dataset = train_dataset.filter(function=filter_fn)
    
    train_data = filtered_dataset.map(function=process_fn, with_indices=True, remove_columns=filtered_dataset.column_names)
    
    datasets = train_data.train_test_split(test_size=0.2, seed=42)['test']
    
    
    datasets_split = datasets.train_test_split(test_size=0.08, seed=42)
    train_data = datasets_split['train']
    test_data =  datasets_split['test']
    
    print(train_data)
    print(test_data)
    print(test_data[0])
    output_path = "/mnt/public/gpfs-jd/data/RL_Data/RL_Train/science/"
    train_data.to_parquet(os.path.join(output_path+"webinstruct", "train.parquet"))
    test_data.to_parquet(os.path.join(output_path+"webinstruct", "test.parquet"))
    test_data.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/webinstrcut")
    print(f"Done! \nTrain data saved to {output_path}")




