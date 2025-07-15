# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
from datasets import concatenate_datasets
import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    data_source = "/mnt/public/gpfs-jd/data/RL_Data/origin/Science_physics"
    dataset = datasets.load_dataset(data_source)
    dataset = dataset['train']

    def process_fn(example, idx):
        question = example["prompt"]
        answer = example["answer"]
        solution = example["solution"]
        instruction_following =  "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
        data = {
            "data_source": "science_physics",
            "prompt": [
                {
                    "role": "user",
                    "content": question + instruction_following,
                }
            ],
            "ability": "science_physics",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "train",
                "index": idx,
                "reference": solution+f"\nSo the answer is \\boxed{{{answer}}}.\n\n",
                "original_prompt": question,
                "prefix": "None",
            },
        }
        return data

    dataset_train_test_split = dataset.train_test_split(test_size=0.6, seed=42)
    
    sft_dataset = dataset_train_test_split['test']
    rl_dataset = dataset_train_test_split['train']

    sft_dataset = sft_dataset.map(function=process_fn, with_indices=True, remove_columns=sft_dataset.column_names)
    rl_dataset = rl_dataset.map(function=process_fn, with_indices=True, remove_columns=rl_dataset.column_names)
    
    rl_dataset_filter = rl_dataset.train_test_split(test_size=0.1, seed=42)
    rl_train_dataset = rl_dataset_filter['train']
    rl_test_dataset = rl_dataset_filter['test']
    
    print(sft_dataset)
    print(rl_train_dataset)
    print(rl_test_dataset)
    print(rl_test_dataset[0])
    sft_dataset.to_parquet(os.path.join("/mnt/public/gpfs-jd/data/SFT_Data/SFT_data_v2/physics_v2", "train.parquet"))
    rl_train_dataset.to_parquet(os.path.join("/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v2/physics", "train.parquet"))
    rl_test_dataset.to_parquet(os.path.join("/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v2/physics", "test.parquet"))
    #test_data.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/science_physics")

































