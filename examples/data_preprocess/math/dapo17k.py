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

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./dapo17k-verl")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()
    #https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed
    data_source = "/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/datasets/train_dataset/dapo-process"

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]

    instruction_following =  "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
    
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["prompt"]
            question = question_raw + instruction_following
            data = {
                "data_source": "math",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": example["reward_model"],
                "extra_info": {
                    "split": "train",
                    "index": idx,
                    "reference": "None",
                    "original_prompt": question_raw,
                    "prefix": "None",
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    print(train_dataset)
    local_dir="/mnt/public/gpfs-jd/data/RL_Data/RL_Train/math/dapo17k"
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    print(local_dir)
