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

pattern = r'<answer>(.*?)</answer>'
replacement = r'\\boxed{\1}'

pattern_answer = r'<answer>(.*?)</answer>'

if __name__ == "__main__":
    data_source = "/mnt/public/gpfs-jd/data/RL_Data/science_data/chemistry"

    dataset = datasets.load_dataset(data_source)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    def process_fn(example, idx):
        #['prompt', 'answer', 'answer_type', 'solution', 'category']
        question_raw = example['prompt']
        question = re.sub(pattern, replacement, question_raw)
        solution = example['answer'][0]
        matches_answer = re.findall(pattern_answer, solution)
        for match in matches_answer:
            answer = match
        instruction_following =  "\nPlease reason step by step, and put your final answer within \\boxed{{}}.(True or False)\n\n"
        data = {
            "data_source": "science_chemstry",
            "prompt": [
                {
                    "role": "user",
                    "content": question+instruction_following,
                }
            ],
            "ability": "science_chemistry",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "train",
                "index": idx,
                "reference": solution+f"\nSo the answer is \\boxed{{{answer}}}.\n\n",
                "original_prompt": question_raw,
                "prefix": "None",
            },
        }
        return data

    sft_dataset = train_dataset.map(function=process_fn, with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=process_fn, with_indices=True, remove_columns=test_dataset.column_names)
    
    sft_dataset = sft_dataset.shuffle()

    rl_dataset = test_dataset.train_test_split(test_size=0.1, seed=42)
    rl_train_data = rl_dataset['train']
    test_data =  rl_dataset['test']
    rl_test_data = test_data.select(range(600))
    
    rl_train_data = rl_train_data.shuffle()
    rl_train_data_split = rl_train_data.select(range(8000))
    
    
    print(test_data)
    print(sft_dataset)
    print(rl_train_data)
    print(rl_test_data)
    print(rl_test_data[0])
    # sft_dataset.to_parquet(os.path.join("/mnt/public/gpfs-jd/data/SFT_Data/SFT_data_v3/chemistry", "train.parquet"))
    # rl_train_data.to_parquet(os.path.join("/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/chemistry-full-38k", "train.parquet"))
    # rl_test_data.to_parquet(os.path.join("/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/chemistry-full-38k", "test.parquet"))
    # test_data.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/science_chemistry")
    print(rl_train_data_split)
    rl_train_data_split.to_parquet(os.path.join("/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/chemistry-split-8k", "train.parquet"))
    rl_test_data.to_parquet(os.path.join("/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/chemistry-split-8k", "test.parquet"))
    
    
    
    