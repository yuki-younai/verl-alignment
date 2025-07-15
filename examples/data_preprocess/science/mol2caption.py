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
    data_source = "/mnt/public/gpfs-jd/data/RL_Data/origin/Science_mol2caption"

    dataset = datasets.load_dataset(data_source)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    def process_fn(example, idx):
        #['prompt', 'answer', 'answer_type', 'solution', 'category']
        question = example['prompt']
        answer = example['answer'][0]
        solution = example['solution']
        instruction_following =  "\nPlease reason step by step, and put your final answer within <answer> answer here </answer>.\n\n"
        data = {
            "data_source": "science_mol2caption",
            "prompt": [
                {
                    "role": "user",
                    "content": question + instruction_following,
                }
            ],
            "ability": "science_mol2caption",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "train",
                "index": idx,
                "reference": solution,
                "original_prompt": question,
                "prefix": "None",
            },
        }
        return data
    train_data = train_dataset.map(function=process_fn, with_indices=True, remove_columns=train_dataset.column_names)
    train_data = train_data.shuffle()
    dataset_train_test_split = train_data.train_test_split(test_size=0.7, seed=42)
    sft_dataset = dataset_train_test_split['test']
    rl_dataset = dataset_train_test_split['train']
    
    test_data = test_dataset.map(function=process_fn, with_indices=True, remove_columns=test_dataset.column_names)
    test_data = test_data.shuffle()
    test_data = test_data.select(range(1000))
    
    print(sft_dataset)
    print(rl_dataset)
    print(test_data)
    print(test_data[0])
    local_dir = "/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v2/mol2caption"
    sft_dataset.to_parquet(os.path.join("/mnt/public/gpfs-jd/data/SFT_Data/mol2caption_v2", "train.parquet"))
    rl_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_data.to_parquet(os.path.join(local_dir, "test.parquet"))
    #test_data.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/science_mol2caption")


