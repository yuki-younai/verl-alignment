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
    data_source = "/mnt/public/gpfs-jd/data/RL_Data/origin/Science_chemstry"

    dataset = datasets.load_dataset(data_source)
    train_dataset = dataset["train"]

    def process_fn(example, idx):
        #['prompt', 'answer', 'answer_type', 'solution', 'category']
        question_raw = example['prompt']
        question = re.sub(pattern, replacement, question_raw)
        solution = example['answer'][0]
        matches_answer = re.findall(pattern_answer, solution)
        for match in matches_answer:
            answer = match
        instruction_following =  "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
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

    train_dataset = train_dataset.map(function=process_fn, with_indices=True, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle()
    dataset = train_dataset.train_test_split(test_size=0.20, seed=42)['test']
    
    train_test_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = train_test_dataset['train']
    test_data =  train_test_dataset['test']

    print(train_data)
    print(test_data)
    print(test_data[0])
    local_dir = "/mnt/public/gpfs-jd/data/RL_Data/RL_Train/chemistry5k"
    train_data.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_data.to_parquet(os.path.join(local_dir, "test.parquet"))
    #test_data.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/science_chemistry")

