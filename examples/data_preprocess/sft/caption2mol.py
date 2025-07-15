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
    sft_data_source = "/mnt/public/gpfs-jd/data/RL_Data/science_data/caption2mol_sft"
    rl_data_source = "/mnt/public/gpfs-jd/data/RL_Data/science_data/caption2mol_rl"
    
    sft_dataset = datasets.load_dataset(sft_data_source)['train']
    rl_dataset = datasets.load_dataset(rl_data_source)
    rl_dataset_train = rl_dataset['train']
    rl_dataset_test = rl_dataset['test']

    def process_fn_sft(example, idx):
        #['prompt', 'answer', 'answer_type', 'solution', 'category']
        question_raw = example['prompt']
        question = re.sub(pattern, replacement, question_raw)
        answer_raw = example['answer'][0]
        answer = re.sub(pattern, replacement, answer_raw)
        solution_raw = example['solution']
        solution = re.sub(pattern, replacement, solution_raw)
        instruction_following =  "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
        data = {
            "data_source": "science_caption2mol",
            "prompt": [
                {
                    "role": "user",
                    "content": question + instruction_following,
                }
            ],
            "ability": "science_caption2mol",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "train",
                "index": idx,
                "reference": solution,
                "original_prompt": ' ',
                "prefix": "None",
            },
        }
        return data
    def process_fn_rl(example, idx):
        #['prompt', 'answer', 'answer_type', 'solution', 'category']
        question_raw = example['prompt'][0]['content']
        question = re.sub(pattern, replacement, question_raw)
        answer_raw = example['reward_model']['ground_truth']
        answer = re.sub(pattern, replacement, answer_raw)
        instruction_following =  "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
        data = {
            "data_source": "science_caption2mol",
            "prompt": [
                {
                    "role": "user",
                    "content": question + instruction_following,
                }
            ],
            "ability": "science_caption2mol",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "train",
                "index": idx,
                "reference": ' ',
                "original_prompt": ' ',
                "prefix": "None",
            },
        }
        return data
    
    sft_dataset = sft_dataset.map(function=process_fn_sft, with_indices=True, remove_columns=sft_dataset.column_names)
    rl_dataset_train = rl_dataset_train.map(function=process_fn_rl, with_indices=True, remove_columns=rl_dataset_train.column_names)
    test_dataset = rl_dataset_test.map(function=process_fn_rl, with_indices=True, remove_columns=rl_dataset_test.column_names)
    rl_dataset_test = test_dataset.select(range(600))
    rl_dataset_train = rl_dataset_train.shuffle()
    rl_dataset_train_split = rl_dataset_train.select(range(6000))
    
    print(sft_dataset)
    print(rl_dataset_train)
    print(rl_dataset_test)
    print(test_dataset)
    print(sft_dataset[0])
    print(rl_dataset_test[0])
    #sft_dataset.to_parquet(os.path.join("/mnt/public/gpfs-jd/data/SFT_Data/SFT_data_v3/caption2mol-full-24k", "train.parquet"))
    #rl_dataset_train.to_parquet(os.path.join("/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/caption2mol-full-21k", "train.parquet"))
    #rl_dataset_test.to_parquet(os.path.join("/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/caption2mol-full-21k", "test.parquet"))
    #test_dataset.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/science_caption2mol")
    print(rl_dataset_train_split)
    rl_dataset_train_split.to_parquet(os.path.join("/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/caption2mol-split-6k", "train.parquet"))
    rl_dataset_test.to_parquet(os.path.join("/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/caption2mol-split-6k", "test.parquet"))    



