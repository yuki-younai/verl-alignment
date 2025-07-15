"""Downloads, processes, and saves math datasets."""

import argparse
import os
from typing import Any, Dict
import datasets


def make_map_fn(split: str, data_source: str, reward_metric: str = "default") -> callable:
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # if original_question in example, use it as the question
        question = example.pop("problem")
        # question = example.pop("problem")
        answer = example.pop("answer")
        if isinstance(answer, list):
            answer = answer[0]
        instruction_following =  "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
        data = {
            "data_source": "math",
            "prompt": [
                {"role": "user", "content": question + instruction_following},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "reward_metric": reward_metric,
                "original_question": question,
            },
        }

        if idx == 0 or idx == 1:
            print(f"data_source: {data_source}, split: {split}, idx: {idx}")
            print("\n" + "=" * 100 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
        return data

    return process_fn


if __name__ == "__main__":
    """Main script execution: parse args, load, process, and save datasets."""

    output_path = "/mnt/public/gpfs-jd/data/RL_Data/RL_Train/math/"
    filter_frac = 0.2
    train_data_source = "/mnt/public/gpfs-jd/data/RL_Data/origin/SDSBmerged_deduped_dapo_or1_dataset"
    #https://huggingface.co/datasets/SDSB/merged_deduped_dapo_or1_dataset
    train_dataset = datasets.load_dataset(train_data_source, trust_remote_code=True, split="train")
    print(train_dataset)
    # Process train dataset
    process_train_fn = make_map_fn(split="train", data_source="math")
    train_data = train_dataset.map(function=process_train_fn, with_indices=True, remove_columns=train_dataset.column_names)
    # Sample
    train_data_filter = train_data.train_test_split(test_size=filter_frac, seed=42)['test']
    # Save
    print(len(train_data_filter))
    print(train_data_filter[0])
    train_data_filter.to_parquet(os.path.join(output_path+"dapomix_retain"+str(filter_frac), "train.parquet"))
    
    print(f"Done! \nTrain data saved to {output_path}")


