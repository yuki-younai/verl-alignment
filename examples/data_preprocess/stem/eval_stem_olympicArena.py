
from datasets import load_dataset
import random
import re
import pandas
from datasets import Dataset, DatasetDict, Features, Value


dataset = load_dataset('/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/datasets/eval_datasets/OlympicArena/Biology', split='validation')
print(dataset)
#https://huggingface.co/datasets/GAIR/OlympicArena
data_path='/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/datasets/eval_datasets/OlympicArena/'
categorys=["Astronomy", "Biology","Chemistry", "CS", "Geography", "Math", "Physics"]

# 定义过滤函数
def filter_fn(example):
    return example['modality'] == "text-only"

results = []

for cate in categorys:
    print(cate)
    dataset = load_dataset(data_path+str(cate))['validation']
    print(dataset)
    filtered_dataset = dataset.filter(function=filter_fn)
    
    for data in filtered_dataset:
        data = {
            "data_source": "GAIR/OlympicArena",
            "prompt": [
                {"role": "user", "content": data['prompt']}
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": data['answer']
            },
            "extra_info": {
                "split": "train",
                "index": 0,
            },
        }
        results.append(data)
        
    print(len(results))

dataset = Dataset.from_list(results)
print()
dataset.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/OlympicArena")





















