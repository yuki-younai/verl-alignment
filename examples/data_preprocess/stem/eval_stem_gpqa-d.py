from datasets import load_dataset
import random
import re
import pandas
from datasets import Dataset, DatasetDict, Features, Value
rng = random.Random(0)
df = pandas.read_csv(
    "/mnt/public/gpfs-jd/code/guoweiyang/Pai-Megatron-Upcycle-own/Reasoning/datasets/eval_datasets/gpqa/gpqa_diamond.csv"
)
#https://huggingface.co/datasets/Idavidrein/gpqa
examples = [row.to_dict() for _, row in df.iterrows()]
examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]

QUERY_TEMPLATE = """
<question>

A: <a>
B: <b>
C: <c>
D: <d>

Only put the option letter in the box, e.g. \\boxed{A}. There is only one correct answer.
""".strip()

results = []
for row in examples:
    #print(row.keys())
    temp = {}
    prompt = row["Question"]
    answer = row["Correct Answer"]
    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    choices = [choices[i] for i in row["permutation"]]
    correct_index = choices.index(row["Correct Answer"])
    correct_answer = "ABCD"[correct_index]
    choices_dict = dict(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
    )
    content = QUERY_TEMPLATE.replace("<question>", row["Question"])
    content = content.replace("<a>", choices[0])
    content = content.replace("<b>", choices[1])
    content = content.replace("<c>", choices[2])
    content = content.replace("<d>", choices[3])
    
    temp["problem"] = content
    temp['answer'] = correct_answer
    data = {
        "data_source": "gpqa-d",
        "prompt": [
            {"role": "user", "content": content}
        ],
        "ability": "stem",
        "reward_model": {
            "style": "rule",
            "ground_truth": correct_answer
        },
        "extra_info": {
            "split": "train",
            "index": 0,
        },
    }
    results.append(data)



dataset = Dataset.from_list(results)
# 2. 定义数据集特征结构（schema）
print(dataset[0])
dataset.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/gpqa-d")
# import os
# output_path = "/mnt/public/gpfs-jd/data/RL_Data/RL_Train/science/"
# dataset.to_parquet(os.path.join(output_path+"webinstruct", "test.parquet"))




