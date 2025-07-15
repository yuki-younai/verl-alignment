import json
import re
from datasets import  load_dataset, load_from_disk

# dataset = load_dataset("/mnt/public/yuannang/uni3dar_data/qwen2.5-datasets/uni3dar/qwen3dl_3d_pad_test_fixed")['test']

# print(dataset)
# print(dataset[2134])


# json_path="/mnt/public/gpfs-jd/data/RL_Data/origin/Science_chemstry/sft_dataset.json"
# with open(json_path, 'r', encoding='utf-8') as f:
#         records = [json.loads(line) for line in f if line.strip()]

json_path="/mnt/public/data_group/ssx_sft/SFTDATA_V0701/benchmark_orienteddata/mixed_data_sftformat.json"
with open(json_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

print(len(records))
# 打印第一条数据的键
if records:
    first_item = records[135]
    print(list(first_item.keys()))
    print(first_item)

# content = first_item['prompt']

# pattern = r'<answer>(.*?)</answer>'
# replacement = r'\\boxed{\1}'
# result = re.sub(pattern, replacement, content)
# print(result)

# solution = first_item['answer'][0]
# pattern_answer = r'<answer>(.*?)</answer>'
# matches_answer = re.findall(pattern_answer, solution)
# for match in matches_answer:
#     answer = match
# print(answer)







