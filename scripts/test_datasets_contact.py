from datasets import  load_dataset, load_from_disk
import os
from datasets import concatenate_datasets

math_path="/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/math/dapo10k"
physics_path="/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/physics-full-17k"
stem_path="/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/science/webinstruct-split-13k"
code_path="/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/code/leetcode2k"
chemistry_path="/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/chemistry-split-8k"
caption2mol_path="/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/caption2mol-split-6k"
mol2caption_path="/mnt/public/gpfs-jd/data/RL_Data/RL_Train_v3/mol2caption-split-6k"

dataset1 = load_dataset(math_path)['train']
dataset2 = load_dataset(physics_path)['train']
#dataset3_1 = load_dataset("/mnt/public/gpfs-jd/data/RL_Data/RL_Train/code/code_taco")["train"]
dataset3_2 = load_dataset(code_path)["train"]
dataset4 = load_dataset(chemistry_path)["train"]
dataset5 = load_dataset(caption2mol_path)["train"]
dataset6 = load_dataset(mol2caption_path)["train"]
dataset7 = load_dataset(stem_path)["train"]
print("Math:", len(dataset1))
print("physics:", len(dataset2))
print("code:", len(dataset3_2))
print("chemistry:", len(dataset4))
print("caption2mol:", len(dataset5))
print("mol2caption:", len(dataset6))
print("stem:", len(dataset7))
combined_dataset = concatenate_datasets([dataset1, dataset2, dataset3_2, dataset4, dataset5, dataset6, dataset7])

print(combined_dataset)

dataset1 = load_dataset(math_path)['test']
dataset2 = load_dataset(physics_path)['test']
dataset4 = load_dataset(chemistry_path)["test"]
dataset5 = load_dataset(caption2mol_path)["test"]
dataset6 = load_dataset(mol2caption_path)["test"]
dataset7 = load_dataset(stem_path)["test"]
print("Math:", len(dataset1))
print("physics:", len(dataset2))
print("chemistry:", len(dataset4))
print("caption2mol:", len(dataset5))
print("mol2caption:", len(dataset6))
print("stem:", len(dataset7))
combined_dataset = concatenate_datasets([dataset1, dataset2, dataset4, dataset5, dataset6, dataset7])

print(combined_dataset)


