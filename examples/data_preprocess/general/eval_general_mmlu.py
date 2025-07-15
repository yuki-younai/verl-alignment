
from datasets import load_dataset
import random
import re
import pandas
from datasets import Dataset, DatasetDict, Features, Value
#https://huggingface.co/datasets/cais/mmlu

category = ['abstract_algebra',  'anatomy', 'astronomy', 'auxiliary_train', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

dataset = load_dataset('/mnt/public/gpfs-jd/data/RL_Data/origin/mmlu', 'all')['test']
def form_options(options: list):
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}\n'
    return option_str

instruction_following = "Please only provide the letter of the answer('A', 'B', 'C', 'D') in the box. Only put the option letter in the box, e.g. \\boxed{A}. There is only one correct answer."

def mcq_answer(num):
    if num==0:
        return 'A'
    elif num==1:
        return 'B'
    elif num==2:
        return 'C'
    elif num==3:
        return 'D'

def process_fn(example, idx):
    question = example['question'] + '\n' + form_options(example['choices']) + '\n'
    question = question + instruction_following
    data = {
        "data_source": "cais/mmlu",
        "prompt": [
            {"role": "user", "content": question}
        ],
        "ability": "mcq",
        "reward_model": {
            "style": "rule",
            "ground_truth": mcq_answer(example['answer'])
        },
        "extra_info": {
            "split": "train",
            "index": 0,
        },
    }
    return data


dataset = dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)
print(dataset)
print(dataset[0])
dataset.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/mmlu")






















