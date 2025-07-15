import json
import os
import torch
import sys
import time
import random
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, set_seed
from datasets import load_from_disk
from examples.reward_function.math_reward import compute_score_math_eval as compute_math_score
from examples.reward_function.math_reward import compute_score_mcq_eval as compute_mcq_score
from examples.reward_function.coder1_reward import compute_score_code_eval as compute_code_score
#from examples.reward_function.ifeval_reward import compute_score_ifeval_eval as compute_ifeval_score
from examples.utils.vllm_models import VLLM_models


def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
def apply_r1_template(question: str, model_name):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )
def save_to_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


TASK2Reward={compute_math_score: ["amc23", "aime24", "math500", "minerva", "olympiad_bench", "gsm8k", "cmath"],
             compute_mcq_score: ["gpqa-d", "mmlu_pro", "supergpqa", "mmlu"],
             compute_code_score: ["humaneval", "humaneval_plus", "mbpp", "mbppplus", "leetcode2k"] }

def chose_config(model_name, task_name, template, tokenizer, use_best_config=False):
    
    if template=='qwen_math':
        apply_template = apply_qwen_math_template
    elif template=='r1':
        apply_template = apply_r1_template
    elif template=="no_box":
        if task_name in ["leetcode2k", "humaneval", "humaneval_plus", "mbpp", "mbppplus", "ifeval"]:
            instrcution = " "
        else:
            instrcution="\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
        apply_template = lambda x: x+instrcution
    elif template=="no":
        apply_template = lambda x: x
    elif template=="default":
        if task_name in ["leetcode2k", "humaneval", "humaneval_plus", "mbpp", "mbppplus", "ifeval"]:
            instrcution = " "
        else:
            instrcution="\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
        def apply_default_template(question):
            return tokenizer.apply_chat_template(
                [
                    {
                        "content": question + instrcution,
                        "role": "user",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        apply_template = apply_default_template

    for func, tasks in TASK2Reward.items():
        if task_name in tasks:
            return func, apply_template
        
    raise ValueError(f"!!!!!未找到与 '{task_name}' 匹配的奖励函数!!!!!")
    
    
def main(args):

    model = VLLM_models(model_name_or_path=args.model_path, device=0, gpu_memory_utilization=0.7, tensor_parallel_size=args.tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    avg_acc = {}
    for task_name in args.tasks:
        avg_acc[task_name] = []

    for epoch in range(args.repeat_num):
        set_seed(seed=random.randint(0, 100))
        results = {}
        avg_lens = {}
        max_lens = {}
        formatted = {}
        for task_name, dataset in load_from_disk(args.dataset_name).items():
            ratio = 200/len(dataset)
            to_be_saved = []
            if task_name not in args.tasks:
                continue
            compute_score, apply_template = chose_config(args.model_path, task_name, args.template, tokenizer, use_best_config=False)    
            prompts = [data[0]['content'] for data in dataset["prompt"]]
            targets = [data["ground_truth"] for data in dataset["reward_model"]]
            outputs = model.generate(prompts, apply_template, temperature=args.temperature, num_generation= args.num_generation)
            batch_scores = []
            batch_formatted = []
            batch_lengths = []
            for k in range(len(outputs)):
                output = outputs[k]
                datasets_repeated = [dataset[k]] * model.sampling_params.n
                gt_repeated = [targets[k]] * model.sampling_params.n
                rewards, formats = [], []
                model_answers =  []
                for data, model_output in zip(datasets_repeated, [o.text for o in output.outputs]):
                    result = compute_score(data_source=data["data_source"], solution_str=model_output, 
                                            ground_truth=data["reward_model"]["ground_truth"], extra_info=data["extra_info"])
                    if random.random()<0.01:
                        print("#####################################")
                        print("[model_ouput]:", model_output)
                        print("[reward]:", result)
                    rewards.append(result["score"])
                    formats.append(result["formatted"])
                    model_answers.append(result["pred"])
                    
                rewards = np.array(rewards)
                batch_formatted.append(np.array(formats).mean())
                batch_lengths.append([len(o.token_ids) for o in output.outputs])
                batch_scores.append(max(rewards))

                if random.random()<ratio:
                    to_be_saved.append(
                        {
                            "task_name": task_name,
                            "prompt": output.prompt,
                            "gt": gt_repeated,
                            "model_output": [o.text for o in output.outputs],
                            "model_answer": model_answers ,
                            "reward": [int(r) for r in rewards],
                        }
                    )
            save_to_json(to_be_saved, args.output_path+"/"+str(task_name)+"_epochs"+str(epoch)+"_num_gen"+str(args.num_generation)+'.json')
            
            results[task_name] = np.mean(batch_scores)
            avg_lens[task_name] = np.mean(batch_lengths)
            if batch_formatted:
                formatted[task_name] = np.mean(batch_formatted)
            max_lens[task_name] = np.max(batch_lengths)
            print(results)
            print("avg:", np.mean(list(results.values())))
            print("avg_lens:", avg_lens)
            print("max_lens:", max_lens)
            print("formatted:", formatted)
            avg_acc[task_name].append(results[task_name])
            
    print("#################################################")
    with open(args.output_path+"/"+'results.txt', 'w') as file:
        print("Task_name", "  repeat_num:", args.repeat_num, " num_generation:", args.num_generation)
        file.write(f"Task_name  repeat_num: {args.repeat_num}  num_generation: {args.num_generation}\n")
        for task_name in args.tasks:
            mean = np.mean(avg_acc[task_name])
            std = np.std(avg_acc[task_name], ddof=1)
            print("Task_name:",task_name, f"{mean:.3f}±{3*std:.3f}" , avg_acc[task_name])
            file.write(f"Task_name: {task_name} {mean:.3f}±{3*std:.3f} {avg_acc[task_name]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Math_Eval')
    parser.add_argument("--model_path", type=str, default=None, help="Attack Model name.")
    parser.add_argument("--verify_model_path", type=str, default=None, help="Attack Model name.")
    parser.add_argument("--output_path", type=str, default=None, help="Attack Model name.")
    parser.add_argument("--dataset_name", type=str, default=None, help="datasets name.")
    parser.add_argument("--tasks", type=str, nargs='+', choices=["aime24", "amc23", "math500", "minerva", "olympiad_bench", "cmath", 
                                                                 "gpqa-d", "mmlu_pro", "supergpqa", "aime24_32repeat", 
                                                                 "gsm8k", "humaneval", "mbppplus", "webinstrcut", "mmlu", 
                                                                 "humaneval_plus", "mbpp", "ifeval", "leetcode2k"], default=['aime'], help="Target Model name(s).")
    parser.add_argument("--template", type=str, default='qwen_math', help="template type")
    parser.add_argument("--temperature", type=float, default=0.6, help="Attack Model name.")
    parser.add_argument("--num_generation", type=int, default=1, help="Attack Model name.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Attack Model name.")
    parser.add_argument("--repeat_num", type=int, default=1, help="Attack Model name.")

    args = parser.parse_args()
    
    main(args)










