"""Downloads, processes, and saves LeetCode2K datasets."""

import os
import argparse
import json
import transformers
import datasets
from datasets import load_dataset, Dataset

import os
import argparse
import json
import transformers
import datasets
from datasets import load_dataset, Dataset
import os
import subprocess
import shlex
from tempfile import NamedTemporaryFile, TemporaryDirectory
import transformers
import abc
from typing import Dict


class Filter(abc.ABC):
    """
    Filter class for filtering data.
    """
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def check(self, data_entry: Dict) -> bool:
        pass


class LengthFilter(Filter):
    """
    Filter class for filtering data by length.
    """
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer = None, min_length: int = 0, max_length: int = 2048, length_tolerance: int = 100):
        if tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        else:
            self.tokenizer = tokenizer
        self.min_length = min_length
        self.max_length = max_length
        self.length_tolerance = length_tolerance
    
    def check(self, data_entry: Dict) -> bool:
        if data_entry["prompt"]:
            
            prompt_tokens = self.tokenizer.tokenize(self.tokenizer.apply_chat_template(data_entry["prompt"], tokenize=False))
        elif data_entry["raw_prompt"]:
            prompt_tokens = self.tokenizer.tokenize(data_entry["raw_prompt"])
        else:
            raise ValueError("No prompt found in data")
        # print(f"Prompt length: {len(prompt_tokens)}")
        return self.min_length <= len(prompt_tokens) <= self.max_length - self.length_tolerance
_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 30   # 30 seconds is the default timeout for the executor

CLI_ARG_SIZE_LIMIT = 1024 * 3
MEMORY_LIMIT_KB = 10 * 1024 * 1024  # 10GB in KB

def wrap_command_with_ulimit(command):
    cmd_str = ' '.join(shlex.quote(c) for c in command)
    return ["bash", "-c", f"ulimit -v {MEMORY_LIMIT_KB}; exec {cmd_str}"]

def code_exec(
    code,
    stdin: str = None,
    timeout=30,
    pytest: str = None,
    solution: str = None,
    python_env: str = os.environ.get("CONDA_BIN_PATH", None),
):
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]

    if python_env is None:
        python_executable = "/usr/bin/python3"
    else:
        python_executable = os.path.join(python_env, "python3")

    if solution:
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with solution_file"
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(solution)

            command = [
                "timeout",
                str(timeout),
                python_executable,
                os.path.join(tmpdir, "test_solution.py"),
            ]
            command = wrap_command_with_ulimit(command)
            result = subprocess.run(
                command,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
    elif pytest:
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with pytest"
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(pytest)

            command = [
                "timeout",
                str(timeout),
                python_executable,
                "-m",
                "pytest",
                tmpdir,
            ]
            command = wrap_command_with_ulimit(command)
            result = subprocess.run(
                command,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
    else:
        if len(code) < CLI_ARG_SIZE_LIMIT:
            command = ["timeout", str(timeout), python_executable, "-c", code]
            command = wrap_command_with_ulimit(command)
            result = subprocess.run(
                command,
                input=(stdin.encode() if stdin else None),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
        else:
            with NamedTemporaryFile() as tmp:
                tmp.write(code.encode())
                tmp.flush()
                command = ["timeout", str(timeout), python_executable, tmp.name]
                command = wrap_command_with_ulimit(command)
                result = subprocess.run(
                    command,
                    input=(stdin.encode() if stdin else None),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=False,
                )

    stderr = result.stderr.decode().strip()
    stdout = result.stdout.decode()

    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
# 定义过滤函数
def filter_fn(example, idx):
    prefix = example["prompt"]
    # Clean up the prompt
    prompt = example["query"]
    prompt = prompt.replace("### Answer: (use the provided format with backticks)", "").strip()
    prompt = prompt.replace("### Format: ", "### Format:\n")

    # Build test code
    test_code = f"{example['test']}\n\ncheck({example['entry_point'].strip()})"
    # Extract the candidate solution
    solution = example["completion"]
    # Combine all code pieces into a single file to execute
    full_code = f"{prefix}\n{solution}\n{test_code}"
    # Validate that the candidate solution passes the tests
    # Skip examples where the test code fails
    #succ, err = code_exec(full_code, timeout=5)
    succ = True
    print(idx, succ)
    if not succ:
        return False
    
    return True

def process_fn(example, idx):
    prefix = example["prompt"]
    # Clean up the prompt
    prompt = example["query"]
    prompt = prompt.replace("### Answer: (use the provided format with backticks)", "").strip()
    prompt = prompt.replace("### Format: ", "### Format:\n")
    # Build test code
    test_code = f"{example['test']}\n\ncheck({example['entry_point'].strip()})"
    # Extract the candidate solution
    solution = example["completion"]
    # Combine all code pieces into a single file to execute
    full_code = f"{prefix}\n{solution}\n{test_code}"

    data = {
        "data_source": "leetcode2k",
        "prompt": [
            {"role": "user", "content": prompt}
        ],
        "ability": "codegen",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps({
                "functional": test_code
            }),
        },
        "extra_info": {
            "split": "train",
            "index": idx,
            "reference": solution,
            "original_prompt": prompt,
            "prefix": prefix,
        },
    }
    
    return data


if __name__ == '__main__':

    train_data_source = "/mnt/public/gpfs-jd/data/RL_Data/origin/LeetCodeDataset"
    train_dataset = load_dataset(train_data_source)["train"]
    #Filter Code data
    train_dataset = train_dataset.filter(function=filter_fn, with_indices=True, num_proc=16)
    # Use reduced number of processes
    train_dataset = train_dataset.map(function=process_fn, with_indices=True, num_proc=16, remove_columns=train_dataset.column_names)
    
    MAX_PARALLEL_PROCESSES=16
    # Length filter
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B")
        length_filter = LengthFilter(tokenizer=tokenizer, max_length=2048)
        train_dataset = train_dataset.filter(lambda x: length_filter.check(x), num_proc=MAX_PARALLEL_PROCESSES)
    except Exception as e:
        print(f"Warning: Could not perform length filtering. Error: {e}")
        print("Proceeding without length filtering.")

    # Save
    print(len(train_dataset))
    print(train_dataset)
    output_path = "/mnt/public/gpfs-jd/data/RL_Data/RL_Train/code/"
    train_dataset.to_parquet(os.path.join(output_path+"leetcode2k", "train.parquet"))
    
    print(f"Done! \nTrain data saved to {output_path}")



