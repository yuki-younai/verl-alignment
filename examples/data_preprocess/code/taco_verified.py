"""Downloads, processes, and saves TACO dataset."""

import os
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import datasets
import transformers
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

def fuzzy_equal(actual: str, expected: str, tolerance: float = 1e-6, verbose=False) -> bool:
    """
    Compare two outputs line by line and element by element for approximate equality.
    Handles:
    1. Integer and floating-point number comparison with tolerance
    2. Case-insensitive comparison for yes/no

    Args:
        actual: The actual output from code execution
        expected: The expected output
        tolerance: Tolerance for floating point number comparison

    Returns:
        bool: True if outputs are approximately equal
    """
    # Save original values for debugging
    original_actual = actual
    original_expected = expected

    # Normalize line endings
    actual = actual.strip().replace("\r\n", "\n")
    expected = expected.strip().replace("\r\n", "\n")

    # If exact match after normalization, return early
    if actual == expected:
        return True

    # Split into lines
    actual_lines = actual.split("\n")
    expected_lines = expected.split("\n")

    # If different number of lines, they're definitely not equal
    if len(actual_lines) != len(expected_lines):
        return False

    # Track fuzzy matches for debugging
    fuzzy_match_reasons = []

    # Compare each line
    for i, (actual_line, expected_line) in enumerate(zip(actual_lines, expected_lines)):
        # If lines match exactly, continue
        if actual_line == expected_line:
            continue

        # Split into tokens by whitespace
        actual_tokens = actual_line.split()
        expected_tokens = expected_line.split()

        # If different number of tokens, they're not equal
        if len(actual_tokens) != len(expected_tokens):
            return False

        # Compare each token
        for j, (actual_token, expected_token) in enumerate(zip(actual_tokens, expected_tokens)):
            # If tokens match exactly, continue
            if actual_token == expected_token:
                continue

            # For yes/no, use case-insensitive comparison
            if actual_token.lower() in ["yes", "no"] and expected_token.lower() in ["yes", "no"]:
                if actual_token.lower() == expected_token.lower():
                    fuzzy_match_reasons.append(f"Line {i + 1}, Token {j + 1}: Case-insensitive yes/no match '{actual_token}' ≈ '{expected_token}'")
                    continue
                else:
                    return False

            # Try numeric comparison
            try:
                actual_num = float(actual_token)
                expected_num = float(expected_token)
                diff = abs(actual_num - expected_num)

                if diff <= tolerance:
                    fuzzy_match_reasons.append(f"Line {i + 1}, Token {j + 1}: Numeric match '{actual_token}' ≈ '{expected_token}' (diff: {diff})")
                    continue
                else:
                    return False
            except ValueError:
                # Not numeric values
                return False

    # Output fuzzy match information if any occurred
    if fuzzy_match_reasons and verbose:
        print("😅 FUZZY MATCH - Outputs approximately equal:")
        print(f"  Expected: {repr(original_expected)}")
        print(f"  Actual:   {repr(original_actual)}")
        print("  Reasons for fuzzy matching:")
        for reason in fuzzy_match_reasons:
            print(f"    • {reason}")

    # If we made it here, all lines are approximately equal
    return True


def remote_check_stdio(code, stdin, stdout):
    succ, output = code_exec(code=code, stdin=stdin)
    return succ, output, stdin, stdout

# Use higher parallelism with random container selection  
MAX_PARALLEL_PROCESSES = 64  # Using random containers so can handle more parallelism

EMPTY_EXAMPLE = {
    "data_source": None,
    "prompt": None,
    "apply_chat_template": False,
    "ability": None,
    "reward_model": None,
    "extra_info": None
}

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
def process_fn(example, idx):
    # Create a default "skip" response with all required fields
    oracle = json.loads(example["input_output"])
    source = example["source"]
    data_source = "taco_verified"
    # Skip poorly formatted examples
    if source in ["geeksforgeeks", "leetcode"]:
        return EMPTY_EXAMPLE

    # Skip examples with too short descriptions
    if len("".join([c for c in example["question"] if c.isalnum()])) < 100:
        return EMPTY_EXAMPLE

    # Skip examples with images
    if "image" in example["question"].lower() or "\n![" in example["question"]:
        return EMPTY_EXAMPLE

    # Build prompt
    prompt_pieces = [
        "Solve the programming task below in a Python markdown code block.",
        example["question"].strip(),
    ]
    if example["starter_code"].strip():
        prompt_pieces.append(
            "You will use the following starter code to write the solution to the problem and enclose your code within ```python delimiters."
        )
        prompt_pieces.append(
            f"```python\n{example['starter_code'].strip()}\n```"
        )
    else:
        prompt_pieces.append(
            "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within ```python delimiters."
        )

    # Process oracle based on format
    if "fn_name" in oracle:  # Function-based tests
        fn_name = oracle["fn_name"]
        if source == "leetcode":
            fn_name = "Solution()." + fn_name

        test_code = f"""\
_inputs = {oracle["inputs"]}
_outputs = {oracle["outputs"]}
import math
def _deep_eq(a, b, tol=1e-5):
if isinstance(a, float) or isinstance(b, float):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
if isinstance(a, (list, tuple)):
    if len(a) != len(b): return False
    return all(_deep_eq(x, y, tol) for x, y in zip(a, b))
return a == b

for i, o in zip(_inputs, _outputs):
"""

        if source in ["leetcode", "hackerrank"]:
            test_code += f"    assert _deep_eq({fn_name}(*i), o)"
        elif source == "codewars":
            test_code += f"    assert _deep_eq({fn_name}(*i), o[0])"
        else:
            print(f"Unknown source: {source}")
            return EMPTY_EXAMPLE

        # Verify the solution passes tests
        _check_test = example["solutions"][-1] + "\n" + test_code
        succ, err = code_exec(_check_test)
        if not succ:
            print(idx, False)
            print(f"Test code failed for {source}")
            return EMPTY_EXAMPLE
        
        oracle_json = json.dumps({"functional": test_code})
        
    elif "inputs" in oracle and "outputs" in oracle:  # STDIN/STDOUT tests
        stdin_list, stdout_list = oracle["inputs"], oracle["outputs"]
        if len(stdin_list) == 0:
            print(idx, False)
            return EMPTY_EXAMPLE
        
        # handle list inputs and normalize line endings
        stdin_list = [
            "\n".join(stdin) if isinstance(stdin, list) else stdin 
            for stdin in stdin_list
        ]
        stdout_list = [
            ("\n".join(stdout) if isinstance(stdout, list) else stdout).replace("\r\n", "\n")
            for stdout in stdout_list
        ]

        # Verify the solution passes tests
        with ThreadPoolExecutor(max_workers=min(len(stdin_list), 8)) as executor:
            futures = []
            for stdin, stdout in zip(stdin_list, stdout_list):
                futures.append(
                    executor.submit(
                        remote_check_stdio,
                        example["solutions"][-1],
                        stdin,
                        stdout,
                    )
                )
            for future in as_completed(futures):
                exec_succ, output, stdin, stdout = future.result()
                pass_test = exec_succ and fuzzy_equal(output.strip(), stdout.strip())
                if not pass_test:
                    print(idx, False)
                    print(f"Test code failed for {source}")
                    return EMPTY_EXAMPLE

        oracle_json = json.dumps({"inputs": stdin_list, "outputs": stdout_list})
    else:
        print(f"Unknown ground truth format: {oracle}")
        return EMPTY_EXAMPLE

    # Format the final prompt
    prompt = "\n".join(prompt_pieces)
    print(idx, True)
    data = {
        "data_source": "code",
        "prompt": [
            {"role": "user", "content": prompt}
        ],
        "ability": "codegen",
        "reward_model": {
            "style": "rule",
            "ground_truth": oracle_json,
        },
        "extra_info": {
            "split": "train",
            "index": idx,
            "reference": example["solutions"][0] if example["solutions"] else "",
            "original_prompt": prompt,
            "prefix": None,
        },
    }
    
    if idx == 0 or idx == 1:
        print("\n" + "=" * 10 + f"{idx}" + "=" * 10)
        print(data)
        
    return data

# 定义过滤函数
def filter_fn(example, idx):
    data_source = example["data_source"]
    if not data_source:
        return False
    
    return True

if __name__ == '__main__':

    import time
    start_time = time.time()
    #https://huggingface.co/datasets/likaixin/TACO-verified
    dataset = load_dataset("/mnt/public/gpfs-jd/data/RL_Data/origin/TACO-verified", split="train")
    dataset = dataset.train_test_split(test_size=0.5, seed=42)['test']
    # Use reduced number of processes
    dataset = dataset.map(function=process_fn, with_indices=True, num_proc=64, remove_columns=dataset.column_names)
    #Filter Code data
    dataset = dataset.filter(function=filter_fn, with_indices=True, num_proc=64)
    
    MAX_PARALLEL_PROCESSES=16
    # Length filter
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("/mnt/public/gpfs-jd/model/Qwen/Official/Qwen2_5/Qwen2.5-7B")
        length_filter = LengthFilter(tokenizer=tokenizer, max_length=2048)
        dataset = dataset.filter(lambda x: length_filter.check(x), num_proc=MAX_PARALLEL_PROCESSES)
    except Exception as e:
        print(f"Warning: Could not perform length filtering. Error: {e}")
        print("Proceeding without length filtering.")
    # Save
    train_test_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = train_test_dataset['train']
    test_data =  train_test_dataset['test']
    
    print(train_data)
    print(test_data)
    print(test_data[0])
    local_dir = "/mnt/public/gpfs-jd/data/RL_Data/RL_Train/code/taco_verified"
    train_data.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_data.to_parquet(os.path.join(local_dir, "test.parquet"))
    test_data.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/taco_verify")
    
    
    
    