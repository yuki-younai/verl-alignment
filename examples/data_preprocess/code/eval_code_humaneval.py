"""Downloads, processes, and saves HumanEval dataset."""

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
    # Extract task ID and prompt
    task_id = example["task_id"]

    prompt = (
        "Write a complete, self-contained Python solution to the following problem. "
        "Your solution must include all necessary imports and the full function definition including "
        "the signature exactly as specified. Do not modify the function signature or docstring.\n\n"
        f"```python\n{example['prompt'].strip()}\n```"
    )
    
    # Extract test function, entry point, and canonical solution
    test_code = example["test"]
    entry_point = example["entry_point"]
    solution = example["canonical_solution"]
    
    # Build test code that calls the entry point
    test_code_with_check = f"{test_code}\n\ncheck({entry_point})"
    
    # Verify the canonical solution passes the tests
    # full_code = f"{prompt}\n{solution}\n{test_code}\n\ncheck({entry_point})"
    full_code = f"{example['prompt']}\n{solution}\n{test_code}\n\ncheck({entry_point})"

    data = {
        "data_source": "openai/openai_humaneval",
        "prompt": [
            {"role": "user", "content": prompt}
        ],
        "ability": "codegen",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps({
                "functional": test_code_with_check
            }),
        },
        "extra_info": {
            "split": "train",
            "index": idx,
            "reference": solution,  # Include the canonical solution as reference
            "original_prompt": prompt,
            "dataset": "openai_humaneval",
            "task_id": task_id,
        },
    }
    
    if idx == 0 or idx == 1:
        print("\n" + "=" * 10 + f" {idx}" + "=" * 10)
        print(data)
        
    return data


if __name__ == '__main__':

    #https://huggingface.co/datasets/openai/openai_humaneval
    dataset = load_dataset("/mnt/public/gpfs-jd/data/RL_Data/origin/Evaluation_data/code/humaneval")["test"]
    # Process the dataset
    dataset = dataset.map(function=process_fn, with_indices=True, num_proc=16, remove_columns=dataset.column_names)

    dataset.save_to_disk("/mnt/public/gpfs-jd/data/RL_Data/eval_data/humaneval")

