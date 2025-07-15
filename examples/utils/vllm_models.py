import torch
import sys
import time
import random
import argparse
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

class VLLM_models:
    def __init__(self, model_name_or_path, device=0, gpu_memory_utilization=0.4, tensor_parallel_size=1):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tensor_parallel_size==1:
            self.llm = LLM(
                model=model_name_or_path,
                device = device,
                gpu_memory_utilization= gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                enable_prefix_caching=True,
                dtype="bfloat16",
            )
        else:
            self.llm = LLM(
                model=model_name_or_path,
                gpu_memory_utilization= gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                enable_prefix_caching=True,
                dtype="bfloat16",
            )
        self.llm.sleep(2)
        torch.cuda.empty_cache()

    def generate(self, messages, apply_chat_template=None, max_model_len=8196, temperature=0, top_p=1.0, num_generation=1):
        torch.cuda.empty_cache()
        self.llm.wake_up()
        self.sampling_params = SamplingParams(
            n=num_generation,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_model_len,
            seed=int(time.time_ns()),
        )
        if apply_chat_template==None:
            prompts = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompts = list(map(apply_chat_template, messages))
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params, use_tqdm=True)
        self.llm.sleep(2)
        torch.cuda.empty_cache()
        return outputs