import os
import re
import numpy as np
from collections import Counter
try:
    import rouge_score
except ImportError:
    os.system('pip install rouge-score')
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
rdBase.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.*')
from transformers import BertTokenizerFast
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from examples.reward_function.math_reward import extract_boxed_answer
from tqdm import tqdm
import argparse
import csv
from transformers import BertTokenizerFast
# 需要下载nltk里面用到bleu的 nltk.download()
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction



def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    
    return None

def extract_solution(solution_str: str) -> str:
    """
    Extract the last <answer>...</answer> block from the solution string.
    """
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    return matches[-1].group(1).strip() if matches else None

def compute_score_format_acc(solution_str: str, ground_truth: str) -> float:
    pred_text = extract_answer(solution_str)
    
    if pred_text is None or not ground_truth:
        return False, False, pred_text
    if pred_text.strip().lower() == ground_truth.strip().lower():
        return True, True, pred_text
    else:
        return True, False, pred_text

text_model='/mnt/public/guojianz/ckpt/llm_ckpt/scibert_scivocab_uncased'
text_tokenizer = BertTokenizerFast.from_pretrained(text_model)

def clean_tokenize(text, tokenizer):
    encoded = tokenizer.encode_plus(
        text,
        truncation=True,
        max_length=1024,
        padding='max_length',
        return_tensors=None,
        return_special_tokens_mask=True,
        return_attention_mask=False
    )
    return [tok for tok, mask in zip(encoded.tokens(), encoded["special_tokens_mask"]) if not mask]

def compute_bleu_score(solution_str: str, ground_truth: str) -> float:
    pred_text = extract_answer(solution_str)

    if pred_text is None or not ground_truth:
        return False, False, pred_text, 0
    
    gt_tokens = clean_tokenize(ground_truth, text_tokenizer)
    out_tokens = clean_tokenize(pred_text, text_tokenizer)
    smoother = SmoothingFunction().method3
    bleu2 = corpus_bleu([[gt_tokens]], [out_tokens], 
                        weights=(.5, .5),
                        smoothing_function=smoother)
    return True, True, pred_text, bleu2
    


def compute_score_mol2caption(data_source, solution_str, ground_truth, extra_info=None, method='strict') -> float:
    is_extract, success , solution, score = compute_bleu_score(solution_str, ground_truth)
    if is_extract:
        formatted = 1
        model_answer = solution
        reward = score*2-1
    else:
        reward = -1
        formatted = 0
        model_answer = "None"

    return {"score": reward, "acc": success, "formatted": formatted, "pred": model_answer}

def compute_score_mol2caption_eval(data_source, solution_str, ground_truth, extra_info=None, method='strict') -> float:
    is_extract, success , solution, score = compute_bleu_score(solution_str, ground_truth)
    if is_extract:
        formatted = 1
        model_answer = solution
        reward = score
    else:
        reward = 0
        formatted = 0
        model_answer = "None"

    return {"score": reward, "acc": success, "formatted": formatted, "pred": model_answer}

def compute_score_mol2caption_(data_source, solution_str, ground_truth, extra_info=None, method='strict') -> float:
    is_extract, success , solution = compute_bleu_score(solution_str, ground_truth)
    if is_extract:
        formatted = 1
        model_answer = solution
        if success:
            reward = 1
        else:
            reward = -0.5
    else:
        reward = -1
        formatted = 0
        model_answer = "None"

    return {"score": reward, "acc": success, "formatted": formatted, "pred": model_answer}

def compute_score_mol2caption_eval_(data_source, solution_str, ground_truth, extra_info=None, method='strict') -> float:
    is_extract, success , solution = compute_bleu_score(solution_str, ground_truth)
    if is_extract:
        formatted = 1
        model_answer = solution
        if success:
            reward = 1
        else:
            reward = 0
    else:
        reward = 0
        formatted = 0
        model_answer = "None"

    return {"score": reward, "acc": success, "formatted": formatted, "pred": model_answer}


if __name__ == "__main__":
    solution_str = """
    Let's reason step by step.
    The correct result is clearly described below.
    <answer>The capital of France is B</answer>
    """
    ground_truth = "The capital of France is Paris"
    
    score = compute_score(solution_str, ground_truth)
    
    print(f"score: {score:.4f}")
