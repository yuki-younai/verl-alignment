
import re

import numpy as np
from Levenshtein import distance as lev
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

rdBase.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.*')
from collections import Counter
import os
os.system("pip install selfies")
import selfies as sf
from rdkit import Chem
from nltk.translate.bleu_score import corpus_bleu
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

def standardize_smi(smi: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

def is_valid_smiles(smi: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smi)
        return mol is not None
    except:
        return False


def exact_string_match(pred_smi: str, gt_smi: str) -> float:
    try:
        can_pred = Chem.MolToSmiles(Chem.MolFromSmiles(pred_smi), canonical=True)
        can_gt = Chem.MolToSmiles(Chem.MolFromSmiles(gt_smi), canonical=True)
        return 1.0 if can_pred == can_gt else 0.0
    except:
        return 0.0


def exact_structure_match(pred_smi: str, gt_smi: str) -> float:
    try:
        m1, m2 = Chem.MolFromSmiles(pred_smi), Chem.MolFromSmiles(gt_smi)
        return 1.0 if Chem.MolToInchi(m1) == Chem.MolToInchi(m2) else 0.0
    except:
        return 0.0

def compute_score_format_acc(solution_str: str, ground_truth: str) -> float:
    # pred selfies convert
    pred_selfies = extract_answer(solution_str)
    if pred_selfies is None:
        return False, False, pred_selfies
    try:
        pred_smi = sf.decoder(pred_selfies)
    except Exception as e:
        return False, False, pred_selfies
    if not is_valid_smiles(pred_smi):
        return False, False, pred_selfies
    pred_smi = standardize_smi(pred_smi)

    # groundtruth selfies convert
    ground_truth_smi = sf.decoder(ground_truth)
    ground_truth_smi = standardize_smi(ground_truth_smi)
    if pred_smi is None or ground_truth_smi is None:
        return False, False, pred_selfies
    exact_text = exact_string_match(pred_smi, ground_truth_smi)
    exact_struct = exact_structure_match(pred_smi, ground_truth_smi)
    if exact_struct == 1.0 or exact_text == 1.0:
        return True, True, pred_selfies
    else:
        return True, False, pred_selfies

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


def compute_score_bleu(solution_str: str, ground_truth: str) -> float:
    # pred selfies convert
    pred_selfies = extract_answer(solution_str)
    if pred_selfies is None:
        return False, False, pred_selfies, -1

    gt_tokens = clean_tokenize(ground_truth, text_tokenizer)
    out_tokens = clean_tokenize(pred_selfies, text_tokenizer)

    bleu = corpus_bleu([[gt_tokens]], [out_tokens])
    return True, True, pred_selfies, bleu

def compute_score_caption2mol(data_source, solution_str, ground_truth, extra_info=None, method='strict') -> float:
    is_extract, success , solution, score = compute_score_bleu(solution_str, ground_truth)
    if is_extract:
        formatted = 1
        model_answer = solution
        reward = score*2-1
    else:
        reward = -1
        formatted = 0
        model_answer = "None"

    return {"score": reward, "acc": success, "formatted": formatted, "pred": model_answer}

def compute_score_caption2mol_eval(data_source, solution_str, ground_truth, extra_info=None, method='strict') -> float:
    is_extract, success , solution, score = compute_score_bleu(solution_str, ground_truth)
    if is_extract:
        formatted = 1
        model_answer = solution
        reward = score
    else:
        reward = 0
        formatted = 0
        model_answer = solution

    return {"score": reward, "acc": success, "formatted": formatted, "pred": model_answer}


def compute_score_caption2mol_(data_source, solution_str, ground_truth, extra_info=None, method='strict') -> float:
    is_extract, success , solution = compute_score_format_acc(solution_str, ground_truth)
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
        model_answer = solution

    return {"score": reward, "acc": success, "formatted": formatted, "pred": model_answer}


def compute_score_caption2mol_eval_(data_source, solution_str, ground_truth, extra_info=None, method='strict') -> float:
    is_extract, success , solution = compute_score_format_acc(solution_str, ground_truth)
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
        model_answer = solution

    return {"score": reward, "acc": success, "formatted": formatted, "pred": model_answer}


if __name__ == "__main__":
    import selfies as sf

    ground_truth_selfies = '[C][O][C][=Branch1][C][=O][/C][=C][/C][C][C@H1][C@@H1][C][C][C][=C][C][=Branch1][C][=O][C][C][C@][Ring1][#Branch1][Branch1][C][C][C@H1][Ring1][N][C][=Branch1][C][=O][C][C@][Ring2][Ring1][Ring2][Ring1][P][C]' 

    solution_str_correct = """
    Let's reason step by step.
    <answer>[C][O][C][=Branch1][C][=O][/C][=C][/C][C][C@H1][C@@H1][C][C][C][=C][C][=Branch1][C][=O][C][C][C@][Ring1][#Branch1][Branch1][C][C][C@H1][Ring1][N][C][=Branch1][C][=O][C][C@][Ring2][Ring1][Ring2][Ring1][P][C]</answer>
    """


    score_correct = compute_score(solution_str_correct, ground_truth_selfies)
    print(f"Score: {score_correct:.2f}") 
