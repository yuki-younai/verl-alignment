from examples.reward_function.math_reward import extract_answer, grade
import re



def extract_first_number(s):
    try:
        # 使用正则表达式匹配第一个数字（考虑负号）
        match = re.search(r'-?\d+', s)
        # 如果找到匹配项，转换为整数
        if match:
            return int(match.group())
        else:
            return s  # 如果没有找到数字，返回None
    except Exception as e:
        # 打印错误信息或进行其他错误处理
        print(f"An error occurred: {e}")
        return s


def is_subset_ignore_case(str1, str2):
    # 将两个字符串都转换为小写，然后转换为集合
    set1 = set(str1.lower())
    set2 = set(str2.lower())
    # 检查set1是否是set2的子集
    return set1.issubset(set2)


def compute_score_stem_eval(data_source, solution_str, ground_truth, extra_info=None, method='strict'):
    
    model_answer = extract_answer(solution_str)
    fast = True
    if model_answer is None:
        box_match = 0
        formatted = 0
        model_answer = "None"
        is_correct = False
    else:
        if isinstance(ground_truth, float) or isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        if isinstance(ground_truth, str):
            is_correct = grade(model_answer, ground_truth, fast)
            if extra_info["prefix"]=="Boolean":
                is_correct |= is_subset_ignore_case(model_answer, ground_truth)
            elif extra_info["prefix"]=="Integer":
                ground_truth_num = extract_first_number(ground_truth)
                model_answer_num = extract_first_number(model_answer)
                is_correct |= grade(str(model_answer_num), str(ground_truth_num), fast)
        elif isinstance(ground_truth, list):
            is_correct = False
            for truth in ground_truth:
                is_correct |= grade(model_answer, truth, fast)
                if extra_info["prefix"]=="Boolean":
                    is_correct |= is_subset_ignore_case(model_answer, ground_truth)
                elif extra_info["prefix"]=="Integer":
                    truth_num = extract_first_number(truth)
                    model_answer_num = extract_first_number(model_answer)
                    is_correct |= grade(str(model_answer_num), str(truth_num), fast)
        if is_correct:
            box_match = 1
        else:
            box_match = 0
        formatted = 1
    
    return {"score": box_match, "acc": is_correct,"pred": model_answer, "formatted": formatted}

def compute_score_stem(data_source, solution_str, ground_truth, extra_info=None, method='strict'):
    
    model_answer = extract_answer(solution_str)
    fast = True
    if model_answer is None:
        box_match = -1
        formatted = 0
        model_answer = "None"
        is_correct = False
    else:
        if isinstance(ground_truth, float) or isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        if isinstance(ground_truth, str):
            is_correct = grade(model_answer, ground_truth, fast)
            if extra_info["prefix"]=="Boolean":
                is_correct |= is_subset_ignore_case(model_answer, ground_truth)
            elif extra_info["prefix"]=="Integer":
                ground_truth_num = extract_first_number(ground_truth)
                model_answer_num = extract_first_number(model_answer)
                is_correct |= grade(str(model_answer_num), str(ground_truth_num), fast)
        elif isinstance(ground_truth, list):
            is_correct = False
            for truth in ground_truth:
                is_correct |= grade(model_answer, truth, fast)
                if extra_info["prefix"]=="Boolean":
                    is_correct |= is_subset_ignore_case(model_answer, ground_truth)
                elif extra_info["prefix"]=="Integer":
                    truth_num = extract_first_number(truth)
                    model_answer_num = extract_first_number(model_answer)
                    is_correct |= grade(str(model_answer_num), str(truth_num), fast)
        if is_correct:
            box_match = 1
        else:
            box_match = -0.5
        formatted = 1
    
    return {"score": box_match, "acc": is_correct,"pred": model_answer, "formatted": formatted}


















