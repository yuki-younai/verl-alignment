from examples.reward_function.math_reward import compute_score_math
from examples.reward_function.coder1_reward import compute_score_code
from examples.reward_function.caption2mol_reward import compute_score_caption2mol
from examples.reward_function.mol2caption_reward import compute_score_mol2caption
from examples.reward_function.stem_reward import compute_score_stem


def compute_score_science(data_source, solution_str, ground_truth, extra_info=None, method='strict'):

    if data_source in ["math", "science_chemstry", "science_physics", "math_dapo", "BytedTsinghua-SIA/AIME-2024"]:
        return compute_score_math(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info)
    elif data_source in ["leetcode2k", "codegen", "code"]:
        return compute_score_code(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info)
    elif data_source in ["stem"]:
        return compute_score_stem(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info)
    elif data_source in ["science_caption2mol"]:
        return compute_score_caption2mol(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info)
    elif data_source in ["science_mol2caption"]:
        return compute_score_mol2caption(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info)
    else:
        raise ValueError(f"!!!!!未找到与 '{data_source}' 匹配的奖励函数!!!!!")
    
























































