import numpy as np
from examples.reward_function.ifeval import instructions_registry



def compute_score_ifeval_eval(data_source, solution_str, ground_truth, extra_info=None, method='strict'):
    """The scoring function for IFEval.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    if "</think>" in solution_str:
        answer = solution_str.split("</think>")[1]
    else:
        answer = solution_str
    is_following_list = []
    for index, instruction_id in enumerate(extra_info["instruction_id_list"]):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {
            k: int(v) if isinstance(v, float) else v.tolist() if isinstance(v, np.ndarray) else v for k, v in ground_truth[index].items() if v is not None
        }

        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=extra_info["prompt"])

        if answer.strip() and instruction.check_following(answer):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return {
        "score": all(is_following_list),
        "acc": all(is_following_list), "pred": answer, "formatted": 1
    }
