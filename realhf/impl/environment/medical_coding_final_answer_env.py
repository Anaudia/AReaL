# Copyright 2025 Ant Group Inc.

import asyncio
import os
import re
from typing import List, Tuple

from functioncall.code.local_verify import code_verify as local_code_verify
from functioncall.code.verify import code_verify
from functioncall.math.verify import math_verify
from realhf.api.core.env_api import EnvironmentService, register_environment
from realhf.base import logging
from realhf.impl.dataset.math_code_dataset import load_metadata
from realhf.impl.dataset.math_parser import parse_lines_in_parallel

ENABLE_FUNCTION_CALL = True if os.getenv("FUNCTIONCALL_SERVICE_DOMAIN", "") else False
math_verify_call = math_verify if ENABLE_FUNCTION_CALL else parse_lines_in_parallel
code_verify_call = code_verify if ENABLE_FUNCTION_CALL else local_code_verify

logger = logging.getLogger("Math Single Step Environment")


# verifier_icd.py
import re
from typing import List, Dict, Tuple, Set

boxed_pattern = re.compile(r"\[(.*?)\]")
code_pattern  = re.compile(r"[A-Z][0-9]{2}(?:\.[0-9A-Za-z]+)?")

def _extract_boxed_codes(text: str) -> Tuple[str, List[str]]:
    """
    Original helper.  On any regex failure returns ("", []) so the
    downstream logic still works and produces a score of 0.0
    """
    m = boxed_pattern.search(text or "")
    if not m:
        return "", []
    inside = m.group(1)
    codes  = code_pattern.findall(inside)
    uniq, seen = [], set()
    for c in codes:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return inside, uniq


def _f1(pred: Set[str], gold: Set[str]) -> float:
    if not pred and not gold:
        return 1.0          # vacuously correct
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall    = tp / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def icd_verify(
    id2info: Dict[str, Dict],
    answers: List[str],
    qids: List[str],
) -> List[float]:
    """
    Safe verifier.
    • Returns a float in [0, 1] (the F-score) for each answer.
    • On *any* exception, or if no codes are boxed, returns 0.0.
    • Accesses id2info the same way as parse_lines_in_parallel:
        qid_base = qid.split("@idx:")[0]
    """
    scores: List[float] = []
    for ans, qid in zip(answers, qids):
        try:
            qid_base   = qid.split("@idx:")[0]
            gold_codes = set(id2info[qid_base]["solutions"])
            _, pred    = _extract_boxed_codes(ans or "")
            f1         = _f1(set(pred), gold_codes)
            scores.append(float(f1))
        except Exception:
            scores.append(0.0)          # fallback guarantees float() works
    return scores



def extract_code(text, min_length=20):
    code_pattern = r"(?i)```(?:python|py|cpp|CPP)?\s*\n?(.*?)\n?```"
    code_blocks = re.findall(code_pattern, text, re.DOTALL)
    valid_blocks = []
    for block in code_blocks:
        clean_block = block.strip()
        if len(clean_block) < min_length:
            continue

        valid_blocks.append(clean_block)

    if not valid_blocks:
        # logger.warning(f"failed to extract python code from {text}")
        return None
    # return the last code block
    return valid_blocks[-1]


class MathCodeSingleStepEnv(EnvironmentService):
    def __init__(self, dataset_path: str):
        self.id2info, _ = load_metadata(dataset_path)

    async def reset(self, seed=None, options=None):
        return None, {}

    async def step(self, action: Tuple[str, List[str]]):
        qid, answers = action
        group_size = len(answers)
        qid = qid.split("@")[0]
        cur_task = self.id2info[qid]["task"]

        if cur_task == "math":
            format_rewards = await asyncio.to_thread(
                icd_verify, self.id2info, answers, [qid for _ in range(group_size)]
            )
            logger.debug(f"MATH task (qid: {qid}): Answers {answers} with these icd_verify results: {format_rewards}")
            # format_rewards = await asyncio.to_thread(
            #     math_verify_call,
            #     self.id2info,
            #     answers,
            #     [qid for _ in range(group_size)],
            # )
        elif cur_task == "code":
            answers = [extract_code(x) for x in answers]
            format_rewards = await asyncio.to_thread(
                code_verify_call,
                self.id2info,
                answers,
                [qid for _ in range(group_size)],
            )
        else:
            raise NotImplementedError()

        return None, format_rewards, True, False, {}


register_environment("medical-coding-final-answer-env", MathCodeSingleStepEnv)