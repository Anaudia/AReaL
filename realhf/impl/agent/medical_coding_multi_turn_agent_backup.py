# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
import json
import re

def load_guideline(index: str,
                   root_dir: str = '/home/scadmin/SC_project/ai-coding-reasoning/flat_kodierhandbuch_machine') -> str:
    """
    Look for a file named {index}.txt (or starting with {index}) in root_dir,
    and return its text contents. Raises FileNotFoundError if nothing matches.
    """
    # 1) try exact match
    fname = f"{index}.txt"
    path = os.path.join(root_dir, fname)
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            return f.read()
    # 2) fallback: any file whose name starts with the index
    for fn in os.listdir(root_dir):
        if fn.startswith(index) and fn.lower().endswith('.txt'):
            with open(os.path.join(root_dir, fn), encoding='utf-8') as f:
                return f.read()
    raise FileNotFoundError(f"No guideline file found for index '{index}' in {root_dir}")



try:
    with open('/home/scadmin/SC_project/ai-coding-reasoning/reasoning_generate_sft_data/code_range_mapping.json', 'r', encoding='utf-8') as f:
        code_range_mapping = json.load(f)
except FileNotFoundError:
    print("Warning: 'code_range_mapping.json' not found. `retrieve_icd_code_chapter` will fail.")
    code_range_mapping = {}

try:
    # This variable name must match the one used in the function below
    with open("/home/scadmin/SC_project/ai-coding-reasoning/sorted_combined_dict_translated_v2.json", "r", encoding="utf-8") as json_file:
        sorted_combined_dict_translated_v2 = json.load(json_file)
except FileNotFoundError:
    print("Warning: 'sorted_combined_dict_translated_v2.json' not found. `retrieve_icd_code_chapter` will fail.")
    sorted_combined_dict_translated_v2 = {}


try:
    with open('/home/scadmin/SC_project/ai-coding-reasoning/reasoning_generate_sft_data/coding_documentation/json_export/coding_groups_data.json', 'r', encoding='utf-8') as file:
        coding_groups_data = json.load(file)
except FileNotFoundError:
    print("CRITICAL_ERROR_STARTUP: 'coding_groups_data.json' not found. `retrieve_icd_code_chapter` validation will fail.")
    coding_groups_data = {}
    

def retrieve_icd_code_chapter(subchapter: str) -> tuple[str, set[str] | None]:
    """
    Retrieves ICD-10-GM code details for a given subchapter index and validates data.

    Args:
        subchapter: The index number, e.g., "1.1" or "19.9".

    Returns:
        A tuple:
        - On success: (string_content_for_llm, set_of_3_digit_base_codes_in_this_subchapter)
        - On failure/data inconsistency: (error_message_string_for_llm, None)
    """
    if not subchapter or not isinstance(subchapter, str):
        error_msg = "Error: Invalid or missing 'subchapter' argument. Please provide a string like '1.1'."
        # print(f"TOOL_ERROR (retrieve_icd_code_chapter): {error_msg}") # Optional: internal server log
        return error_msg, None

    match = re.search(r"\b\d{1,2}\.\d{1,2}\b", subchapter)
    if not match:
        error_msg = f"Error: Could not find a valid subchapter format (e.g., '1.1') in your input '{subchapter}'. Please use a valid subchapter index."
        # print(f"TOOL_ERROR (retrieve_icd_code_chapter): Invalid format for input '{subchapter}'") # Optional: internal server log
        return error_msg, None
    
    chapter_key_from_input = match.group(0) # e.g., "1.1"

    # 1. Validate against code_range_mapping
    actual_code_range_str = code_range_mapping.get(chapter_key_from_input)
    if not actual_code_range_str:
        error_msg = (f"Error: Subchapter index '{chapter_key_from_input}' provided does not exist or is not recognized. "
                     f"Please ensure you are using a valid subchapter index from the provided list.")
        # print(f"TOOL_ERROR_DATA (retrieve_icd_code_chapter): Subchapter '{chapter_key_from_input}' not in code_range_mapping.") # Optional
        return error_msg, None

    # 2. Validate against coding_groups_data using the mapped range
    group_data_for_range = coding_groups_data.get(actual_code_range_str)

    codes_list_from_group = group_data_for_range["codes"]

    # 3. Retrieve content from sorted_combined_dict_translated_v2
    # The data key for sorted_combined_dict_translated_v2 is lowercase
    dict_key_for_content = actual_code_range_str.lower()
    chapter_content_text = sorted_combined_dict_translated_v2.get(dict_key_for_content)

    # All checks passed, prepare successful return
    base_codes_in_subchapter = {code[:3] for code in codes_list_from_group}

    return chapter_content_text, base_codes_in_subchapter


def retrieve_icd_guideline_chapter(guideline_id: str) -> str:
    """
    Retrieves the content of a specific ICD-10-GM coding guideline.

    Args:
        guideline_id: The guideline identifier, e.g., "SD0207a".

    Returns:
        The text content of the guideline, or an error message.
    """
    if not guideline_id or not isinstance(guideline_id, str):
        return "Error: Invalid or missing 'guideline_id' argument. Please provide a string like 'SD0207a'."
        
    try:
        # The load_guideline function is now imported from utils
        guideline_content = load_guideline(guideline_id)
        return guideline_content
    except FileNotFoundError:
        return f"Error: Guideline with ID '{guideline_id}' was not found."
    except Exception as e:
        return f"An unexpected error occurred in retrieve_icd_guideline_chapter: {str(e)}"
    


def parse_assistant_response_with_tool_calls(llm_response: str) -> tuple[str, dict, str]:
    """
    Parses the full LLM response, separating the think block, text content, and tool calls.

    Returns a tuple:
    1. The original llm_response string (unchanged input).
    2. A dictionary formatted as an assistant message for the history (without the think block).
       This message contains 'role', 'content', and optionally 'tool_calls'.
    3. A string containing the content of the <think> block (or an empty string if no think block).
    
    Note: The return signature (str, dict, str) matches the implementation provided in the prompt.
    """
    # 1. Extract the <think> block first
    think_content = ""
    content_after_think = llm_response
    
    think_match = re.search(r"<think>(.*?)</think>", llm_response, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        content_after_think = llm_response[think_match.end():].lstrip()

    # 2. Now, parse the rest of the content (text and tool calls)
    tool_calls = []
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    matches = list(pattern.finditer(content_after_think))
    
    if not matches:
        # No tool calls, the entire remaining response is content
        # text_content = re.sub(r"<\|im_end\|>$", "", content_after_think).strip()
        assistant_message = {"role": "assistant", "content": content_after_think}
    else:
        # <tool_call> blocks are present.
        # Text content is what appears *before* the first <tool_call> block.
        first_match_start = matches[0].start()
        current_text_content = content_after_think[:first_match_start].strip()
        
        # Iterate through all found <tool_call> blocks
        for match in matches:
            json_str = match.group(1).strip()
            
            fallback_call = {
                "name": "unknown_function",
                "arguments": {
                    "subchapter": "unknown_function" 
                }
            }

            try:
                # Attempt to parse the JSON string from the tool call block
                parsed_json = json.loads(json_str)

                # Validate the structure: must be a dict with 'name' (str) and 'arguments' (dict)
                if (isinstance(parsed_json, dict) and
                        "name" in parsed_json and isinstance(parsed_json["name"], str) and
                        "arguments" in parsed_json and isinstance(parsed_json["arguments"], dict)):
                    # Parsed JSON has the expected structure for a tool call
                    tool_calls.append(parsed_json)
                else:
                    # JSON was valid, but its structure doesn't match a tool call's requirements
                    # (e.g., not a dictionary, 'name' or 'arguments' missing or wrong type).
                    reason = "Malformed tool call structure (e.g., not a dict, or missing/invalid 'name' or 'arguments' keys/types)."
                    print(f"Warning: {reason} '") # Raw content: '{json_str}
                    
                    # Add diagnostic information to the fallback call's arguments
                    fallback_call["arguments"]["subchapter"] = "No Content"
                    tool_calls.append(fallback_call)

            except json.JSONDecodeError as e:
                # Failed to parse the string as JSON.
                reason = f"JSONDecodeError: {str(e)}"
                print(f"Warning: Failed to parse JSON from tool call block: {e}. ")  #Raw content: '{json_str}'
                
                # Add diagnostic information to the fallback call's arguments
                fallback_call["arguments"]["subchapter"] =  "No Content"
                tool_calls.append(fallback_call)

        # Construct the assistant message with the extracted text content
        assistant_message = {"role": "assistant", "content": current_text_content}
        # If tool_calls list has items (it will, if 'matches' was not empty), add it to the message.
        if tool_calls: 
            assistant_message["tool_calls"] = tool_calls
 
    return llm_response, assistant_message, think_content



# ➊ (Put this near the top of the file once)
def _safe_tool_call(name: str, args: dict[str, str]) -> str:
    """Execute a tool and always return a string for the LLM."""
    if name == "retrieve_icd_code_chapter":
        subchapter = args.get("subchapter")
        txt, _ = retrieve_icd_code_chapter(subchapter)
        return txt
    elif name == "retrieve_icd_guideline_chapter":
        gid = args.get("guideline_id")
        return retrieve_icd_guideline_chapter(gid)
    else:
        return f"Error: The tool '{name}' is not a valid tool."







































import asyncio
import json
import os
from datetime import datetime
from typing import List

import colorama
import numpy as np
import torch

from realhf.api.core.agent_api import Agent, register_agent
from realhf.api.core.data_api import SequenceSample, load_hf_tokenizer
from realhf.api.core.env_api import EnvironmentService
from realhf.api.core.model_api import BundledGenerationOutputs
from realhf.base import constants, logging

logger = logging.getLogger("Math Code Agent")


class MathMultiTurnAgent(Agent):
    """A multi-turn reasoning agent for mathematical tasks.

    In each turn the agent produces an answer and receives evaluation results from the environment.

    By default, we use 4 turns with a token budget=1K at each round.
    """

    def __init__(
        self,
        gconfig,
        tokenizer_path,
        reward_scaling=1.0,
        reward_bias=0.0,
        turn_level_discount: float = 1.0,
        num_turns: int = 5,
    ):
        self.gconfig = gconfig.new(n=1)
        self.tokenizer = load_hf_tokenizer(tokenizer_path)

        self.reward_scaling = reward_scaling
        self.reward_bias = reward_bias
        self.turn_level_discount = turn_level_discount

        self.num_turns = num_turns

    async def collect_trajectory(
        self,
        prompt: SequenceSample,
        env: EnvironmentService,
        obs_queue: asyncio.Queue,
        act_queue: asyncio.Queue,
    ) -> List[SequenceSample]:
        # reset does nothing, just to make it like multi-step environments
        await env.reset()

        assert prompt.bs == 1
        assert self.gconfig.n == 1

        prompt_token_ids = prompt.data["packed_prompts"].cpu().numpy().tolist()
        qid = prompt.ids[0]
        birth_time = int(datetime.now().timestamp() * 1000)

        prompt_str = self.tokenizer.batch_decode(
            [prompt_token_ids],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )[0]

        token_ids = prompt_token_ids
        all_rewards = []
        all_answers = []
        all_success = []
        x = dict(
            keys=[
                "packed_input_ids",
                "prompt_mask",
                "packed_logprobs",
                "seq_no_eos_mask",
                "packed_prompts",
                "version_start",
                "version_end",
                "rewards",
                "birth_time",
            ],
            ids=[qid],
            dtypes=dict(
                packed_prompts=torch.long,
                packed_input_ids=torch.long,
                prompt_mask=torch.bool,
                seq_no_eos_mask=torch.bool,
                version_start=torch.int,
                version_end=torch.int,
                packed_logprobs=torch.float32,
                rewards=torch.float32,
                birth_time=torch.long,
            ),
            trailing_shapes=dict(
                packed_input_ids=(),
                prompt_mask=(),
                seq_no_eos_mask=(),
                packed_prompts=(),
                version_end=(),
                version_start=(),
                packed_logprobs=(),
                rewards=(),
                birth_time=(),
            ),
            seqlens=dict(
                packed_input_ids=[[]],
                packed_logprobs=[[]],
                packed_prompts=[[len(prompt_token_ids)]],
                prompt_mask=[[]],
                seq_no_eos_mask = [[]],  # seq_no_eos_mask=[[1 for _ in range(self.num_turns)]],
                rewards = [[]],          # rewards=[[1 for _ in range(self.num_turns)]],
                version_start = [[]],    # version_start=[[1 for _ in range(self.num_turns)]],
                version_end = [[]],      # version_end=[[1 for _ in range(self.num_turns)]],
                birth_time=[[1]],
            ),
            data=dict(
                packed_prompts=list(prompt_token_ids),
                packed_logprobs=[],
                packed_input_ids=[],
                seq_no_eos_mask=[],
                rewards=[],
                version_start=[],
                version_end=[],
                birth_time=torch.tensor([birth_time], dtype=torch.long),
                prompt_mask=[],
            ),
        )

        for turn in range(self.num_turns):
            await obs_queue.put((qid, token_ids, self.gconfig))

            act: BundledGenerationOutputs = await act_queue.get()

            seq_strs = self.tokenizer.batch_decode(
                act.seqs,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )
            prompt_str = self.tokenizer.batch_decode(
                [act.prompt_ids],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )[0]

            answers = [seq_str.split(prompt_str)[1] for seq_str in seq_strs]



            #### My Large modification ####
            
            llm_response, assistant_message_from_parse, original_think_content = parse_assistant_response_with_tool_calls(answers[0])

            tool_feedback = ""
            for call in assistant_message_from_parse.get("tool_calls", []):
                tool_name = call["name"]
                tool_args = call["arguments"] or {}
                print(f"Executing tool: {tool_name} with args: {tool_args}")
                tool_resp = _safe_tool_call(tool_name, tool_args)
                tool_feedback += f"\n<tool_response>\n{tool_resp}\n</tool_response>"
                logger.debug(f"Tool Call: {tool_feedback[:1000]}")

                
            #### My modification END ####

            # single-step env for evaluating generated solutions
            if assistant_message_from_parse.get("tool_calls"):
                num_generated_sequences = len(answers)
                success = [0.0] * num_generated_sequences
                rewards = [0.0] * num_generated_sequences
                logger.debug(f"Reward Zero because of Tool Call: {rewards}")
            else:
                _, success, *_ = await env.step((qid, answers))
                rewards = [
                    ((float(r) - 0.5) * 2 - self.reward_bias) * self.reward_scaling
                    for r in success
                ]
                logger.debug(f"Reward Positive because End of Generation: {rewards}")

            all_success.extend(success)
            all_answers.extend(answers)

            x["data"]["packed_input_ids"].extend(list(act.seqs[0]))
            x["data"]["packed_logprobs"].extend(list(act.logprobs[0]))
            x["data"]["seq_no_eos_mask"].append(act.no_eos[0])
            all_rewards.append(rewards[0])
            x["data"]["prompt_mask"].extend(
                [1] * act.prompt_len + [0] * (act.seqlens[0] - act.prompt_len)
            )

            x["data"]["version_start"].extend(list(act.version_start))
            x["data"]["version_end"].extend(list(act.version_end))

            x["seqlens"]["packed_input_ids"][0].append(act.seqlens[0])
            x["seqlens"]["packed_logprobs"][0].append(act.seqlens[0] - 1)
            x["seqlens"]["prompt_mask"][0].append(act.seqlens[0])

            x["seqlens"]["seq_no_eos_mask"][0].append(1)
            x["seqlens"]["rewards"][0].append(1)
            x["seqlens"]["version_start"][0].append(1)
            x["seqlens"]["version_end"][0].append(1)

            token_ids = list(act.seqs[0])

            feedback = None

            if not assistant_message_from_parse.get("tool_calls") and not success[0]:
                break

            if success[0]:
                # feedback = "Congratulations! You are correct!"
                break
            else:
                feedback = tool_feedback

            
            feedback = "\n" + self.tokenizer.apply_chat_template(
                [dict(content=feedback, role="user")],
                add_generation_prompt=True,
                tokenize=False,
            )
            
            logger.debug(f"New Feedback: {feedback[:2000]}")
            feedback = self.tokenizer(feedback)["input_ids"]

            generation_buffer = self.gconfig.max_new_tokens
            max_allowed_len = self.max_context_length - generation_buffer

            
            token_ids.extend(feedback)
            print(f"LLM Reponse: {llm_response}")
            print(f"Feedback tokens: {feedback[:2000]}")
            print(f"Feedback token ids: {len(feedback)}")
            print(f"Original feedback: {feedback[:2000]}")
            print(f"Total token length: {len(token_ids)}")
            print(f"Tokens decoded: {self.tokenizer.decode(token_ids)}")

        self.log_rewards_to_file(
            str(qid),
            prompt_str,
            seqlens=x["seqlens"]["packed_input_ids"][0],
            answers=all_answers,
            prompt_len=len(prompt_token_ids),
            rewards=all_rewards,
            success=all_success,
            version_starts=x["data"]["version_start"],
            version_ends=x["data"]["version_end"],
        )

        for i in reversed(range(len(all_rewards) - 1)):
            all_rewards[i] = (
                all_rewards[i] + all_rewards[i + 1] * self.turn_level_discount
            )
        x["data"]["rewards"] = all_rewards

        for k in x["keys"]:
            if not isinstance(x["data"][k], torch.Tensor):
                x["data"][k] = torch.tensor(x["data"][k], dtype=x["dtypes"][k])

        x = SequenceSample(**x)

        if "task_ids" in prompt.keys:
            y = SequenceSample(
                keys=["task_ids"],
                ids=[qid],
                dtypes=dict(task_ids=torch.long),
                trailing_shapes=dict(task_ids=()),
                seqlens=dict(task_ids=[[1]]),
                data=dict(task_ids=prompt.data["task_ids"]),
            )
            x.update_(y)

        return [x]

    def log_rewards_to_file(
        self,
        qid: str,
        prompt: str,
        prompt_len: int,
        answers: List[str],
        seqlens: List[int],
        rewards: List[float],
        success: List[bool],
        version_starts: List[int],
        version_ends: List[int],
    ):
        group_size = len(answers)

        for group_idx in range(group_size):
            # NOTE: we can ensure that only one process is logging this query id
            gen_file_path = os.path.join(
                constants.LOG_ROOT,
                constants.experiment_name(),
                constants.trial_name(),
                "generated",
                str(version_starts[group_idx]),
                f"{qid}.txt",
            )
            os.makedirs(os.path.dirname(gen_file_path), exist_ok=True)

            version_start = version_starts[group_idx]
            version_end = version_ends[group_idx]
            reward = rewards[group_idx]
            answer = answers[group_idx]
            seqlen = seqlens[group_idx]
            with open(gen_file_path, "a") as _f:
                info = "\n".join(
                    [
                        f"idx: {group_idx + 1} / {group_size}, seqlen: {seqlen}, "
                        f"head version: {version_start}, tail version: {version_end}.",
                        f"reward is {reward}, prompt is {colorama.Fore.YELLOW + colorama.Style.DIM}{prompt}{colorama.Style.RESET_ALL}",
                        f"sequence is: {colorama.Fore.YELLOW + colorama.Style.DIM}{answer}{colorama.Style.RESET_ALL}.",
                    ]
                )
                _f.write(info + "\n")

            train_pass_monitor_file_path = os.path.join(
                constants.LOG_ROOT,
                constants.experiment_name(),
                constants.trial_name(),
                "training_monitor",
                str(version_starts[group_idx]),
                f"{qid}.jsonl",
            )
            os.makedirs(os.path.dirname(train_pass_monitor_file_path), exist_ok=True)

            with open(train_pass_monitor_file_path, "a") as monitor_file:
                monitor_file.write(
                    json.dumps(
                        {
                            "version_start": int(version_start),
                            "version_end": int(version_end),
                            "success": bool(success),
                            "prompt_len": prompt_len,
                            "answer_len": seqlen - prompt_len,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


register_agent("medical-coding-multi-turn-agent", MathMultiTurnAgent)