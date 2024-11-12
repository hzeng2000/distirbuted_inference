import pandas as pd
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from typing import List, Optional, Tuple
from vllm.inputs import PromptInputs
import json

def sample_OpenOrca_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: Optional[AutoTokenizer],
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[PromptInputs, int, int, str]]:
    
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    
    # Read the dataset
    raw_dataset = pd.read_parquet(dataset_path)
    
    # Merge question and response
    dataset = [(row['question'], row['response']) for _, row in raw_dataset.iterrows()]

    # Initialize filtered dataset
    filtered_dataset: List[Tuple[str, int, int, str]] = []
    
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the question (prompt) and response
        prompt = dataset[i][0]  # question
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]  # response
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        
        # Filter out sequences that are too short
        if prompt_len < 4 or output_len < 4:
            continue

        # Append to filtered data
        filtered_dataset.append((prompt, prompt_len, output_len, completion))

    return filtered_dataset

def sample_LLaVA_CC3M_Pretrain_595K_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[PromptInputs, int, int, str]]:
    import os
    import base64
    from PIL import Image 
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    
    with open(os.path.join(dataset_path, 'chat.json')) as f:
        dataset = json.load(f)

    # filtering
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"],
                data["image"]) for data in dataset]

    # random.shuffle(dataset)

    # process the dataset
    filtered_dataset: List[Tuple[str, str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        prompt = dataset[i][0]
        completion = dataset[i][1]
        # image_path = os.path.join(dataset_path, 'images', dataset[i][2])
        image_path = os.path.join(dataset_path, dataset[i][2])
        
        # check if image exists
        if not os.path.exists(image_path):
            print(f"Image file {image_path} does not exist, skipping")
            continue

        prompt_token_ids = tokenizer(prompt).input_ids
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        
        if prompt_len < 4 or output_len < 4:
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            continue
        

        PromptInputs = {
                # "prompt": f"USER: <image>\n{prompt}\nASSISTANT:",
                "prompt": prompt,
                "multi_modal_data": {
                    "image": Image.open(image_path)
                }
            }
        filtered_dataset.append((PromptInputs, prompt_len, output_len, completion))

    return filtered_dataset