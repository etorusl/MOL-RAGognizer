"""
NOTE: If you get `ImportError: cannot import name 'NEED_SETUP_CACHE_CLASSES_MAPPING' from 'transformers.generation.utils'`, then please adapt the line to this:
from transformers.generation.utils import logger
NEED_SETUP_CACHE_CLASSES_MAPPING = []

Also change lines 1101 and 1133 in `.venv/lib/python3.10/site-packages/trl/trainer/sft_trainer.py` to:
`if False and not self.args.use_liger_kernel:`
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ragognizer')))

import json
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pandas as pd

load_dotenv(find_dotenv()) # might require "HF_TOKEN" to be set in the .env file ("HF_HOME" optionally)

from ragognizer.benchmarks.RAGTruth import RAGTruth
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from trl import SFTTrainer, SFTConfig

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score

parser = ArgumentParser()
parser.add_argument("model")
parser.add_argument("dataset")
parser.add_argument("outname")
parser.add_argument("--balanced", action="store_true")
parser.add_argument("--linear", action="store_true")
parser.add_argument("--masked", action="store_true")
parser.add_argument("--allentries", action="store_true")
parser.add_argument("--ragtruth", action="store_true")
parser.add_argument("--nolmhead", action="store_true")
parser.add_argument("--headatend", action="store_true")
parser.add_argument("--quantized", action="store_true")
parser.add_argument("--mlp", action="store_true", help="Use separate MLP + LayerAggregator instead of transformer-heads")
parser.add_argument("--mlp_hidden_dims", type=str, default="1024,512", help="Comma-separated MLP hidden dimensions")
parser.add_argument("--multimodal", action="store_true", help="Use multimodal dataset (shroom-vision) instead of RAGognize")
parser.add_argument("--image_dir", type=str, default="./images/shroom", help="Directory containing images referenced by dataset")
args = parser.parse_args()

EVAL_PERC = 0.15 # For RAGTruth
EPOCHS = 5
MODEL_NAME = args.model.strip() # "mistralai/Mistral-7B-Instruct-v0.1" # "meta-llama/Llama-3.1-8B-Instruct" # "mistralai/Mistral-7B-Instruct-v0.3" # "meta-llama/Llama-2-7b-chat-hf"
OUTNAME = args.outname.strip()
BALANCED = args.balanced
LINEAR = args.linear
MASKED = args.masked
ALL_ENTRIES = args.allentries
USE_RAGTRUTH = args.ragtruth
NO_LM_HEAD = args.nolmhead
HEAD_AT_END = args.headatend
QUANTIZED = args.quantized
USE_MLP = args.mlp
MLP_HIDDEN_DIMS = [int(x) for x in args.mlp_hidden_dims.split(",")]
USE_MULTIMODAL = args.multimodal
IMAGE_DIR = args.image_dir

PAD_TOKEN = {
    "meta-llama/Llama-2-7b-chat-hf": "<pad>",
    "mistralai/Mistral-7B-Instruct-v0.1": "<pad>",
    "mistralai/Mistral-7B-Instruct-v0.3": "<pad>",
    "LiquidAI/LFM2-1.2B-RAG": None, #"<|pad|>",
    "meta-llama/Llama-3.1-8B-Instruct": "<|reserved_special_token_0|>",
    "meta-llama/Llama-3.2-1B-Instruct": "<|reserved_special_token_0|>",
}.get(MODEL_NAME, None)

hidden_size = None  # set later: MLP mode from model config, transformer-heads mode from get_model_params
head_name = "hallu_labels"

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURR_DIR, "..", "ragognize", "data")
OUTPUT_DIR = os.path.join(CURR_DIR, OUTNAME, MODEL_NAME.split("/")[-1])
os.makedirs(OUTPUT_DIR, exist_ok=True)

# if PAD_TOKEN is None:
#     raise Exception("Padding token is None! Please, set add an appropiate pad_token.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
_processor = None
try:
    from transformers import AutoProcessor
    _processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if hasattr(_processor, 'tokenizer'):
        tokenizer = _processor.tokenizer
    print(f"AutoProcessor loaded OK for {MODEL_NAME}")
except Exception as e:
    print(f"Warning: AutoProcessor failed ({e}), using AutoTokenizer only")

if USE_MULTIMODAL and _processor is None:
    raise RuntimeError("Multimodal mode requires AutoProcessor. Install torchvision and ensure preprocessor_config.json is downloaded.")

def _apply_chat_template(messages, tokenize=True, add_generation_prompt=False, **kwargs):
    text = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=add_generation_prompt, **kwargs)
    if tokenize:
        return tokenizer.encode(text, add_special_tokens=True)
    return text

# print(tokenizer.pad_token_id)
# print(tokenizer.convert_ids_to_tokens([0])[0])

def mask_mixed_windows(arr, false_l=4, true_l=3, true_only_backward=True, min_true_group_size=3):
    arr = np.array(arr, dtype=object)  # use object type to allow np.nan assignment
    result = arr.copy()

    for i in range(len(arr)):
        if arr[i]:
            l = true_l
        else:
            l = false_l

        start = max(0, i - l)
        if true_only_backward and arr[i]:
            end = i+1
        else:
            end = min(len(arr), i + l + 1)
        window = arr[start:end]

        true_group_size = 0
        if arr[i]:
            true_group_size += 1
            for j in range(i+1, arr.shape[0]):
                if not arr[j]:
                    break
                true_group_size += 1
            for j in range(i-1, -1, -1):
                if not arr[j]:
                    break
                true_group_size += 1

        if true_group_size > 0 and true_group_size <= min_true_group_size:
            continue

        if True in window and False in window:
            result[i] = np.nan

    return result

def nan_following_true_groups(arr):
    arr = np.array(arr, dtype=object)
    result = arr.copy()

    in_first_true_group = False
    after_first_true_group = False

    for i in range(len(arr)):
        if arr[i] is True:
            if not in_first_true_group and not after_first_true_group:
                # First time seeing True → start first True group
                in_first_true_group = True
            elif after_first_true_group:
                # Already past first group, mask any future True
                result[i] = np.nan
        else:
            if in_first_true_group:
                # We finished the first True group
                in_first_true_group = False
                after_first_true_group = True
            if after_first_true_group:
                result[i] = np.nan

    return result

# Function to prepare the prompts for fine-tuning
def formatted_dataset(dataset, max_length=None):
    entries = []

    model = MODEL_NAME.split("/", 1)[1]
    
    # Go through dataset row by row
    for i, entry in enumerate(dataset):
        if max_length is not None and len(entries) >= max_length:
            break

        if "responses" not in entry:
            continue

        for curr_model in entry["responses"]:
            if curr_model == "golden_answer":
                continue

            if (curr_model == model) or ALL_ENTRIES:
                annotations = entry["responses"][curr_model]

                prompt = entry["rag_prompt"] + [
                    {
                        "role": "assistant",
                        "content": entry["responses"][curr_model]["text"]
                    }
                ]
                entry_tok = { "input_ids": _apply_chat_template(prompt, tokenize=True, enable_thinking=False) }
                entry_tok["attention_mask"] = np.ones(shape=(len(entry_tok["input_ids"]),), dtype=np.int32).tolist()

                entry_tok["labels"] = list(entry_tok["input_ids"])
                user_tokens = _apply_chat_template(entry["rag_prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False)
                assistant_token_start = len(user_tokens)

                token_starts = []
                ass_tokens = entry_tok["input_ids"][assistant_token_start:]
                last_start = 0
                last_token_found = True
                for ass_i in range(len(ass_tokens)):
                    curr_text = tokenizer.decode(ass_tokens[:ass_i+1], skip_special_tokens=True)

                    try:
                        starts_at = entry["responses"][curr_model]["text"].index(curr_text) + len(tokenizer.decode(ass_tokens[:ass_i], skip_special_tokens=True))
                        last_token_found = True
                    except:
                        starts_at = last_start
                        if last_token_found:
                            starts_at += 1
                        last_token_found = False
                        # if entry["responses"][curr_model]["text"] in curr_text:
                        #     starts_at = len(entry["responses"][curr_model]["text"])
                        # else:
                        #     print(f'#{curr_text}#')
                        #     print(f'#{entry["responses"][curr_model]["text"]}#')
                        #     print(f"curr token:", ass_tokens[ass_i], "aka", f"'{tokenizer.decode([ass_tokens[ass_i]], skip_special_tokens=False)}'")
                        #     print(f"index: {ass_i} | len: {len(ass_tokens)}")
                        #     raise Exception("Failed to verify that the cum text is part of the entire LLM response")
                        
                    token_starts.append(starts_at)
                    last_start = starts_at

                # print(token_starts)
                # print(f'#{entry["responses"][curr_model]["text"][token_starts[0]:token_starts[-1]]}#')
                # print(f'#{entry["responses"][curr_model]["text"]}#')
                # exit()

                label_per_token = np.zeros(shape=len(token_starts), dtype=bool)
                for h in annotations["hallucinations"]:
                    first_token_index_of_hallu = len(token_starts) - 1
                    try:
                        while token_starts[first_token_index_of_hallu] > h["start"]:
                            first_token_index_of_hallu -= 1
                    except:
                        first_token_index_of_hallu = 0
                    first_token_index_of_hallu = max(0, min(first_token_index_of_hallu, len(token_starts) - 1))

                    last_token_index_of_hallu = 0
                    try:
                        while token_starts[last_token_index_of_hallu] < h["end"]:
                            last_token_index_of_hallu += 1
                    except:
                        last_token_index_of_hallu = len(token_starts) - 1
                    last_token_index_of_hallu = max(0, min(last_token_index_of_hallu, len(token_starts) - 1))

                    # Set all tokens of hallucination to True
                    label_per_token[first_token_index_of_hallu:last_token_index_of_hallu] |= True
                
                if MASKED:
                    label_per_token = nan_following_true_groups(mask_mixed_windows(label_per_token, false_l=8, true_l=3, min_true_group_size=3)).astype(float)
                    label_per_token[np.isnan(label_per_token)] = -1

                # Set label of user prompt tokens to -1 (which will be masked out and ignored in loss computation)
                labels_before_response = np.full(shape=len(entry_tok["input_ids"]) - label_per_token.shape[0], fill_value=-1)
                hallu_per_token = np.concatenate([labels_before_response, label_per_token]).astype(float).tolist()
                entry_tok[head_name] = [[h] for h in hallu_per_token]
                # print(tokenizer.decode(entry_tok["labels"][assistant_token_start:], skip_special_tokens=False))

                # Ignore user tokens for language generation during training
                for label_i in range(assistant_token_start):
                    entry_tok["labels"][label_i] = -100

                entries.append(entry_tok)

                # Compare and check if still correct
                # if np.sum(label_per_token) > 0:
                #     print("#" * 50)
                #     print(annotations["hallucinations"])
                #     hids = np.array(entry_tok["input_ids"])[np.array(hallu_per_token) > 0]
                #     print(hids)
                #     print(f"%{tokenizer.batch_decode([hids], skip_special_tokens=False)[0]}%")
                #     input()

    # Create new dataset
    ds = Dataset.from_list(entries)
    if len(entries) > 0:
        e0 = entries[0]
        hp = np.array([h[0] for h in e0[head_name]])
        print(f"DEBUG formatted_dataset: {len(entries)} entries, ids={len(e0['input_ids'])} hallu_nonneg={int((hp >= 0).sum())} hallu_ones={int((hp == 1).sum())}")
    return ds

def formatted_ragtruth(get_test: bool=False, eval_perc: float=None):
    train_entries = []
    val_entries = []

    if get_test:
        eval_perc = 0.0
    else:
        if eval_perc is None:
            eval_perc = EVAL_PERC

    summ_ragtruth = RAGTruth("all", "Summary", is_test=get_test).get_entries(token_level=True)
    qa_ragtruth   = RAGTruth("all", "QA", is_test=get_test).get_entries(token_level=True)
    d2t_ragtruth  = RAGTruth("all", "Data2txt", is_test=get_test).get_entries(token_level=True)

    summ_border = int(len(summ_ragtruth) * (1 - EVAL_PERC))
    summ_ragtruth_train = summ_ragtruth[:summ_border]
    summ_ragtruth_val  = summ_ragtruth[summ_border:]

    qa_border = int(len(qa_ragtruth) * (1 - EVAL_PERC))
    qa_ragtruth_train = qa_ragtruth[:qa_border]
    qa_ragtruth_val  = qa_ragtruth[qa_border:]

    d2t_border = int(len(d2t_ragtruth) * (1 - EVAL_PERC))
    d2t_ragtruth_train = d2t_ragtruth[:d2t_border]
    d2t_ragtruth_val  = d2t_ragtruth[d2t_border:]

    train_list = summ_ragtruth_train + qa_ragtruth_train + d2t_ragtruth_train
    val_list = summ_ragtruth_val + qa_ragtruth_val + d2t_ragtruth_val

    if get_test:
        splits = [train_list]
    else:
        splits = [train_list, val_list]

    for split_index in range(len(splits)):
        for entry in splits[split_index]:
            annotations = entry["annotations"]

            prompt = entry["chat"]
            user_prompt = [msg for msg in prompt if msg["role"] != "assistant"]

            entry_tok = { "input_ids": _apply_chat_template(prompt, tokenize=True, enable_thinking=False) }
            entry_tok["attention_mask"] = np.ones(shape=(len(entry_tok["input_ids"]),), dtype=np.int32).tolist()

            entry_tok["labels"] = list(entry_tok["input_ids"])
            user_tokens = _apply_chat_template(user_prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            assistant_token_start = len(user_tokens)


            token_starts = []
            ass_tokens = entry_tok["input_ids"][assistant_token_start:]
            last_start = 0
            last_token_found = True
            for ass_i in range(len(ass_tokens)):
                curr_text = tokenizer.decode(ass_tokens[:ass_i+1], skip_special_tokens=True)

                try:
                    starts_at = entry["response"].index(curr_text) + len(tokenizer.decode(ass_tokens[:ass_i], skip_special_tokens=True))
                    last_token_found = True
                except:
                    starts_at = last_start
                    if last_token_found:
                        starts_at += 1
                    last_token_found = False
                    # if entry["responses"][curr_model]["text"] in curr_text:
                    #     starts_at = len(entry["responses"][curr_model]["text"])
                    # else:
                    #     print(f'#{curr_text}#')
                    #     print(f'#{entry["responses"][curr_model]["text"]}#')
                    #     print(f"curr token:", ass_tokens[ass_i], "aka", f"'{tokenizer.decode([ass_tokens[ass_i]], skip_special_tokens=False)}'")
                    #     print(f"index: {ass_i} | len: {len(ass_tokens)}")
                    #     raise Exception("Failed to verify that the cum text is part of the entire LLM response")
                    
                token_starts.append(starts_at)
                last_start = starts_at

            # print(token_starts)
            # print(f'#{entry["responses"][curr_model]["text"][token_starts[0]:token_starts[-1]]}#')
            # print(f'#{entry["responses"][curr_model]["text"]}#')
            # exit()

            label_per_token = np.zeros(shape=len(token_starts), dtype=bool)
            for h in annotations:
                if h["label"] == 0:
                    continue

                first_token_index_of_hallu = len(token_starts) - 1
                try:
                    while token_starts[first_token_index_of_hallu] > h["start"]:
                        first_token_index_of_hallu -= 1
                except:
                    first_token_index_of_hallu = 0
                first_token_index_of_hallu = max(0, min(first_token_index_of_hallu, len(token_starts) - 1))

                last_token_index_of_hallu = 0
                try:
                    while token_starts[last_token_index_of_hallu] < h["end"]:
                        last_token_index_of_hallu += 1
                except:
                    last_token_index_of_hallu = len(token_starts) - 1
                last_token_index_of_hallu = max(0, min(last_token_index_of_hallu, len(token_starts) - 1))

                # Set all tokens of hallucination to True
                label_per_token[first_token_index_of_hallu:last_token_index_of_hallu] |= True
            
            if MASKED:
                label_per_token = nan_following_true_groups(mask_mixed_windows(label_per_token, false_l=8, true_l=3, min_true_group_size=3)).astype(float)
                label_per_token[np.isnan(label_per_token)] = -1

            # Set label of user prompt tokens to -1 (which will be masked out and ignored in loss computation)
            labels_before_response = np.full(shape=len(entry_tok["input_ids"]) - label_per_token.shape[0], fill_value=-1)
            hallu_per_token = np.concatenate([labels_before_response, label_per_token]).astype(float).tolist()
            entry_tok[head_name] = [[h] for h in hallu_per_token]
            # print(tokenizer.decode(entry_tok["labels"][assistant_token_start:], skip_special_tokens=False))

            # Ignore user tokens for language generation during training
            for label_i in range(assistant_token_start):
                entry_tok["labels"][label_i] = -100

            if split_index == 0:
                train_entries.append(entry_tok)
            else:
                val_entries.append(entry_tok)

            # Compare and check if still correct
            # if np.sum(label_per_token) > 0:
            #     print("#" * 50)
            #     print(annotations)
            #     hids = np.array(entry_tok["input_ids"])[np.array(hallu_per_token) > 0]
            #     print(hids)
            #     print(f"%{tokenizer.batch_decode([hids], skip_special_tokens=False)[0]}%")
            #     input()

    # Create new datasets
    if get_test:
        Dataset.from_list(train_entries)
    else:
        return Dataset.from_list(train_entries), Dataset.from_list(val_entries)


def formatted_multimodal(dataset_dir, image_dir, processor):
    import glob
    jsonl_files = sorted(glob.glob(os.path.join(dataset_dir, "*.labeled.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(f"No labeled JSONL files found in {dataset_dir}")
    print(f"Loading multimodal data from: {jsonl_files}")

    raw_entries = []
    for fpath in jsonl_files:
        with open(fpath) as f:
            for line in f:
                raw_entries.append(json.loads(line))
    print(f"Total raw entries: {len(raw_entries)}")

    entries = []
    missing = 0
    for entry in raw_entries:
        response = entry["response"]
        prompt_text = entry["prompt"]
        image_path = os.path.join(image_dir, entry["image_name"])
        if not os.path.exists(image_path):
            missing += 1
            if missing == 1:
                print(f"First missing image: {image_path}")
            continue

        user_content = [
            {"type": "image", "url": image_path},
            {"type": "text", "text": prompt_text},
        ]
        full_msgs = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
        user_msgs = [
            {"role": "user", "content": user_content},
        ]

        try:
            full_ids = _apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False)
            user_ids = _apply_chat_template(user_msgs, tokenize=True, add_generation_prompt=True)
        except Exception as e:
            print(f"WARNING: Failed to tokenize entry {entry.get('id', '?')}: {e}")
            continue

        assistant_token_start = len(user_ids)
        if isinstance(full_ids, list):
            full_ids = torch.tensor(full_ids, dtype=torch.long)

        token_starts = []
        ass_tokens = full_ids[assistant_token_start:]
        last_start = 0
        for ass_i in range(len(ass_tokens)):
            curr_text = tokenizer.decode(ass_tokens[:ass_i+1], skip_special_tokens=True)
            try:
                starts_at = response.index(curr_text) + len(tokenizer.decode(ass_tokens[:ass_i], skip_special_tokens=True))
            except Exception:
                starts_at = last_start + 1 if ass_i > 0 else 0
            token_starts.append(starts_at)
            last_start = starts_at

        label_per_token = np.zeros(len(token_starts), dtype=bool)
        for lbl in entry.get("labels", []):
            first = len(token_starts) - 1
            try:
                while token_starts[first] > lbl["start"]:
                    first -= 1
            except Exception:
                first = 0
            first = max(0, min(first, len(token_starts) - 1))
            last = 0
            try:
                while token_starts[last] < lbl["end"]:
                    last += 1
            except Exception:
                last = len(token_starts) - 1
            last = max(0, min(last, len(token_starts) - 1))
            label_per_token[first:last] |= True

        if MASKED:
            label_per_token = nan_following_true_groups(mask_mixed_windows(label_per_token, false_l=8, true_l=3, min_true_group_size=3)).astype(float)
            label_per_token[np.isnan(label_per_token)] = -1

        labels_before_response = np.full(len(full_ids) - len(label_per_token), -1.0)
        hallu_per_token = np.concatenate([labels_before_response, label_per_token.astype(float)]).tolist()

        entry_tok = {
            "input_ids": full_ids.tolist() if isinstance(full_ids, torch.Tensor) else list(full_ids),
            "attention_mask": np.ones(len(full_ids), dtype=np.int32).tolist(),
            "labels": full_ids.tolist() if isinstance(full_ids, torch.Tensor) else list(full_ids),
            head_name: [[h] for h in hallu_per_token],
            "image_path": image_path,
        }
        for label_i in range(assistant_token_start):
            entry_tok["labels"][label_i] = -100

        entries.append(entry_tok)

    if len(entries) == 0:
        raise RuntimeError(f"No valid entries produced ({missing} images missing, dir: {image_dir}). Check image_dir path.")

    e0 = entries[0]
    hp = np.array([h[0] for h in e0[head_name]])
    print(f"DEBUG multimodal: {len(entries)} entries, ids={len(e0['input_ids'])} hallu_nonneg={int((hp >= 0).sum())} hallu_ones={int((hp == 1).sum())}")

    ds = Dataset.from_list(entries)
    split_ds = ds.train_test_split(test_size=0.1, seed=42)
    return split_ds["train"], split_ds["test"]

if USE_MULTIMODAL:
    train_dataset, val_dataset = formatted_multimodal(
        args.dataset, IMAGE_DIR,
        _processor if _processor is not None else tokenizer
    )
    test_dataset = val_dataset
elif USE_RAGTRUTH:
    train_dataset, val_dataset = formatted_ragtruth()
    test_dataset = formatted_ragtruth(get_test=True)
else:
    dataset_path = args.dataset
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        # dataset = load_dataset("F4biian/RAGognize")
        dataset = load_dataset(dataset_path)
    split_dataset = formatted_dataset(dataset["train"])
    split_dataset = split_dataset.train_test_split(
        test_size=0.1,
        seed=42
    )
    train_dataset = split_dataset["train"]
    val_dataset   = split_dataset["test"]
    test_dataset  = formatted_dataset(dataset["test"])

if len(train_dataset) == 0:
    model_short = MODEL_NAME.split("/", 1)[1]
    raise ValueError(
        f"Empty dataset! Model '{model_short}' has no responses in the dataset. "
        f"Use --allentries to train on all available responses, or switch to a model that exists in the dataset."
    )

if not USE_MLP:
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", head_name, "labels"],
    )
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", head_name, "labels"],
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", head_name, "labels"],
    )

# Calculcate weight for hallucinated tokens (to account for imbalance)
zeros = 0
ones = 0
if BALANCED:
    for ten in train_dataset[head_name]:
        arr = np.array(ten) if isinstance(ten, list) else ten.numpy()
        zeros += int((arr == 0).sum())
        ones += int((arr == 1).sum())

    ones_weight = 1 / (ones / (zeros + ones)) if (zeros + ones) > 0 else 1.0
else:
    ones_weight = 1.0

print("Zeros:", zeros)
print(" Ones:", ones)
print("Total:", ones + zeros)
print("Ones Weight:", ones_weight)

print(train_dataset)
print(val_dataset)
print(test_dataset)

if USE_MLP:
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model
    import torch.nn.functional as F
    from ragognizer.detectors.RAGognizer import LayerAggregator, MLP
    from torch.optim import AdamW
    from PIL import Image

    quantization_config_mlp = None
    if QUANTIZED:
        quantization_config_mlp = BitsAndBytesConfig(load_in_8bit=True)

    lora_config_mlp = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=None,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config_mlp,
        device_map={"": torch.cuda.current_device()},
        torch_dtype=torch.bfloat16,
    )
    llm_model = get_peft_model(llm_model, lora_config_mlp)

    if tokenizer.pad_token is None and PAD_TOKEN:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm_model.config.pad_token_id = tokenizer.pad_token_id
    if "reserved" not in str(tokenizer.pad_token):
        llm_model.resize_token_embeddings(len(tokenizer))

    cfg = llm_model.config
    num_layers = (cfg.text_config if hasattr(cfg, 'text_config') else cfg).num_hidden_layers + 1
    hidden_size = (cfg.text_config if hasattr(cfg, 'text_config') else cfg).hidden_size
    layer_aggregator = LayerAggregator(num_layers).to("cuda")
    mlp = MLP(input_size=hidden_size, hidden_dims=MLP_HIDDEN_DIMS).to("cuda")

    optimizer = AdamW([
        {"params": [p for n, p in llm_model.named_parameters() if p.requires_grad], "lr": 4e-5},
        {"params": layer_aggregator.parameters(), "lr": 4e-5},
        {"params": mlp.parameters(), "lr": 4e-5},
    ])

    def _extract_entry(item, key, dtype):
        val = item[key]
        try:
            return torch.tensor(val, dtype=dtype).view(-1)
        except (TypeError, ValueError):
            pass
        if isinstance(val, dict):
            vals = list(val.values())
            if len(vals) == 1:
                val = vals[0]
            else:
                val = vals
        return torch.tensor(val, dtype=dtype).view(-1)

    def _dataset_to_tensors(ds):
        entries = []
        for i in range(len(ds)):
            item = ds[i]
            ids = _extract_entry(item, "input_ids", torch.long)
            L = ids.size(0)
            am = torch.ones(L, dtype=torch.long)
            try:
                lbl = _extract_entry(item, "labels", torch.long)
                if lbl.size(0) < L:
                    lbl = torch.cat([lbl, torch.full((L - lbl.size(0),), -100, dtype=torch.long)])
                elif lbl.size(0) > L:
                    lbl = lbl[:L]
            except Exception:
                lbl = ids.clone()
            hl_raw = item[head_name]
            if isinstance(hl_raw, torch.Tensor):
                hl_raw = hl_raw.flatten().tolist()
            elif isinstance(hl_raw, dict):
                hl_raw = list(hl_raw.values())
                if len(hl_raw) == 1:
                    hl_raw = list(hl_raw[0]) if hasattr(hl_raw[0], '__iter__') else [hl_raw[0]]
            hl_vals = []
            for v in hl_raw:
                if isinstance(v, (int, float)):
                    hl_vals.append(v)
                elif hasattr(v, '__iter__') and not isinstance(v, str):
                    hl_vals.append(list(v)[0] if len(list(v)) > 0 else -1.0)
                else:
                    hl_vals.append(-1.0)
            hl = torch.tensor(hl_vals, dtype=torch.float32)
            if hl.size(0) < L:
                hl = torch.cat([hl, torch.full((L - hl.size(0),), -1.0)])
            elif hl.size(0) > L:
                hl = hl[:L]
            entries.append((ids, am, lbl, hl, item.get("image_path", None)))
        print(f"_dataset_to_tensors: {len(entries)} samples, first hallu min={entries[0][3].min().item():.1f} max={entries[0][3].max().item():.1f} nonzero={(entries[0][3] != -1).sum().item()} nonneg={(entries[0][3] >= 0).sum().item()}")
        return entries

    train_tensors = _dataset_to_tensors(train_dataset)
    val_tensors = _dataset_to_tensors(val_dataset)
    test_tensors = _dataset_to_tensors(test_dataset)

    def collate_tensors(batch):
        max_len = max(t[0].size(0) for t in batch)
        B = len(batch)
        input_ids = torch.full((B, max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn_mask = torch.zeros(B, max_len, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        hallu = torch.full((B, max_len), -1.0)
        pixel_values_list = []
        image_sizes_list = []
        for i, tup in enumerate(batch):
            ids, am, lbl, hl, img_path = tup
            L = ids.size(0)
            input_ids[i, :L] = ids
            attn_mask[i, :L] = am
            labels[i, :L] = lbl
            hallu[i, :L] = hl
            if USE_MULTIMODAL and img_path is not None:
                from PIL import Image
                img = Image.open(img_path).convert("RGB")
                proc_inputs = _processor(image=img, return_tensors="pt")
                pixel_values_list.append(proc_inputs["pixel_values"].squeeze(0))
                image_sizes_list.append(proc_inputs.get("image_sizes", torch.tensor(img.size[::-1])).squeeze(0))
        batch_dict = {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels, head_name: hallu}
        if pixel_values_list:
            batch_dict["pixel_values"] = torch.stack(pixel_values_list)
            batch_dict["image_sizes"] = torch.stack(image_sizes_list)
        return batch_dict

    train_loader = DataLoader(train_tensors, batch_size=1, shuffle=True, collate_fn=collate_tensors)
    val_loader = DataLoader(val_tensors, batch_size=1, shuffle=False, collate_fn=collate_tensors)
    test_loader = DataLoader(test_tensors, batch_size=1, shuffle=False, collate_fn=collate_tensors)

    GRAD_ACCUM = 8
    SCALE_FACTOR = GRAD_ACCUM
    pos_weight_tensor = torch.tensor([ones_weight]).to("cuda") if BALANCED else None

    def eval_mlp(is_val=True):
        llm_model.eval()
        layer_aggregator.eval()
        mlp.eval()
        loader = val_loader if is_val else test_loader

        all_probs = []
        all_labels = []
        total_lm_loss = 0.0
        total_hallu_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc="Evaluating"):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.no_grad():
                outputs = llm_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch.get("pixel_values"),
                    image_sizes=batch.get("image_sizes"),
                    output_hidden_states=True,
                )
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                lm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                hidden_states = torch.stack(outputs.hidden_states, dim=1)
                aggregated = layer_aggregator(hidden_states)
                hallu_logits = mlp(aggregated).squeeze(-1)
                hallu_labels = batch[head_name]
                mask = hallu_labels >= -0.1
                if mask.any():
                    hallu_loss = F.binary_cross_entropy_with_logits(
                        hallu_logits[mask], hallu_labels[mask],
                        pos_weight=pos_weight_tensor,
                    )
                else:
                    hallu_loss = torch.tensor(0.0, device="cuda")
                probs = torch.sigmoid(hallu_logits[mask]).cpu().numpy()
                labels_np = hallu_labels[mask].cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels_np.tolist())
                total_lm_loss += lm_loss.item()
                total_hallu_loss += hallu_loss.item()
                num_batches += 1
            del outputs, aggregated, hallu_logits, hallu_labels
            torch.cuda.empty_cache()

        all_probs_np = np.array(all_probs)
        all_labels_np = np.array(all_labels, dtype=int)
        if len(all_labels_np) == 0:
            return {
                "loss": (total_lm_loss + total_hallu_loss) / max(num_batches, 1),
                "roc_auc": 0.5,
                "pr_auc": 0.0,
                "best_threshold": 0.5,
            }
        total_roc_auc = roc_auc_score(all_labels_np, all_probs_np)
        total_pr_auc = average_precision_score(all_labels_np, all_probs_np)
        fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_np)
        best_threshold = thresholds[np.argmax(tpr - fpr)]

        return {
            "loss": (total_lm_loss + total_hallu_loss) / max(num_batches, 1),
            "roc_auc": float(total_roc_auc),
            "pr_auc": float(total_pr_auc),
            "best_threshold": float(best_threshold),
        }

    training_loss_history = pd.Series()
    eval_loss_history = pd.Series()
    eval_auroc_history = pd.Series()
    eval_auprc_history = pd.Series()

    print("Pre-Training Evaluation (MLP mode)")
    eval_res = eval_mlp()
    print("evaluation:", eval_res)
    eval_loss_history[0] = eval_res["loss"]
    eval_auroc_history[0] = eval_res["roc_auc"]
    eval_auprc_history[0] = eval_res["pr_auc"]

    for epoch in range(EPOCHS):
        llm_model.train()
        layer_aggregator.train()
        mlp.train()

        total_loss_epoch = 0.0
        optimizer.zero_grad()

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for bi, batch in enumerate(progress):
            batch = {k: v.to("cuda") for k, v in batch.items()}

            outputs = llm_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            hidden_states = torch.stack(outputs.hidden_states, dim=1)
            aggregated = layer_aggregator(hidden_states)
            hallu_logits = mlp(aggregated).squeeze(-1)
            hallu_labels = batch[head_name]
            mask = hallu_labels >= -0.1
            if mask.any():
                hallu_loss = F.binary_cross_entropy_with_logits(
                    hallu_logits[mask], hallu_labels[mask],
                    pos_weight=pos_weight_tensor,
                )
            else:
                hallu_loss = torch.tensor(0.0, device="cuda")

            loss = (lm_loss + hallu_loss) / SCALE_FACTOR
            loss.backward()
            total_loss_epoch += loss.item() * SCALE_FACTOR

            if (bi + 1) % GRAD_ACCUM == 0 or (bi + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            progress.set_postfix({"loss": f"{loss.item() * SCALE_FACTOR:.4f}"})
            del outputs, hidden_states, aggregated, hallu_logits, hallu_labels

        torch.cuda.empty_cache()
        avg_loss = total_loss_epoch / len(train_loader)
        training_loss_history[epoch + 1] = avg_loss
        print(f"Epoch {epoch+1} training loss: {avg_loss:.4f}")

        eval_res = eval_mlp()
        print("evaluation:", eval_res)
        eval_loss_history[epoch + 1] = eval_res["loss"]
        eval_auroc_history[epoch + 1] = eval_res["roc_auc"]
        eval_auprc_history[epoch + 1] = eval_res["pr_auc"]

        save_dir = os.path.join(OUTPUT_DIR, f"checkpoint_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        print("Saving to", save_dir)

        try:
            llm_model.save_pretrained(save_dir)
            torch.save(mlp.state_dict(), os.path.join(save_dir, "mlp_state.pt"))
            torch.save(layer_aggregator.state_dict(), os.path.join(save_dir, "mlp_layer_weights.pt"))
            with open(os.path.join(save_dir, "mlp_config.json"), "w") as f:
                json.dump(mlp.config, f, ensure_ascii=False, indent=4)
            with open(os.path.join(save_dir, "mlp_other_data.json"), "w") as f:
                json.dump({
                    "original_repo_id": MODEL_NAME,
                    "binarization_threshold": eval_res["best_threshold"],
                }, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"WARNING: Failed to save checkpoint: {e}")

    test_res = eval_mlp(is_val=False)
    print("test:", test_res)

    print("training_loss_history")
    print(training_loss_history)
    print("eval_loss_history")
    print(eval_loss_history)
    print("eval_auroc_history")
    print(eval_auroc_history)
    print("eval_auprc_history")
    print(eval_auprc_history)

    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    with open(os.path.join(plot_dir, "metrics.json"), "w") as file:
        json.dump({
            "training_loss_history": training_loss_history.to_dict(),
            "eval_loss_history": eval_loss_history.to_dict(),
            "eval_auroc_history": eval_auroc_history.to_dict(),
            "eval_auprc_history": eval_auprc_history.to_dict(),
            "test": test_res,
        }, file, ensure_ascii=False, indent=4)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(training_loss_history.index, training_loss_history.values, label='Training Loss', color='blue')
    ax1.plot(eval_loss_history.index, eval_loss_history.values, label='Eval Loss', color='orange')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.plot(eval_auroc_history.index, eval_auroc_history.values, label='Eval AUROC', color='green')
    ax2.plot(eval_auroc_history.index, eval_auprc_history.values, label='Eval AUPRC', color='red')
    ax2.set_ylabel("AUROC / AUPRC")
    ax2.tick_params(axis='y')
    plt.title("Training & Evaluation Loss and Evaluation AUROC / AUPRC Over Epochs")
    fig.tight_layout()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    ax1.grid(True)
    plt.savefig(os.path.join(plot_dir, "loss_and_aucs.svg"), format='svg')
    plt.close()

    import sys
    sys.exit(0)

from transformer_heads.util.helpers import DataCollatorWithPadding, get_model_params
from transformer_heads import create_headed_qlora
from transformer_heads.config import HeadConfig
from transformer_heads.util.model import print_trainable_parameters
from transformer_heads.output import HeadedModelOutput
from transformer_heads.constants import model_type_map, loss_fct_map
from transformers import Lfm2Model, Qwen3Model, GraniteMoeHybridModel, Gemma3TextModel

model_type_map["qwen3"] = ("model", Qwen3Model)
model_type_map["lfm2"] = ("model", Lfm2Model)
model_type_map["granitemoehybrid"] = ("model", GraniteMoeHybridModel)
model_type_map["gemma3_text"] = ("model", Gemma3TextModel)

model_params = get_model_params(MODEL_NAME)
model_class = model_params["model_class"]
hidden_size = model_params["hidden_size"]
vocab_size = model_params["vocab_size"]

head_perc = 0.5
layer_hook = int(model_params["num_hidden_layers"] * head_perc)
if HEAD_AT_END:
    layer_hook = 1

class Masked_BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # Mask out entries where target < -0.1 (should be -1 and -0.1 is for safety)
        mask = target >= -0.1

        # Avoid computing loss if all entries are masked out
        if not mask.any():
            return torch.tensor(0.0, device=input.device, requires_grad=True)

        return super().forward(input[mask], target[mask])

# Add custom loss to transformer heads lib
loss_fct_map["masked_bce_with_logits"] = Masked_BCEWithLogitsLoss(pos_weight=torch.tensor([ones_weight]).to("cuda"))


if NO_LM_HEAD:
    head_configs = [
        HeadConfig(
            name=head_name,
            layer_hook=-layer_hook,
            in_size=hidden_size,
            hidden_size=1024,
            num_layers=1 if LINEAR else 3,
            output_activation="linear",
            pred_for_sequence=False,
            loss_fct="masked_bce_with_logits",
            num_outputs=1,
        )
    ]
else:
    head_configs = [
        HeadConfig(
            name=head_name,
            layer_hook=-layer_hook,
            in_size=hidden_size,
            hidden_size=1024,
            num_layers=1 if LINEAR else 3,
            output_activation="linear",
            pred_for_sequence=False,
            loss_fct="masked_bce_with_logits",
            num_outputs=1,
        ),
        HeadConfig(
            name="lm_head",
            layer_hook=-1,
            in_size=hidden_size,
            output_activation="linear",
            is_causal_lm=True,
            pred_for_sequence=False,
            loss_fct="cross_entropy",
            num_outputs=vocab_size,
            is_regression=False,
            trainable=False,
        )
    ]

quantization_config = None
if QUANTIZED:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = create_headed_qlora(
    base_model_class=model_class,
    model_name=MODEL_NAME,
    quantization_config=quantization_config,
    lora_config=lora_config,
    head_configs=head_configs,
    fully_trained_heads=True,
    device_map={"": torch.cuda.current_device()},
)

print_trainable_parameters(model)

# Add pad_token to tokenizer
if PAD_TOKEN:
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        model.config.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "right"
        if "reserved" not in PAD_TOKEN:
            model.resize_token_embeddings(len(tokenizer))

collator = DataCollatorWithPadding(
    feature_name_to_padding_value={
        "input_ids": tokenizer.pad_token_id,
        "attention_mask": 0,
    }
)


# Config for fine-tuning
args = SFTConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=4e-5,
    num_train_epochs=1,
    packing=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    save_strategy="epoch",
    save_steps=99,
    max_length=1024,
    logging_strategy="epoch",
    logging_steps=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    gradient_checkpointing_kwargs=dict(use_reentrant=False),
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    train_dataset=train_dataset,
    data_collator=collator
)

# test_dataset = trainer._prepare_dataset_for_kl_loss(test_dataset)

def eval(is_val: bool=True):
    model.eval()
    if is_val:
        loader = DataLoader(val_dataset, batch_size=1, collate_fn=collator)
    else:
        loader = DataLoader(test_dataset, batch_size=1, collate_fn=collator)

    all_probs  = []
    all_labels = []
    losses = []

    for bi, batch in tqdm(
        enumerate(loader), total=len(loader), desc="Evaluating"
    ):
        try:
            outputs: HeadedModelOutput = model(
                **{key: value.to(model.device) for key, value in batch.items()}
            )
            losses.append(float(outputs.loss.item()))

            preds_per_token = outputs.preds_by_head[head_name].flatten()
            
            probs = torch.sigmoid(preds_per_token).cpu().detach().numpy()
            labels = batch[head_name][0, :].cpu().detach().numpy().flatten()

            # Mask out user prompt token (having -1 label)
            mask = labels >= -0.1

            all_probs.extend(list(probs[mask]))
            all_labels.extend(list(labels[mask]))

            del outputs, preds_per_token
        except torch.OutOfMemoryError:
            print(f"OOM Error for batch {bi}")
        del batch

    torch.cuda.empty_cache()

    loss = float(np.mean(losses))

    all_probs_np = np.array(all_probs).flatten()
    all_labels_np = np.array(all_labels, dtype=int).flatten()
    total_roc_auc = roc_auc_score(all_labels_np, all_probs_np)
    total_pr_auc = average_precision_score(all_labels_np, all_probs_np)

    fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_np)
    j_statistic = tpr - fpr
    best_threshold_index = np.argmax(j_statistic)
    best_threshold = thresholds[best_threshold_index]

    result = {
        "loss": loss,
        "roc_auc": float(total_roc_auc),
        "pr_auc": float(total_pr_auc),
        "best_threshold": float(best_threshold),
    }

    return result

training_loss_history = pd.Series()
eval_loss_history  = pd.Series()
eval_auroc_history = pd.Series()
eval_auprc_history = pd.Series()

print("Pre-Training Evaluation")
eval_res = eval()
print("evaluation:", eval_res)
eval_loss_history[0] = eval_res["loss"]
eval_auroc_history[0] = eval_res["roc_auc"]
eval_auprc_history[0] = eval_res["pr_auc"]

for epoch in range(EPOCHS):
    model.train()
    print("EPOCH:", epoch+1)
    out = trainer.train()
    print("training: ", out)
    
    torch.cuda.empty_cache()

    training_loss_history[epoch+1] = out.training_loss

    eval_res = eval()
    print("evaluation:", eval_res)

    eval_loss_history[epoch+1] = eval_res["loss"]
    eval_auroc_history[epoch+1] = eval_res["roc_auc"]
    eval_auprc_history[epoch+1] = eval_res["pr_auc"]

    save_dir = os.path.join(OUTPUT_DIR, f"checkpoint_{epoch+1}")
    print("Saving to", save_dir)

    try:
        model.save_pretrained(save_dir)
    except Exception as e:
        print(f"WARNING: Failed to save checkpoint: {e}")

test_res = eval(is_val=False)
print("test:", test_res)

print("training_loss_history")
print(training_loss_history)
print("eval_loss_history")
print(eval_loss_history)
print("eval_auroc_history")
print(eval_auroc_history)
print("eval_auprc_history")
print(eval_auprc_history)

plot_dir = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Save metrics
with open(os.path.join(plot_dir, "metrics.json"), "w") as file:
    json.dump({
        "training_loss_history": training_loss_history.to_dict(),
        "eval_loss_history": eval_loss_history.to_dict(),
        "eval_auroc_history": eval_auroc_history.to_dict(),
        "eval_auprc_history": eval_auprc_history.to_dict(),
        "test": test_res,
    }, file, ensure_ascii=False, indent=4)


# Save plot
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(training_loss_history.index, training_loss_history.values, label='Training Loss', color='blue')
ax1.plot(eval_loss_history.index, eval_loss_history.values, label='Eval Loss', color='orange')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
ax2.plot(eval_auroc_history.index, eval_auroc_history.values, label='Eval AUROC', color='green')
ax2.plot(eval_auroc_history.index, eval_auprc_history.values, label='Eval AUPRC', color='red')
ax2.set_ylabel("AUROC / AUPRC")
ax2.tick_params(axis='y')
plt.title("Training & Evaluation Loss and Evaluation AUROC / AUPRC Over Epochs")
fig.tight_layout()
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
ax1.grid(True)
plt.savefig(os.path.join(plot_dir, "loss_and_aucs.svg"), format='svg')
plt.close()