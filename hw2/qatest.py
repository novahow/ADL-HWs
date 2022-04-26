import enum
from fileinput import filename
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pickletools import optimize
from re import S
import sched
from typing import Dict
import numpy as np
import torch
from tqdm import trange
from torch.utils.data import DataLoader, Dataset
from os import listdir
from os.path import isfile, join
from dataset import MultDs, QAds, ConDs
from qaut import evaluate, post_proc
import random
import transformers
import torch.nn as nn
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    AdamW,
    AutoModelForQuestionAnswering,
    get_cosine_schedule_with_warmup,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    pipeline
)
from accelerate import Accelerator

import pandas as pd
from datasets import load_dataset, load_metric

from tqdm.auto import tqdm
TRAIN = "train"
DEV = "test"
SPLITS = [DEV]

criterion = nn.CrossEntropyLoss()

def main(args):
    # data_files = {"train": "./data/train.json", "test": "./data/test.json"}
    # dataset = load_dataset('json',  data_files=data_files)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    contextPath = args.context_file
    df = pd.read_json(contextPath)
    # context = tokenizer([e.values[0] for i, e in df.iterrows()], add_special_tokens=False)
    data_paths = {split: args.test_file for split in SPLITS}
    datasets: Dict[str, ConDs] = {
        split: ConDs(tokenizer, filename, df, args, args.batch_size, split)
        for split, filename in data_paths.items()
    }
    

    device = args.device
    # --- fp16 ---
    accelerator = Accelerator(fp16=True)
    device = accelerator.device
    # --- fp16 ---
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name,
    ).to(device)
    
    model.load_state_dict(torch.load(args.qa_path, map_location=device))
    model.eval()
    valid_acc = 0
    result, ids = [], []
    with torch.no_grad():
        for i in tqdm(range(len(datasets[DEV]))):
            outputs = []
            ques = datasets[DEV][i]
            # print('137', ques)
            for e in range(len(ques['input_ids'])):
                opt = model(input_ids=ques['input_ids'][e].unsqueeze(0).to(device), attention_mask=ques['attention_mask'][e].unsqueeze(0).to(device))
                outputs.append(opt)
                
            answer = evaluate(ques, outputs, tokenizer)
            answer = datasets[DEV].contexts[0].values[answer[0]][answer[1]: answer[2]]
            answer = post_proc(answer)
            result.append(answer)
            ids.append(datasets[DEV].data['id'].iloc[i])
            if random.random() < 0.1:
                print(datasets[DEV].data['id'].iloc[i], answer)
            # valid_acc += ((answer) == datasets[DEV].data.loc[i]['answer']["text"])
    with open(args.pred_file, 'w') as f:	
        f.write("id,answer\n")
        for i, test_question in enumerate(result):
            # Replace commas in answers with empty strings (since csv is separated by comma)
            # Answers in kaggle are processed in the same way
            f.write(f"{ids[i]},{test_question.replace(',', '')}\n")

    print(f"Completed! Result is in {args.pred_file}")
            
            
       
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="/tmp2/b08902047/adl/hw1/cache/intent/",
    )
    
    parser.add_argument(
        "--mc_path",
        type=Path,
        help="Path to mc checkpoint.",
        required=False
    )
    
    parser.add_argument(
        "--qa_path",
        type=Path,
        help="Path to qa checkpoint.",
        required=True
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Path to qa checkpoint.",
        default='wptoux/albert-chinese-large-qa'
    )

    # data
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--pred_file", type=Path, default="./pred.csv", required=True)
    parser.add_argument("--context_file", type=Path, default="./data/context.json", required=True)
    
    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--doc_stride", type=int, default=177)
    parser.add_argument("--max_qlen", type=int, default=100)
    parser.add_argument("--max_plen", type=int, default=409)
    
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
