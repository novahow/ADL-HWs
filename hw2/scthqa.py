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
from dataset import ConDs
from qaut import evaluate
import random
import transformers
import torch.nn as nn
from transformers import (
    BertTokenizerFast,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    AdamW,
    BertConfig,
    BertForQuestionAnswering,
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
DEV = "valid"
SPLITS = [TRAIN, DEV]

criterion = nn.CrossEntropyLoss()

def main(args):
    # data_files = {"train": "./data/train.json", "test": "./data/test.json"}
    # dataset = load_dataset('json',  data_files=data_files)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    contextPath = args.data_dir / 'context.json'
    df = pd.read_json(contextPath)
    # context = tokenizer([e.values[0] for i, e in df.iterrows()], add_special_tokens=False)
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    datasets: Dict[str, ConDs] = {
        split: ConDs(tokenizer, filename, df, args, args.batch_size, split)
        for split, filename in data_paths.items()
    }
    
    for k, v in datasets.items():
        continue
        v.run()
    loaders = {split: DataLoader(datasets[split], batch_size=args.batch_size, 
                pin_memory=True, shuffle=(split==TRAIN), collate_fn=datasets[split].collate_fn) 
                    for split in SPLITS}
    device = args.device
    # --- fp16 ---
    accelerator = Accelerator(fp16=True)
    device = accelerator.device
    # --- fp16 ---
    configuration = BertConfig.from_pretrained('./macconf.json')
    model = BertForQuestionAnswering(
        configuration
    ).to(device)
    if args.qa_path != None:
        model.load_state_dict(torch.load(args.qa_path, map_location=device))
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    logging_step = 100
    accum_iter = 4
    # --- fp16 ---
    model, optimizer, loaders[TRAIN] = accelerator.prepare(model, optimizer, loaders[TRAIN])
    # ------------
    scheduler = get_cosine_schedule_with_warmup(optimizer, len(loaders[TRAIN]) * 1 // (20 * accum_iter), 
                                                len(loaders[TRAIN]) * 1 // accum_iter)         
    best_loss, best_acc = float('inf'), float('-inf')
    for epoch in range(args.num_epoch):
        step = 1
        train_loss = train_acc = 0
        vstep = 0
        # --- train ---
        model.train()
        for data in (tqdm(loaders[TRAIN])):	
            output = model(input_ids=data['input_ids'], attention_mask=data['attention_mask'], 
                           token_type_ids=data['token_type_ids'],
                           start_positions=data['sp'], end_positions=data['ep'])
            # print('95', data, output)
            # print('96', data['rel'] == 1)
            tsi, tei = output.start_logits[data['rel'] == 1, :], output.end_logits[data['rel'] == 1, :]
            tsp, tep = data['sp'][data['rel'] == 1], data['ep'][data['rel'] == 1]
            start_index = torch.argmax(tsi, dim=1)
            end_index = torch.argmax(tei, dim=1)
            # print('98', tsi, tei, start_index, end_index, tsp, tep)
            # Prediction is correct only if both start_index and end_index are correct
            if len(start_index):
                train_acc += ((start_index == tsp) & (end_index == tep)).float().mean()
                vstep += 1
            loss = output.loss / accum_iter 
            
        # backward pass
            accelerator.backward(loss)
            train_loss += loss.detach().cpu()
        # weights update
            if ((step) % accum_iter == 0) or (step == len(loaders[TRAIN])):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            step += 1

            ##### TODO: Apply linear learning rate decay #####
            
            # Print training loss and accuracy over past logging step
            if step % logging_step == 0 or step == len(loaders[TRAIN]):
                vstep = max(vstep, 1)
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / vstep:.3f}")
                if train_loss < best_loss:
                    torch.save(model.state_dict(), join(args.ckpt_dir, 'qa{}.ckpt'.format(args.model_name.split('/')[-1])))
                    best_loss = train_loss
                train_loss = train_acc = 0
                vstep = 0
                
        # --- eval ---
        model.eval()
        valid_acc = 0
        with torch.no_grad():
            for i in tqdm(range(len(datasets[DEV]))):
                outputs = []
                ques = datasets[DEV][i]
                # print('137', ques)
                for e in range(len(ques['input_ids'])):
                    opt = model(input_ids=ques['input_ids'][e].unsqueeze(0).to(device), 
                                attention_mask=ques['attention_mask'][e].unsqueeze(0).to(device)
                                , token_type_ids=ques['token_type_ids'][e].unsqueeze(0).to(device))
                    outputs.append(opt)
                    
                answer = evaluate(ques, outputs, tokenizer)
                answer = datasets[DEV].contexts[0].values[answer[0]][answer[1]: answer[2]]
                if random.random() < 0.07:
                    print(datasets[DEV].data['id'].iloc[i], answer)
                valid_acc += ((answer) == datasets[DEV].data.loc[i]['answer']["text"])
        
        # backward pass
            # valid_loss += loss.detach().cpu()
        # weights update
            
        print(f"Epoch {epoch + 1} | Step {step} | loss = 0, acc = {valid_acc / len(datasets[DEV]):.3f}")
        
        torch.save(model.state_dict(), join(args.ckpt_dir, 'qa4{}.ckpt'.format(args.model_name.split('/')[-1])))
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="/tmp2/b08902047/adl/hw1/cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    
    parser.add_argument(
        "--qa_path",
        type=Path,
        help="Directory to save the model file.",
        default=None,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Directory to save the model file.",
        default='hfl/chinese-macbert-base',
    )

    # data
    parser.add_argument("--max_len", type=int, default=512)

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
    parser.add_argument("--batch_size", type=int, default=2)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
