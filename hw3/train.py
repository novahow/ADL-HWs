from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from tqdm import trange
from torch.utils.data import DataLoader, Dataset
from os import listdir
from os.path import isfile, join
import torch.nn as nn
from transformers import BertTokenizerFast
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import (
    DataCollatorForSeq2Seq,
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
    get_linear_schedule_with_warmup,
    MT5ForConditionalGeneration,
    T5Tokenizer
)
from accelerate import Accelerator

import pandas as pd
from datasets import load_dataset, load_metric
# from ADLHW3.tw_rouge.tw_rouge import get_rouge
from tqdm.auto import tqdm
def main(args):
    prefix = "summarize: "

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    raw_datasets = load_dataset(path='./sumds.py', data_files={'train': join(args.data_dir, 'train.jsonl'),
                         'dev': join(args.data_dir, 'public.jsonl')}, use_auth_token=False)

    def preprocess_function(examples):
        # print('43', type(examples))
        inputs = examples["maintext"]
        model_inputs = tokenizer(inputs, max_length=args.max_plen, truncation=True, padding='max_length')

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["title"], max_length=args.max_qlen, truncation=True, padding='max_length')

        model_inputs["labels"] = labels["input_ids"]
        # print('51', model_inputs)
        return model_inputs

    print('53', raw_datasets)
    raw_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns = raw_datasets["train"].column_names, num_proc=2)
    model = MT5ForConditionalGeneration.from_pretrained(args.model_name if args.qa_path==None else args.qa_path)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)#, padding='max_length', max_length=1024)
    train_dl = DataLoader(raw_datasets['train'], batch_size=args.batch_size, collate_fn=data_collator
                                , shuffle=True, num_workers=2)
    val_dl = DataLoader(raw_datasets['dev'], batch_size=args.batch_size, collate_fn=data_collator
                                , shuffle=False)


    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
                                        num_warmup_steps=len(train_dl) * args.num_epoch // (args.g_step * 20),
                                        num_training_steps=len(train_dl) * args.num_epoch // args.g_step)
    
    device = args.device
    # --- fp16 ---
    accelerator = Accelerator(fp16=True)
    device = accelerator.device
    # --- fp16 ---
    # --- fp16 ---
    model, optimizer, train_dl, scheduler = accelerator.prepare(model, optimizer, train_dl, scheduler)
    tloss = 0
    for epoch in range(args.num_epoch):
        model.train()
        pbar = tqdm(train_dl)
        for step, batch in enumerate(pbar):
           
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.g_step
            accelerator.backward(loss)
            tloss += loss.detach()
            if step % args.g_step == 0 or step == len(train_dl) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_description(f'loss:{loss:.3f}')
        model.save_pretrained(args.ckpt_dir / 't5_test.ckpt')
        tloss /= len(train_dl)
        print(f'epoch:{epoch}/{args.num_epoch} | loss:{tloss:.3f}')
        '''
        model.eval()
        titles = []
        gts = []
        r_score = []
        best_r = 0
        for step, batch in enumerate(tqdm(val_dl)):
            sample_outputs = model.generate(
                inputs=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=args.max_qlen, 
                num_beams=10, 
                no_repeat_ngram_size=2, 
                early_stopping=True,
                num_return_sequences=1
            )
            titles.extend([tokenizer.decode(e, skip_special_tokens=True) for e in sample_outputs])
            gts.extend([tokenizer.decode(e, skip_special_tokens=True) for e in batch['labels']])

            res = get_rouge(titles, res)
            r_score.append(res["rouge-l"]["f"])

        avg_score = sum(r_score) / len(r_score)
        if avg_score > best_r:
            model.save_pretrained(args.ckpt_dir / 't5_0.ckpt')
            best_r = avg_score

        print(f'epoch:{epoch}/{args.num_epoch}, rouge-l={avg_score}')
        '''
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
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
        default='google/mt5-small',
    )

    # data
    parser.add_argument("--g_step", type=int, default=4)

    # model
    parser.add_argument("--doc_stride", type=int, default=177)
    parser.add_argument("--max_qlen", type=int, default=50)
    parser.add_argument("--max_plen", type=int, default=512)

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
