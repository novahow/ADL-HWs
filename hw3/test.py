from transformers import AutoTokenizer
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
import json
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

from tqdm.auto import tqdm

decode_configs = {
    'beam': {'num_return_sequences': 1, 
             'num_beams':5, 
            'no_repeat_ngram_size':2, 
            'early_stopping':True,
            'num_return_sequences':1,
            # 'temperature': 0.3,
            },
    'temp': {'do_sample':True,
            'top_k':0, 
            'temperature':0.3,
            'num_return_sequences': 1
            },
    'topk': {'do_sample':True,
             'top_k': 25,
             'num_return_sequences': 1
             },
    'topp': {'do_sample':True,
             'top_p': 0.98,
             'top_k': 0,
             'num_return_sequences': 1
             },
    # 'toppk':{'do_sample':True,
    #          'top_p': 0.92,
    #          'top_k': 50,
    #          'num_return_sequences': 1
    #          }
}

def main(args):
    prefix = "summarize: "

    tokenizer = T5Tokenizer.from_pretrained(join(args.ckpt_dir, 'tokenizer.ckpt'))
    raw_datasets = load_dataset(path='./sumds.py', data_files={
                         'dev': args.test_file}, use_auth_token=False)

    def preprocess_function(examples):
        # print('43', type(examples))
        inputs = examples["maintext"]
        model_inputs = tokenizer(inputs, max_length=args.max_plen, truncation=True, padding='max_length')

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["title"], max_length=args.max_qlen, truncation=True, padding='max_length')

        model_inputs["labels"] = labels["input_ids"]
        model_inputs['aid'] = examples['id']
        # print('51', model_inputs)
        return model_inputs

    print('53', raw_datasets)
    gts = raw_datasets['dev']['title']
    raw_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns = raw_datasets["dev"].column_names, num_proc=2)

    

    device = args.device
    # --- fp16 ---
    accelerator = Accelerator(fp16=True)
    device = accelerator.device
    # --- fp16 ---
    
    if args.qa_path != None:
        model = MT5ForConditionalGeneration.from_pretrained(args.qa_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)#, padding='max_length', max_length=1024)
    val_dl = DataLoader(raw_datasets['dev'], batch_size=args.batch_size, collate_fn=data_collator
                                , shuffle=False, num_workers=4)
    model = model.to(device)
    
    # --- fp16 ---
    model, val_dl = accelerator.prepare(model, val_dl)
    model.eval()
    
    
    cfgs = decode_configs.keys() if args.useall else [args.decode]
    for k in cfgs:
        ids = []
        titles = []
        for step, batch in enumerate(tqdm(val_dl)):
            # print(batch)
            '''
            sample_outputs = model.generate(
                inputs=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=args.max_qlen, 
                num_beams=5, 
                no_repeat_ngram_size=2, 
                early_stopping=True,
                num_return_sequences=1
            )'''
            sample_outputs = model.generate(
                inputs=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=args.max_qlen, 
                min_length=15,
                **(decode_configs[k])
            )

            ids.extend(batch['aid'].cpu().numpy().tolist())
            # print('91', type(sample_outputs))
            titles.extend(tokenizer.batch_decode(sample_outputs, skip_special_tokens=True))
            # gts.extend(tokenizer.batch_decode(batch['labels'], skip_special_tokens=True))
            # print('93', titles[-1], gts[-1])
        
        if args.gt != None:
            with open(args.gt, "w", encoding="utf8") as fp:
                for p, idx in enumerate(gts):
                    json.dump({"id":ids[p], "title":idx}, fp, ensure_ascii = False)
                    fp.write("\n")
            
        pred_file = args.pred
        if args.useall:
            pred_file, ft = str(args.pred).split('.')
            pred_file = f'{pred_file}_{k}.{ft}'
        with open(f'{pred_file}', "w", encoding="utf8") as fp:
            for p, idx in enumerate(titles):
                json.dump({"id":str(ids[p]), "title":idx}, fp, ensure_ascii = False)
                fp.write("\n")
            
    
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
    parser.add_argument("--useall", type=int, default=0)

    # model
    parser.add_argument("--doc_stride", type=int, default=177)
    parser.add_argument("--max_qlen", type=int, default=50)
    parser.add_argument("--max_plen", type=int, default=409)
    parser.add_argument("--decode", type=str, default='beam')

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=2)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--test_file", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
