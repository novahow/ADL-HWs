from ADLHW3.tw_rouge.tw_rouge import get_rouge
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
from torch.distributions import Categorical
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

class Adv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pred = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Dropout(0.05),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Dropout(0.01),
            nn.Linear(8, 1),
        )
        self.eps = 1e-8
    def forward(self, x):
        return self.pred(x)

    def cal_loss(self, rewards, adv):
        std_r = 0 if torch.any(torch.std(rewards, -1, keepdim=True) == 0).item()\
                    else (rewards - torch.mean(rewards, -1, keepdim=True)) / (torch.std(rewards, -1, keepdim=True))
        act = std_r - adv
        # act = (act - torch.mean(act)) / torch.std(act)
        # rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)
        c_loss = act.pow(2).sum()
        return c_loss, act
        
def main(args):
    prefix = "summarize: "

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    raw_datasets = load_dataset(path='./sumds.py', data_files={'train': '/media/howhow/E_DISK/NTU/junior/adl/mytmp/adl/hw3/data/train.jsonl',
                         'dev': '/media/howhow/E_DISK/NTU/junior/adl/mytmp/adl/hw3/data/public.jsonl'}, use_auth_token=False)

    def preprocess_function(examples):
        # print('43', type(examples))
        inputs = examples["maintext"]
        model_inputs = tokenizer(inputs, max_length=args.max_plen, truncation=True, padding='max_length')

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["title"], max_length=args.max_qlen, truncation=True, padding='max_length')

        model_inputs["labels"] = labels["input_ids"]
        model_inputs['lat'] = labels['attention_mask']
        # print('51', model_inputs)
        return model_inputs

    print('53', raw_datasets)
    raw_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns = raw_datasets["train"].column_names, num_proc=4)
    model = MT5ForConditionalGeneration.from_pretrained(args.model_name)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)#, padding='max_length', max_length=1024)
    train_dl = DataLoader(raw_datasets['train'], batch_size=args.batch_size, collate_fn=data_collator
                                , shuffle=True, drop_last=True, num_workers=4)
    val_dl = DataLoader(raw_datasets['dev'], batch_size=args.val_size, collate_fn=data_collator
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
    
    if args.qa_path != None:
        model = MT5ForConditionalGeneration.from_pretrained(args.qa_path)
    
    # torch.cuda.empty_cache()
    model = model.to(device)
    adv = Adv().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    ado = AdamW(adv.parameters(), lr=5e-4)
    # --- fp16 ---
    model, optimizer, train_dl, val_dl, scheduler =\
        accelerator.prepare(model, optimizer, train_dl, val_dl, scheduler)
    adv, ado = accelerator.prepare(adv, ado)
    tloss = trew = 0
    for epoch in range(args.num_epoch):
        '''
        model.train()
        pbar = tqdm(train_dl)
        for step, batch in enumerate(pbar):
            torch.cuda.empty_cache()
            lab_att = batch.pop('lat')
            outputs = model(**batch, output_hidden_states=True)
            loss = outputs.loss
            policys = torch.empty((args.batch_size, 0)).to(device)
            action = torch.empty((args.batch_size, 0)).to(device)
            head = outputs.decoder_hidden_states[-1]
            head.detach()
            vtheta = torch.empty((args.batch_size, 0)).to(device)
            for idx, logit in enumerate(outputs.logits.transpose(0, 1)):
                d = Categorical(logits=logit)
                act = d.sample()
                vt = adv(head[:, idx, :])
                # print('132', vt.shape, logit.shape, head.shape)
                action = torch.cat([action, act.unsqueeze(-1)], dim=-1)
                policys = torch.cat([policys, d.log_prob(act).unsqueeze(-1)], dim=-1)
                vtheta = torch.cat([vtheta, vt], dim=-1)
            action = tokenizer.batch_decode(action.to(device), skip_special_tokens=True)
            text = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            R = get_rouge(action, text, avg=False)
            # print('106', R)
            vtheta *= lab_att
            batch_r = torch.tensor([e['rouge-l']['f'] for e in R]).to(device)
            rew_mean = batch_r.mean()
            trew += rew_mean
            # implement baseline
            rewards = torch.zeros(args.batch_size, policys.shape[-1])
            # print('108', policys.shape, batch_r.shape)
            for p in range(policys.shape[-1] - 1, -1, -1):
                # implement baseline
                rewards[:, p] = batch_r * lab_att[:, p]
                batch_r *= args.gamma
            
            rewards = rewards.to(device)
            c_loss, vts = adv.cal_loss(rewards, vtheta)
            # rewards = torch.tensor(rewards)
            rloss = torch.sum(vts.to(device) * policys, -1).mean()
            loss = loss - rloss * 0.01 + c_loss * 0.01
            loss = loss / args.g_step
            accelerator.backward(loss)
            tloss += loss.detach()
            if step % args.g_step == 0 or step == len(train_dl) - 1:
                optimizer.step()
                ado.step()
                scheduler.step()
                optimizer.zero_grad()
                ado.zero_grad()

            pbar.set_description(f'loss:{outputs.loss:.3f}|{rloss:.3f}|{c_loss:.3f}, rew:{rew_mean:.3f}')
        # model.save_pretrained(args.ckpt_dir / 't5_1.ckpt')
        # torch.save(adv.state_dict(), args.ckpt_dir / 'adv_1.ckpt')
        tloss /= len(train_dl)
        trew /= len(train_dl)
        print(f'epoch:{epoch}/{args.num_epoch} | loss:{tloss:.3f} | reward:{trew:.3f}')
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
                num_beams=5, 
                no_repeat_ngram_size=2, 
                early_stopping=True,
                num_return_sequences=1
            )
            titles.extend(tokenizer.batch_decode(sample_outputs, skip_special_tokens=True))
            gts.extend(tokenizer.batch_decode(batch['labels'], skip_special_tokens=True))

            res = get_rouge(titles, gts, avg=True)
            r_score.append(res["rouge-l"]["f"])

        avg_score = sum(r_score) / len(r_score)
        if avg_score > best_r:
            model.save_pretrained(args.ckpt_dir / 't5_1.ckpt')
            best_r = avg_score

        print(f'epoch:{epoch}/{args.num_epoch}, rouge-l={avg_score}')
        
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
    parser.add_argument("--val_size", type=int, default=8)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument('--gamma', type=float, default='0.99')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
