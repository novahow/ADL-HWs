import sys
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
import faulthandler

faulthandler.enable()
import pandas as pd
from datasets import load_dataset, load_metric
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
        model_inputs['lat'] = labels['attention_mask']
        # print('51', model_inputs)
        return model_inputs

    print('53', raw_datasets)
    raw_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns = raw_datasets["train"].column_names, num_proc=4)
    


    print('126', args.device, args.fp)
    device = args.device
    if args.fp:
        # --- fp16 ---
        accelerator = Accelerator(fp16=True)
        device = accelerator.device
        # --- fp16 ---
    model = MT5ForConditionalGeneration.from_pretrained(args.model_name if args.qa_path==None else args.qa_path)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)#, padding='max_length', max_length=1024)
    train_dl = DataLoader(raw_datasets['train'], batch_size=args.batch_size, collate_fn=data_collator
                                , shuffle=True, drop_last=True, num_workers=2)
    val_dl = DataLoader(raw_datasets['dev'], batch_size=args.val_size, collate_fn=data_collator
                                , shuffle=False)
    # adv = Adv().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
                                        num_warmup_steps=len(train_dl) * args.num_epoch // (args.g_step * 20),
                                        num_training_steps=len(train_dl) * args.num_epoch // args.g_step)
    

    # ado = AdamW(adv.parameters(), lr=5e-4)
    # --- fp16 ---
    if args.fp:
        model, optimizer, train_dl, val_dl, scheduler =\
            accelerator.prepare(model, optimizer, train_dl, val_dl, scheduler)
    # adv, ado = accelerator.prepare(adv, ado)
    
    
    def get_rew(logits, greedy=False):
        policys = torch.empty((args.batch_size, 0)).to(device)
        action = torch.empty((args.batch_size, 0)).to(device)
        for idx, logit in enumerate(logits.transpose(0, 1)):
            if not greedy:
                d = Categorical(logits=logit)
                act = d.sample()
                action = torch.cat([action, act.unsqueeze(-1)], dim=-1)
                policys = torch.cat([policys, d.log_prob(act).unsqueeze(-1)], dim=-1)
            else:
                act = logit.argmax(-1)
                action = torch.cat([action, act.unsqueeze(-1).detach()], dim=-1)
        
        return action, policys
    
    best_rew = args.init_rew
    for epoch in range(args.num_epoch):
        tloss = trew = rtloss = 0
        model.train()
        pbar = tqdm(train_dl)
        for step, batch in enumerate(pbar):
            torch.cuda.empty_cache()
            if not args.fp:
                for e in batch.keys():
                    batch[e] = batch[e].to(device)
            lab_att = batch.pop('lat')
            
            outputs = model(**batch, output_hidden_states=True)
            # print('131', batch)
            loss = outputs.loss
            
            with torch.no_grad():
                greedy_sents, _ = get_rew(outputs.logits, greedy=True)
            greedy_sents = tokenizer.batch_decode(greedy_sents.to(device), skip_special_tokens=True)
            text = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            lens = np.array([len(e) for e in greedy_sents])
            # assert np.any(lens==0) == False
            Rbase = get_rouge(greedy_sents, text, avg=False, ignore_empty=True)# if not np.any(lens==0) else 0
            batch_rb = torch.tensor([(e['rouge-l']['f'] + e['rouge-1']['f'] + e['rouge-2']['f']) for e in Rbase], device=device)# \
                #if not np.any(lens==0) else torch.zeros(args.batch_size, device=device) 
            fixb = torch.zeros(args.batch_size, device=device)
            fixb[lens!=0] = batch_rb
            batch_rb = fixb
            rew_mean = batch_rb.mean()
            # print('108', policys.shape, batch_r.shape)
            # print('161', rewards_s[:, p], rewards_b[:, p])
            
            # print('135', greedy_sents, text, lens, batch_rb, rew_mean, lens.any(0))
            trew += rew_mean
            rloss = 0
            
            if best_rew >= 0.05:
                sample_sents, policys = get_rew(outputs.logits)
                action = tokenizer.batch_decode(sample_sents.to(device), skip_special_tokens=True)
                Rsamp = get_rouge(action, text, avg=False, ignore_empty=True)
                batch_rs = torch.tensor([(e['rouge-l']['f'] + e['rouge-1']['f'] + e['rouge-2']['f']) for e in Rsamp], device=device)
                # implement baseline
                lens = np.array([len(e) for e in action])
                fixs = torch.zeros(args.batch_size, device=device)
                fixs[lens!=0] = batch_rs
                batch_rs = fixs
                rewards_b = torch.zeros(args.batch_size, policys.shape[-1], device=device)
                rewards_s = torch.zeros(args.batch_size, policys.shape[-1], device=device)
                for p in range(policys.shape[-1] - 1, -1, -1):
                    # implement baseline
                    gam_ts = torch.zeros((args.batch_size), device=device)
                    gam_ts[lab_att[:, p] == 1] = args.gamma
                    gam_ts[lab_att[:, p] != 1] = 1
                    
                    rewards_s[:, p] = batch_rs
                    batch_rs *= gam_ts
                    rewards_b[:, p] = batch_rb
                    batch_rb *= gam_ts
                # rewards_b += batch_rb.unsqueeze(-1)
                # rewards_s += batch_rs.unsqueeze(-1)
                # rewards = torch.tensor(rewards)
                # print('180', rewards_b, rewards_s, policys)
                rloss = (rewards_b - rewards_s) * (policys * lab_att)
                rloss = rloss.sum(-1).mean()
                loss = loss * (1 - args.rl_prop) + rloss * args.rl_prop        
                rtloss += rloss.detach()
            
            loss = loss / args.g_step
            if args.fp:
                accelerator.backward(loss)
            else:
                loss.backward()
            tloss += outputs.loss.detach()
            if step % args.g_step == 0 or step == len(train_dl) - 1:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            pbar.set_description(f'loss:{outputs.loss:.3f}|{rloss:.3f}, rew:{rew_mean:.3f}')
        model.save_pretrained(args.ckpt_dir / 't5_1.ckpt')
        tloss /= len(train_dl)
        rtloss /= len(train_dl)
        trew /= len(train_dl)
        best_rew = max(best_rew, trew)
        print(f'epoch:{epoch}/{args.num_epoch} | loss:{tloss:.3f}|{rtloss:.3f} | reward:{trew:.3f}')
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
    parser.add_argument("--val_size", type=int, default=8)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument('--gamma', type=float, default='0.99')
    parser.add_argument('--rl_prop', type=float, default='0.98')
    parser.add_argument('--fp', type=int, default=1)
    parser.add_argument('--init_rew', type=float, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
