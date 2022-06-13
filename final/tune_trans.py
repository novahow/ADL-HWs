import argparse
import json
import random
from train_g import get_freer_gpu
import torch
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from find_intent import IntentFinder
from tune_intent import intent_questions
import numpy as np
from transition import Trans
from train_g import get_freer_gpu
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from os.path import join
from typing import List
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="facebook/blenderbot-400M-distill",
        type=str,
        help="model to chat with simulator",
    )

    parser.add_argument("--num_chats", default=5, type=int, help="the number of round")

    parser.add_argument("--split", default="train", type=str, help="split")

    parser.add_argument("--seed", default=26, type=int, help="random seed")

    parser.add_argument(
        "--interactive_mode",
        action="store_true",
        help="make the simualtor interact with the user (type 'stop' to stop the program, type 'exit' to leave current round)",
    )

    parser.add_argument(
        "--output",
        default="output.jsonl",
        type=str,
        help="file to save the dialogs",
    )

    parser.add_argument(
        "--disable_output_dialog",
        action="store_true",
        help="whether output the dialogs to the command line",
    )
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--input_dir", type=str, default="blender.jsonl")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epoch", type=int, default=5)

    args = parser.parse_args()

    return args

with open('keywords.json', 'r') as f:
    keys = json.load(f)

for k, v in keys.items():
    keys[k] = set(v)

def check(example):
    for e in example:
        for w in e.split():
            for k, v in keys.items():
                if w.lower() in v or (w + 's').lower() in v:
                    return 1
    return 0
def preprocess(example):

    example["personas"] = [f"your persona: {p}" for p in example["personas"]]
    example["context"] = (
        example["previous_utterance"] +
        [e for t in zip(example['free_messages'], example['guided_messages']) for e in t]
    )

    if check(example['context']):
        return example
    
    example['context'] = []
    return example

def dprep(example):
    if check(example['context']['dialog']):
        example['context'] = example['context']['dialog']
        return example
    
    example['context'] = []
    return example

class Tune(Dataset):
    def __init__(self, ds, itok, ttok) -> None:
        self.ds = ds.filter(lambda example: len(example['context']) > 0)
        self.ds = self.ds.remove_columns([col for col in self.ds.column_names if col != "context"])
        with open('keywords.json', 'r') as f:
            self.keys = json.load(f)

        for k, v in self.keys.items():
            self.keys[k] = set(v)
        self.dds = load_dataset(path='./utils/botd/', data_files={'train': 'gd2.jsonl',
                                }, use_auth_token=False)['train']
        self.dds = self.dds.map(dprep, remove_columns=[
            "id",
        ],)
        self.dds = self.dds.filter(lambda example: len(example['context']) > 0)
        # self.dds = self.dds.remove_columns([col for col in self.dds.column_names if col != "context"])
        self.ds = concatenate_datasets([self.ds, self.dds])
        self.itok = itok
        self.ttok = ttok
        self.ops = {'Transportation': ['Movie', 'Song', 'Restaurant'], 
                    'Hotel': ['Movie', 'Song', 'Restaurant'],
                    'Song': ['Restaurant', 'Transportation', 'Hotel', 'Attraction'],
                    'Movie': ['Restaurant', 'Transportation', 'Hotel', 'Attraction'],
                    'Attraction': ['Movie', 'Song'],
                    'Restaurant': ['Movie', 'Song', 'Transportation', 'Hotel']
                    }
        self.max_len = 32

        
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, index):
        ut = np.random.randint(2, len(self.ds[index]['context']))
        skey, pkey = None, None
        # print('103', self.ds[index]['context'][ut])
        sp = self.ds[index]['context'][ut].split()
        for e in sp:
            for k, v in self.keys.items():
                if e in v:
                    skey = k
                    break

        sp = self.ds[index]['context'][ut - 2].split()
        for e in sp:
            for k, v in self.keys.items():
                if e in v:
                    pkey = k
                    break
        
        sent = ' '.join(sp)
        label = -1
        cq = ''

        if skey != None:
            skey = skey[0].upper() + skey[1:].lower()
            label = 1
            cq = np.random.choice(intent_questions[skey])
        else:
            label = 0
            if pkey != None:
                pkey = pkey[0].upper() + pkey[1:].lower()
                if np.random.random() < 0.3:
                    label = 1
                    cq = np.random.choice(intent_questions[pkey])
                else:
                    op_class = np.random.choice(list(self.ops[pkey]))
                    cq = np.random.choice(intent_questions[op_class])
            else:
                op_class = np.random.choice(list(self.ops.keys()))
                cq = np.random.choice(intent_questions[op_class])
            
        
        return self.ds[index]['context'][ut], self.ds[index]['context'][max(0, ut-5):ut-1], self.ds[index]['context'][ut - 1], cq, label
    def collate_fn(self, samples):
        # print('129', samples[np.random.choice(len(samples))])
        qs = [e[-2] for e in samples]
        context = [f'yes. no. {q[1][-1]}' for q in samples]
        tgt = [e[2] for e in samples]
        examples = [(
                f"<context> {' '.join(e[1])} </context> <blank> <future> {e[0]} </future>"
            ) for e in samples]
        inputs = self.ttok(
            examples, max_length=self.max_len * 4, truncation=True, return_tensors="pt", padding='max_length'
        )
        tgt = self.ttok(tgt, max_length=self.max_len, truncation=True, return_tensors="pt", padding='max_length')
        qc = []
        for q, c in zip(qs, context):
            qc.append((q, c))
        qc = self.itok(qc, padding='max_length', max_length=self.max_len * 2, return_tensors='pt', truncation='only_second')
        # print('192', qc[0])
        sps = torch.tensor([e.char_to_token(0 if samples[i][-1] else 5, sequence_index=1) for i, e in enumerate(qc._encodings)])
        eps = torch.tensor([e.char_to_token(3 if samples[i][-1] else 7, sequence_index=1) for i, e in enumerate(qc._encodings)])
        
        return {'qc': qc, 'sp': sps, 'ep': eps, 'tr': inputs, 'tgt': tgt}        


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device, _ = get_freer_gpu()
    mname = "facebook/blenderbot-400M-distill"
    # load your bot
    intentchecker = IntentFinder(int(device.split(':')[-1]), ckpt='./ckpt/qa_g.ckpt')
    transer = Trans(int(device.split(':')[-1]))
    dataset = load_dataset("blended_skill_talk", split=args.split)
    dataset = dataset.map(
        preprocess,
        remove_columns=[
            "free_messages",
            "guided_messages",
            "suggestions",
            "personas",
            "additional_context",
            "previous_utterance",
        ],
    )

    tuner = Tune(dataset, intentchecker.tokenizer, transer.tokenizer)
    intentchecker.model.train()
    transer.model.train()
    loader = DataLoader(tuner, batch_size=args.batch_size, num_workers=2, collate_fn=tuner.collate_fn, shuffle=True)
    ioptimizer = torch.optim.AdamW(intentchecker.model.parameters(), lr=args.lr)
    toptimizer = torch.optim.AdamW(transer.model.parameters(), lr=args.lr)
    logging_step = 100
    accum_iter = 4
    accelerator = Accelerator(fp16=True)
    device = accelerator.device    
    # --- fp16 ---
    intentchecker.model, transer.model, ioptimizer, toptimizer, loader = \
        accelerator.prepare(intentchecker.model, transer.model, ioptimizer, toptimizer, loader)
    for epoch in range(args.num_epoch):
        step = 1
        train_loss = train_acc = 0
        tloss = 0
        # --- train ---
        intentchecker.model.train()
        transer.model.train()
        pbar = tqdm(loader)
        for data in pbar:	
            output = intentchecker.model(**(data['qc']) ,
                            start_positions=data['sp'], end_positions=data['ep'])
            # print('85', data, output)
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)
            
            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == data['sp']) & (end_index == data['ep'])).float().mean()
            iloss = output.loss / accum_iter 
            
        # backward pass
            accelerator.backward(iloss)
            train_loss += iloss.detach().cpu()
        # weights update
            output = transer.model(**data['tr'], labels=data['tgt']['input_ids'])
            loss = output.loss / accum_iter
            accelerator.backward(loss)
            tloss += loss.detach().cpu()
            pbar.set_description(desc=f'iloss:{iloss:.4f}, tloss:{loss:.4f}')
            if ((step) % accum_iter == 0) or (step == len(loader)):
                ioptimizer.step()
                ioptimizer.zero_grad()
                toptimizer.step()
                toptimizer.zero_grad()
            step += 1

            ##### TODO: Apply linear learning rate decay #####
            
            # Print training loss and accuracy over past logging step
            if step % logging_step == 0 or step == len(loader):
                lstep = (logging_step) if step % logging_step == 0 else step % logging_step
                print(f"Epoch {epoch + 1} | Step {step} | iloss:{train_loss.item()/lstep:.3f}, tloss:{tloss.item()/lstep:.3f}, acc = {train_acc/lstep:.3f}")
                train_loss = train_acc = tloss = 0
                if step == len(loader):
                    intentchecker.model.save_pretrained(join(args.ckpt_dir, 'qa_b.ckpt'))
                    transer.model.save_pretrained(join(args.ckpt_dir, 'tr_b.ckpt'))


