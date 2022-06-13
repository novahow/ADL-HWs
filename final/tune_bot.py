import argparse
import json
import random
from regex import R

from transformers.utils.dummy_pt_objects import AdamW
from train_g import get_freer_gpu
import torch
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from find_intent import IntentFinder
from tune_intent import intent_questions
import numpy as np
from transition import Trans
from transformers import (
    AdamW,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)
from evaluate import load
from torch.distributions import Categorical
import torch.nn.functional as F
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
    parser.add_argument("--gamma", type=float, default=1)

    args = parser.parse_args()

    return args


BLEND = load_dataset("blended_skill_talk", split='train')['personas']
MAXLEN=32
with open('keywords.json', 'r') as f:
    _keys = json.load(f)

for k, v in _keys.items():
    _keys[k] = set(v)

def check(example):
    poses = set()
    for i, e in enumerate(example):
        if i < 2:
            continue
        for w in e.split():
            for k, v in _keys.items():
                if w.lower() in v or (w + 's').lower() in v:
                    poses.add(i)
    return poses
def preprocess(example):
    
    example["persona"] = [f"your persona: {p}" for p in example["personas"]]
    example['prevs'] = "\n".join(
        example["persona"]
        + example["previous_utterance"]
    )
    example["context"] = example["previous_utterance"] + (
        [e for t in zip(example['free_messages'], example['guided_messages']) for e in t]
    )

    example['pos'] = check(example["context"])
    return example

def dprep(example):
    if len(example['persona']) == 0:
        example['persona'] = BLEND[np.random.choice(len(BLEND))]
    example["persona"] = [f"your persona: {p}" for p in example["persona"]]
    example['prevs'] = "\n".join(
        example["persona"]
        + example["context"][:2]
    )
    # example['context'] = example['context']
    example['pos'] = check(example["context"])
    return example

class Tune(Dataset):
    def __init__(self, ds, ttok=None) -> None:
        self.ds = ds
        with open('keywords.json', 'r') as f:
            self.keys = json.load(f)

        for k, v in self.keys.items():
            self.keys[k] = set(v)
        self.dds = load_dataset(path='./utils/botd/', data_files={'train': 'gd2.jsonl', 'val': 'dd.jsonl'
                                }, use_auth_token=False)
        self.dds = self.dds.map(dprep, remove_columns=[
            "id",
        ],num_proc=2)
        # self.dds = self.dds.filter(lambda example: len(example['context']) > 0)
        # self.dds = self.dds.remove_columns([col for col in self.dds.column_names if col != "context"])
        self.ds = concatenate_datasets([self.ds, self.dds['train'], self.dds['val']])
        self.rich_ds = self.ds.filter(lambda example: len(example['pos']) > 0)
        self.poor_ds = self.ds.filter(lambda example: len(example['pos']) == 0)
        self.ttok = ttok
        self.ops = {'Transportation': ['Movie', 'Song', 'Restaurant'], 
                    'Hotel': ['Movie', 'Song', 'Restaurant'],
                    'Song': ['Restaurant', 'Transportation', 'Hotel', 'Attraction'],
                    'Movie': ['Restaurant', 'Transportation', 'Hotel', 'Attraction'],
                    'Attraction': ['Movie', 'Song'],
                    'Restaurant': ['Movie', 'Song', 'Transportation', 'Hotel']
                    }
        self.max_len = MAXLEN

        self.rp = 4        
    def __len__(self):
        return len(self.poor_ds) // self.rp + len(self.rich_ds)
    def __getitem__(self, index):
        # if np.random.random() < 0.25:
        #     index += len(self.poor_ds)
        if index >= len(self.poor_ds) // 3:
            index -= (len(self.poor_ds) // 3)
            # index %= len(self.rich_ds)
            
            ut = np.random.choice(list(self.rich_ds[index]['pos']))
            prev_ut = self.rich_ds[index]['context'][max(ut - 4, 0): max(0, ut - 1)]
            label = 1
            tgt = self.rich_ds[index]['context'][ut - 1]
            prevs = self.rich_ds[index]['prevs']
        else:
            p = np.random.choice(self.rp)
            index = min(index * self.rp + p, len(self.poor_ds) - 1)
            ut = np.random.randint(1, len(self.poor_ds[index]['context']))
            prev_ut = self.poor_ds[index]['context'][max(ut - 3, 0): ut]
            label = 0
            tgt = self.poor_ds[index]['context'][ut]
            prevs = self.poor_ds[index]['prevs']
        return prev_ut, label, tgt, prevs
    def collate_fn(self, samples):
        # print('129', samples[np.random.choice(len(samples))])
        scontext = ['</s> <s>'.join(q[0]) for q in samples if q[1] == 1]
        # print('169', scontext)
        rcontext = ['</s> <s>'.join(q[0]) for q in samples if q[1] == 0]
        # assert len(scontext) > 0
        tgt = [q[2] for q in samples]
        sinputs = self.ttok(
            scontext, max_length=self.max_len * 3, truncation=True, return_tensors="pt", padding=True
        ) if len(scontext) else torch.empty(0)
        rinputs = self.ttok(
            rcontext, max_length=self.max_len * 3, truncation=True, return_tensors="pt", padding=True
        ) if len(rcontext) else torch.empty(0)
        tgt = self.ttok(tgt, max_length=self.max_len, truncation=True, return_tensors="pt", padding=True)
        labels = torch.tensor([q[1] for q in samples])
        prevs = [q[-1] for q in samples]
        prevs = self.ttok(
            prevs, max_length=self.max_len * 3, truncation=True, return_tensors="pt", padding=True
        )

        l2 = ['\n'.join(q[0][1:]) for q in samples]
        l2 = self.ttok(
            l2, max_length=self.max_len * 2, truncation=True, return_tensors="pt", padding=True
        )
        return {'su': sinputs, 'rl': rinputs, 'tgt': tgt, 'label': labels, 'prev': prevs, 'l2': l2}        


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device, _ = get_freer_gpu()
    dev_id = int(device.split(':')[-1])
    torch.cuda.set_device(dev_id)
    mname = "facebook/blenderbot-400M-distill"
    # load your bot
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
        num_proc=2
    )
    # device='cpu' 
    bot = AutoModelForSeq2SeqLM.from_pretrained(mname)
    bot_tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    tuner = Tune(dataset, ttok=bot_tokenizer)
    loader = DataLoader(tuner, batch_size=args.batch_size, num_workers=2, collate_fn=tuner.collate_fn, shuffle=True)
    logging_step = 100
    accum_iter = 4
    accelerator = Accelerator(fp16=True)
    device = accelerator.device
    # device='cpu'    
    extra_device = f'cuda:{1-dev_id}'
    simulator = BlenderbotForConditionalGeneration.from_pretrained(mname).to(extra_device)
    simulator_tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    model_id = "gpt2-medium"
    perplexity = load("perplexity", module_type="metric")
    perp = GPT2LMHeadModel.from_pretrained(model_id).to(extra_device)
    ptokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    optimizer = AdamW(bot.parameters(), lr=args.lr)
    # --- fp16 ---
    bot, optimizer, loader = \
        accelerator.prepare(bot, optimizer, loader)
    def get_rew(logits, greedy=False):
        policys = torch.empty((logits.shape[0], 0)).to(device)
        action = torch.empty((logits.shape[0], 0)).to(device)
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

    def strip_rec(l2, bot_tokenizer, gip, nip, prev):
        prevs = bot_tokenizer.batch_decode(l2, skip_special_tokens=True)
        psa = bot_tokenizer.batch_decode(prev, skip_special_tokens=True)
        retg = list()
        retn = list()
        batch_p = list()
        for i, e in enumerate(prevs):
            prevs[i] = e.strip().split('\n')
            batch_p.append(prevs[i])
            # print('258', prevs[i], psa[i], gip[i])
            newc = [psa[i]] + prevs[i] + [gip[i]]
            nc = [psa[i]] + prevs[i] + [nip[i]]
            newc = '</s> <s>'.join(newc)
            retg.append(newc)
            nc = '</s> <s>'.join(nc)
            retn.append(nc)
        return retg, retn, batch_p

    def rew_counter(text):
        stext = text.split()
        cnt = 0
        for w in stext:
            for k, v in _keys.items():
                if w.lower() in v or (w + 's').lower() in v:
                    cnt += 1

        return cnt / len(stext)

    def _perplexity(l2, text):
        max_length = 16
        stride = 5
        ptokenizer.pad_token = ptokenizer.eos_token
        text = ['\n\n'.join(l2[i] + [text[i]]) for i in range(len(text))]
        # print(f'293, {text}')
        encodings = ptokenizer(text, return_tensors="pt", max_length=max_length, padding='longest', truncation=True)
        nlls = []
        for i in (range(0, encodings.input_ids.size(1), stride)):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(extra_device)
            mask = encodings.attention_mask[:, begin_loc:end_loc].to(extra_device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = perp(input_ids, attention_mask=mask, labels=target_ids)
                loss = F.cross_entropy(outputs.logits.transpose(-1, -2), target_ids, reduction='none')
                neg_log_likelihood = loss.mean(-1) * trg_len
            # print('305', neg_log_likelihood.shape)
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls, -1).sum(-1) / end_loc)
        # print('305', ppl.shape)
        return ppl

    for epoch in range(args.num_epoch):
        step = 1
        train_loss = train_acc = 0
        tloss = 0
        # --- train ---
        bot.train()
        simulator.eval()
        pbar = tqdm(loader)
        
        for data in pbar:	
            iloss = loss = -1
            torch.cuda.empty_cache()
            tgt = data['tgt'].input_ids[data['label']==1]
            if tgt.shape[0]:
            # print('271', torch.max(tgt), torch.max(data['su'].input_ids), bot.config.vocab_size, tgt.shape)
                
                output = bot(**(data['su']), labels=tgt)
                # print('85', data, output)
                
                iloss = output.loss / accum_iter 
                
            # backward pass
                accelerator.backward(iloss)
                train_loss += iloss.detach().cpu()
        # weights update
            mask = (data['label']==0)
            tgt = data['tgt'].input_ids[mask]
            if tgt.shape[0]:
                output = bot(**(data['rl']), labels=tgt)
                bout = simulator.generate(input_ids=data['rl'].input_ids.to(extra_device),
                                            attention_mask=data['rl'].attention_mask.to(extra_device))
                greedy_sents, _ = get_rew(output.logits.detach(), greedy=True)
                greedy_sents = bot_tokenizer.batch_decode(greedy_sents.to(device), skip_special_tokens=True)
                sample_sents, policys = get_rew(output.logits)
                action = bot_tokenizer.batch_decode(sample_sents.to(device), skip_special_tokens=True)
                newg, newn, l2 = strip_rec(data['l2']['input_ids'][mask], bot_tokenizer, 
                                            greedy_sents, action, data['prev'].input_ids[mask])
                sim_g = simulator_tokenizer(newg, max_length=MAXLEN * 4, truncation=True, 
                                            return_tensors="pt", 
                                            padding='max_length').to(extra_device)
                sim_n = simulator_tokenizer(newn, max_length=MAXLEN * 4, truncation=True, 
                                            return_tensors="pt", 
                                            padding=True).to(extra_device)
                
                res_g = simulator.generate(**sim_g)
                text_g = simulator_tokenizer.batch_decode(
                    res_g, skip_special_tokens=True
                )
                res_n = simulator.generate(**sim_n)
                text_n = simulator_tokenizer.batch_decode(
                    res_n, skip_special_tokens=True
                )
                text_p = simulator_tokenizer.batch_decode(
                    bout, skip_special_tokens=True
                )
                rewards_b = torch.zeros(tgt.shape[0], policys.shape[-1], device=device)
                rewards_s = torch.zeros(tgt.shape[0], policys.shape[-1], device=device)
                # lab_att_g = torch.tensor(res_g!=bot_tokenizer.pad_token_id)
                # lab_att_n = torch.tensor(res_n!=bot_tokenizer.pad_token_id)
                batch_rs = torch.tensor([rew_counter(e) for e in text_n], device=device)
                # torch.cuda.set_device(1-dev_id)
                rlp = bot_tokenizer.batch_decode(data['rl'].input_ids)
                pscore_s = _perplexity(l2, action).to(device)
                batch_rs -= pscore_s
                batch_rb = torch.tensor([rew_counter(e) for e in text_g], device=device)
                pscore_b = _perplexity(l2, text_p).to(device)
                batch_rb -= pscore_b
                if np.random.random() < 0.03:
                    rd = np.random.choice(tgt.shape[0])
                    print('367', action[rd], text_p[rd])
                    print('368', pscore_s[rd], pscore_b[rd], newn[rd])
                # rewards_b += batch_rb
                # rewards_s += batch_rs
                for p in range(policys.shape[-1] - 1, -1, -1):
                    # implement baseline
                    '''
                    gam_ts_n = torch.zeros((args.batch_size), device=device)
                    gam_ts_g = torch.zeros((args.batch_size), device=device)
                    gam_ts_g[lab_att_g[:, p] == 1] = args.gamma
                    gam_ts_n[lab_att_n[:, p] == 1] = args.gamma
                    gam_ts_n[lab_att_n[:, p] != 1] = 1
                    gam_ts_g[lab_att_g[:, p] != 1] = 1
                    '''
                    
                    rewards_s[:, p] = batch_rs
                    batch_rs *= args.gamma
                    rewards_b[:, p] = batch_rb
                    batch_rb *= args.gamma

                rloss = (rewards_b - rewards_s) * (policys)# * (lab_att_n)
                # print('355', rloss.sum(-1).mean(), output.loss)
                rloss = args.gamma * rloss.sum(-1).mean() + (1 - args.gamma) * output.loss
                loss = rloss / accum_iter
                accelerator.backward(loss)
                tloss += rloss.detach().cpu()
            pbar.set_description(desc=f'iloss:{iloss:.4f}, tloss:{loss:.4f}')
            if ((step) % accum_iter == 0) or (step == len(loader)):
                optimizer.step()
                optimizer.zero_grad()
            step += 1

            ##### TODO: Apply linear learning rate decay #####
            
            # Print training loss and accuracy over past logging step
            if step % logging_step == 0 or step == len(loader):
                lstep = (logging_step) if step % logging_step == 0 else step % logging_step
                print(f"Epoch {epoch + 1} | Step {step} | iloss:{train_loss.item()/lstep:.3f}, tloss:{tloss.item()/lstep:.3f}, acc = {train_acc/lstep:.3f}")
                train_loss = train_acc = tloss = 0
                if step == len(loader):
                    bot.save_pretrained(join(args.ckpt_dir, 'bot.ckpt'))


