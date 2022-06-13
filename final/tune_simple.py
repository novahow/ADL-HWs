import argparse
import json
import random
from regex import R
from torch.multiprocessing import Pool as ThreadPool
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
    DistilBertTokenizerFast, 
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
    BlenderbotTokenizerFast,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
# from evaluate import load
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from os.path import join
from typing import List
# torch.set_num_threads(4)
torch.autograd.anomaly_mode.set_detect_anomaly(True)
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
    parser.add_argument("--mname", type=str, default="facebook/blenderbot-400M-distill")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=1)

    args = parser.parse_args()

    return args

print('*'*50 + 'finish import' + '*'*50)

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
    if 'persona' not in example.keys():
        example['persona'] = ''
        example['context'] = example['dialog']
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
        self.dd = load_dataset('daily_dialog', split='train')
        self.dd = self.dd.map(dprep, remove_columns=['act', 'emotion'])
        # self.dds = self.dds.filter(lambda example: len(example['context']) > 0)
        # self.dds = self.dds.remove_columns([col for col in self.dds.column_names if col != "context"])
        self.ds = concatenate_datasets([self.ds, self.dds['train'], self.dds['val'], self.dd])
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
        print(len(self.poor_ds), len(self.rich_ds))
        self.rp = 4        
    def __len__(self):
        return (len(self.poor_ds) // self.rp) + len(self.rich_ds)
    def __getitem__(self, index):
        # if np.random.random() < 0.25:
        #     index += len(self.poor_ds)
        if index >= len(self.poor_ds) // self.rp:
            index -= (len(self.poor_ds) // self.rp)
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
        rcontext = ['</s> <s>'.join(q[0]) for q in samples]
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

        l2 = ['\n'.join(q[0][max(0, len(q[0] - 2)):]) for q in samples]
        l2 = self.ttok(
            l2, max_length=self.max_len * 2, truncation=True, return_tensors="pt", padding=True
        )
        return {'su': sinputs, 'rl': rinputs, 'tgt': tgt, 'label': labels, 'prev': prevs, 'l2': l2}        


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rank, _ = get_freer_gpu()
    # device, _ = get_freer_gpu()
    dev_id = int(rank[0])
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
    bot = BlenderbotForConditionalGeneration.from_pretrained(args.mname)
    bot_tokenizer = BlenderbotTokenizerFast.from_pretrained(mname)
    tuner = Tune(dataset, ttok=bot_tokenizer)
    loader = DataLoader(tuner, batch_size=args.batch_size, num_workers=4, 
                            collate_fn=tuner.collate_fn, shuffle=True, pin_memory=True)
    logging_step = 100
    accum_iter = 4
    accelerator = Accelerator(fp16=True)
    device = accelerator.device
    # device='cpu'    
    # device = f'cuda:{rank[1]}'
    simulator = BlenderbotForConditionalGeneration.from_pretrained(mname).to(device)
    simulator_tokenizer = BlenderbotTokenizerFast.from_pretrained(mname)
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    # perplexity = load("perplexity", module_type="metric")
    perp = DistilBertForSequenceClassification.from_pretrained(model_id).to(device)
    ptokenizer = DistilBertTokenizerFast.from_pretrained(model_id)
    optimizer = AdamW(bot.parameters(), lr=args.lr)
    # optimizer = AdamW(list(bot.parameters() + perp.parameters()), lr=args.lr)
    doptimizer = AdamW(perp.parameters(), lr=(args.lr / 10))
    scheduler = get_linear_schedule_with_warmup(optimizer, len(loader) * args.batch_size // (20 * accum_iter), 
                                                    len(loader) * args.batch_size // accum_iter)
    dscheduler = get_linear_schedule_with_warmup(doptimizer, len(loader) * args.batch_size // (20 * accum_iter), 
                                                    len(loader) * args.batch_size // accum_iter)
    # --- fp16 ---
    bot, perp, simulator, optimizer, doptimizer, loader, scheduler, dscheduler = \
        accelerator.prepare(bot, perp, simulator, optimizer, doptimizer, loader, scheduler, dscheduler)
    def get_rew(logits, greedy=False):
        policys = torch.empty((logits.shape[0], 0), dtype=torch.long).to(device)
        action = torch.empty((logits.shape[0], 0), dtype=torch.long).to(device)
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
    
    def s_join(p, psa, g, n):
        p = p.strip().split('\n')
        newc = [psa, *p, g]
        nc = [psa, *p, n]
        newc = '</s> <s>'.join(newc)
        nc = '</s> <s>'.join(nc)
        return newc, nc, p

    def strip_rec(l2, bot_tokenizer, gip, nip, prev):
        prevs = bot_tokenizer.batch_decode(l2, skip_special_tokens=True)
        psa = bot_tokenizer.batch_decode(prev, skip_special_tokens=True)
        retg = list()
        retn = list()
        batch_p = list()
        # print('280', type(prevs), type(psa), type(nip), type(gip), type(prevs[0]), type(psa[0]), type(nip[0]), type(gip[0]))
        with ThreadPool(4) as p:
            all_list = p.starmap(s_join, zip(prevs, psa, gip, nip))
        all_list = np.array(all_list, dtype=object)
        
        batch_p = all_list[:, -1].tolist()
        retg = all_list[:, 0].tolist()
        retn = all_list[:, 1].tolist()
        # print('289', type(retg), type(retn), type(retg[0]), type(retn[0]))
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
        max_length = 32
        stride = 16
        ptokenizer.pad_token = ptokenizer.eos_token
        text = ['\n\n'.join(l2[i] + [text[i]]) for i in range(len(text))]
        # print(f'293, {text}')
        encodings = ptokenizer(text, return_tensors="pt", max_length=perp.config.max_length, padding='longest', truncation=True)
        nlls = []
        for i in (range(0, encodings.input_ids.size(1), stride)):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            mask = encodings.attention_mask[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = perp(input_ids, attention_mask=mask, labels=target_ids)
                # loss = F.cross_entropy(outputs.logits.transpose(-1, -2), target_ids, reduction='none')
                # neg_log_likelihood = loss.mean(-1) * trg_len
                neg_log_likelihood = outputs.loss * trg_len
            # print('305', neg_log_likelihood.shape)
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum(-1) / end_loc)
        # print('305', ppl.shape)
        return ppl

    hitline = torch.tensor([0.1] * min(5, (len(loader) // logging_step)), device=device, dtype=torch.float)
    for epoch in range(args.num_epoch):
        step = 1
        train_loss = train_acc = 0
        tloss = hit = 0
        # --- train ---
        bot.train()
        perp.train()
        simulator.eval()
        pbar = tqdm(loader)
        for data in pbar:	
            iloss = loss = -1
            # torch.cuda.empty_cache()
            tgt = torch.empty(0)
            mask = (data['label']>=0)
            tgt = data['tgt'].input_ids[mask]

            bgt = bot_tokenizer.batch_decode(tgt, skip_special_tokens=True)
            # ------------- G -----------------------#
            perp.eval()
            bot.train()
            output = bot(**(data['rl']), labels=tgt)
            bout = simulator.generate(input_ids=data['rl'].input_ids.to(device),
                                        attention_mask=data['rl'].attention_mask.to(device))
            bout = bot_tokenizer.batch_decode(bout, skip_special_tokens=True)
            greedy_sents, _ = get_rew(output.logits.detach(), greedy=True)
            # print('349', greedy_sents.dtype)
            sample_sents, policys = get_rew(output.logits, greedy=False)
            g_a = bot_tokenizer.batch_decode(torch.cat([greedy_sents, sample_sents], 0).to(device), skip_special_tokens=True)
            greedy_sents = g_a[:greedy_sents.shape[0]]
            action = g_a[-sample_sents.shape[0]:]
            newg, newn, l2 = strip_rec(data['l2']['input_ids'][mask], bot_tokenizer, 
                                        greedy_sents, action, data['prev'].input_ids[mask])
            gen = ['\n'.join([l2[i][-1], action[i]]) for i in range(tgt.shape[0])]
            btext = ['\n'.join([l2[i][-1], bout[i]]) for i in range(tgt.shape[0])]
            bgreed = ['\n'.join([l2[i][-1], greedy_sents[i]]) for i in range(tgt.shape[0])]
            bgt = ['\n'.join([l2[i][-1], bgt[i]]) for i in range(tgt.shape[0])]
            bgen = ptokenizer(btext + gen, max_length=MAXLEN*2, truncation=True, padding=True, return_tensors='pt').to(device)
            logits = F.softmax(perp(**bgen).logits, -1)
            slogits = logits[-len(gen):, :]
            blogits = logits[:len(btext), :]
            
            sim_g_n = simulator_tokenizer(newg + newn, max_length=MAXLEN * 4, truncation=True, 
                                        return_tensors="pt", 
                                        padding=True).to(device)
            
            res_g_n = simulator.generate(**sim_g_n)
            text_g_n = simulator_tokenizer.batch_decode(
                res_g_n, skip_special_tokens=True
            )
            text_g = text_g_n[:len(newg)]
            text_n = text_g_n[-len(newn):]
            mean_len = [len(e) for e in text_n]
                # text_p = simulator_tokenizer.batch_decode(
                #     bout, skip_special_tokens=True
                # )
            if np.random.random() < 0.03:
                bi = np.random.choice(tgt.shape[0])

                print(f'373, newg: {newg[bi]}, samp: {gen[bi]}, slog:{slogits[bi][-1]:.4f}, base: {btext[bi]}, blog:{blogits[bi][-1]:.4f}')
            _rewards_b = blogits[:, -1].detach()#.unsqueeze(-1).repeat(1, policys.shape[-1])
            _rewards_s = slogits[:, -1].detach()#.unsqueeze(-1).repeat(1, policys.shape[-1])
            # lab_att_g = torch.tensor(res_g!=bot_tokenizer.pad_token_id)
            # lab_att_n = torch.tensor(res_n!=bot_tokenizer.pad_token_id)
            _batch_rs = torch.tensor([rew_counter(e) for e in text_n], device=device)#.unsqueeze(-1).repeat(1, policys.shape[-1])
            _batch_rb = hitline.mean().view(1).repeat(tgt.shape[0]) / np.mean(mean_len)#.view(1, 1).repeat(batch_rs.shape[0], policys.shape[-1]) / np.mean(mean_len)
            g_rew_b = torch.zeros(tgt.shape[0], policys.shape[-1], device=device)
            g_rew_s = torch.zeros(tgt.shape[0], policys.shape[-1], device=device)
            h_rew_b = torch.zeros(tgt.shape[0], policys.shape[-1], device=device)
            h_rew_s = torch.zeros(tgt.shape[0], policys.shape[-1], device=device)
            # batch_rb = torch.tensor([rew_counter(e) for e in text_g], device=device)
            # torch.cuda.set_device(1-dev_id)
            loss = F.cross_entropy(output.logits.transpose(-1, -2), tgt, 
                                    # ignore_index=bot_tokenizer.pad_token_id, 
                                    reduction='none').mean(-1)
            masked_loss = loss[data['label']==1]
            if masked_loss.shape[0] == 0:
                masked_loss = torch.zeros(1, device=device)
            ng_loss = loss[data['label']==0]
            if ng_loss.shape[0] == 0:
                ng_loss = torch.zeros(1, device=device)
            # print('350', policys)
            for p in range(policys.shape[-1] - 1, -1, -1):
                h_rew_s[:, p] = _batch_rs
                _batch_rs *= 0.99
                h_rew_b[:, p] = _batch_rb
                _batch_rb *= 0.99
                g_rew_s[:, p] = _rewards_s
                _rewards_s *= 0.99
                g_rew_b[:, p] = _rewards_b
                _rewards_b *= 0.99
            
            # ld = loss.mean() - output.loss
            # print(f'427 ld : {ld:.5f}')
            gloss = ((g_rew_b - g_rew_s) * policys).mean()
            hloss = ((h_rew_b - h_rew_s) * policys).mean()
            rloss = (1 - args.gamma) * (output.loss) + (args.gamma) * masked_loss.mean()#output.loss#
            # print('355', rloss, dloss)
            # rloss = args.gamma * rloss.sum(-1).mean() + (1 - args.gamma) * output.loss
            
            loss = (rloss + (args.gamma / 3) * gloss + (1 - (args.gamma / 3)) * hloss) / accum_iter
            accelerator.backward(loss)
            tloss += loss.detach().cpu()
            # ---------- G -------------#
            # ---------- D -------------#

            perp.train()
            bot.eval()
            
            bgen = ptokenizer(bgreed + bgt, max_length=MAXLEN*2, truncation=True, padding=True, return_tensors='pt').to(device)
            labels = torch.tensor([0] * tgt.shape[0] + [1] * tgt.shape[0], dtype=torch.long, device=device)
            perm = torch.randperm(2 * tgt.shape[0], device=device)
            logits = perp(bgen.input_ids[perm], attention_mask=bgen.attention_mask[perm], labels=labels[perm])
            iloss = logits.loss / accum_iter
            accelerator.backward(iloss)
            train_loss += iloss.detach().cpu()
            hit += ((_batch_rs>0).sum() / args.batch_size)
            pbar.set_description(desc=f'Dloss:{iloss:.4f}, Gloss:{loss:.4f}, hitrate:{((_batch_rs>0).sum()/args.batch_size):.4f}, base:{hitline.mean():.4f}')
            
            if ((step) % accum_iter == 0) or (step == len(loader)):
                optimizer.step()
                scheduler.step()
                doptimizer.step()
                dscheduler.step()
                optimizer.zero_grad()
                doptimizer.zero_grad()
                
            step += 1

            ##### TODO: Apply linear learning rate decay #####
            
            # Print training loss and accuracy over past logging step
            if step % logging_step == 0 or step == len(loader):
                lstep = (logging_step) if step % logging_step == 0 else step % logging_step
                print(f"Epoch {epoch + 1} | Step {step} | Dloss:{train_loss/lstep:.3f}, Gloss:{tloss/lstep:.3f}, hit = {hit/lstep:.3f}")
                _hitline = hitline.clone()
                _hitline[:-1] = hitline[1:]
                _hitline[-1] = (hit/lstep)
                hitline = _hitline.clone()
                train_loss = train_acc = tloss = hit = 0
                if step == len(loader):
                    bot.save_pretrained(join(args.ckpt_dir, 'bot_gan.ckpt'))


