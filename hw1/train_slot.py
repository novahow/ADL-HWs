import enum
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from re import S
from statistics import mode
from typing import Dict
import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from os import listdir
from os.path import isfile, join
from dataset import SeqClsDataset, SlotDataset
from model import SeqClassifier, Slottagger
from joint import Encoder, Decoder
from utils import Vocab
import random
from sched import get_linear_schedule_with_warmup

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
INTENT = 'intent'
SLOT = 'slot'
TASK = [INTENT, SLOT]

def calacc(pred, gt):
    amax = torch.argmax(pred, -1)
    return torch.eq(amax, gt).sum()

def jacc(pred, gt, slen):

    tac = 0
    for i, e in enumerate(gt):
        spred = torch.argmax(pred[i][:slen[i]], -1)
        tac += (torch.eq(spred, e[:slen[i]]).sum() // slen[i])
    
    return tac
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def main(args):
    with open(args.scache_dir / "vocab.pkl", "rb") as f:
        svocab: Vocab = pickle.load(f)

    with open(args.icache_dir / "vocab.pkl", "rb") as f:
        ivocab: Vocab = pickle.load(f)

    slot_idx_path = args.scache_dir / "tag2idx.json"
    slot2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())
    intent_idx_path = args.icache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    intentDpaths = {split: args.idata_dir / f"{split}.json" for split in SPLITS}
    iData = {split: json.loads(path.read_text()) for split, path in intentDpaths.items()}
    iDatasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, ivocab, intent2idx, args.max_len)
        for split, split_data in iData.items()
    }
    slotDpaths = {split: args.sdata_dir / f"{split}.json" for split in SPLITS}
    sData = {split: json.loads(path.read_text()) for split, path in slotDpaths.items()}
    sDatasets: Dict[str, SlotDataset] = {
        split: SlotDataset(split_data, svocab, slot2idx, args.max_len)
        for split, split_data in sData.items()
    }
    criterion = torch.nn.CrossEntropyLoss()
    iLoaders = {split: DataLoader(iDatasets[split], batch_size=args.batch_size, 
                num_workers=4, collate_fn=iDatasets[split].collate_fn, shuffle=True) 
                    for split in SPLITS}
    sLoaders = {split: DataLoader(sDatasets[split], batch_size=args.batch_size, 
                num_workers=4, collate_fn=sDatasets[split].collate_fn, shuffle=True) 
                    for split in SPLITS}
    # TODO: crecate DataLoader for train / dev datasets

    iembeddings = torch.load(args.icache_dir / "embeddings.pt")
    sembeddings = torch.load(args.scache_dir / "embeddings.pt")
    
    tloader = {INTENT: iLoaders, SLOT: sLoaders}
    tembed = {INTENT: iembeddings.to(args.device), SLOT: sembeddings.to(args.device)}
    tclass = {INTENT: len(intent2idx.keys()), SLOT: len(slot2idx.keys())}
    model = Slottagger(tembed, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, tclass).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    totl = sum(len(tloader[e][TRAIN]) for e in TASK)
    scheduler = get_linear_schedule_with_warmup(optimizer, totl * args.num_epoch // 20,
                                                     totl * args.num_epoch)
    
    tbestacc = {task: 0 for task in TASK}
    for epoch in (range(args.num_epoch)):
        for task, loader in tloader.items():
            losses = []
            tacc, tsum = 0, 0
            model.train()
            for batch in tqdm(loader[TRAIN]):
                optimizer.zero_grad()
                btext = batch['text'].to(args.device)
                bgt = batch[task].to(args.device)
                # print(btext, bgt, task)
                logits = model(btext, task)[int(task[0] != 'i')]
                # print(bgt.shape, logits.shape)
                loss = criterion(logits, bgt) if task[0] == 'i' \
                            else criterion(logits.permute(0, 2, 1), bgt)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if(task[0] == 'i'):
                    acc = calacc(logits, bgt)
                    tacc += acc.detach().cpu()
                    tsum += bgt.shape[0]
                    losses.append(loss.detach().cpu().item())
                else:
                    
                    acc = jacc(logits, bgt, batch['len'])
                    tacc += acc.detach().cpu()
                    tsum += bgt.shape[0]
                    losses.append(loss.detach().cpu().item())
            print('Train {} | Epoch: {:3d}, Acc: {:.5f}, Loss: {:.5f}'.format(task, epoch, tacc / tsum, np.mean(losses)))

            model.eval()
            for batch in tqdm(loader[DEV]):
                with torch.no_grad():
                    btext = batch['text'].to(args.device)
                    bgt = batch[task].to(args.device)
                    logits = model(btext, task)[int(task[0] != 'i')]
                    # print(bgt.shape, logits.shape)
                    loss = criterion(logits, bgt) if task[0] == 'i' \
                                else criterion(logits.permute(0, 2, 1), bgt)
                    if(task[0] == 'i'):
                        acc = calacc(logits, bgt)
                        tacc += acc.detach().cpu()
                        tsum += bgt.shape[0]
                        losses.append(loss.detach().cpu().item())
                    else:
                        acc = jacc(logits, bgt, batch['len'])
                        tacc += acc.detach().cpu()
                        tsum += bgt.shape[0]
                        losses.append(loss.detach().cpu().item())
            if (tacc / tsum) > tbestacc[task]:
                torch.save(model.state_dict(), join(args.ckpt_dir, '{}{}.ckpt'.format(task[0], args.v)))
                tbestacc[task] = tacc / tsum
            print('Valid {} | Epoch: {:3d}, Acc: {:.5f}, Loss: {:.5f}'.format(task, epoch, tacc / tsum, np.mean(losses)))






def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--sdata_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--idata_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--scache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="/tmp2/b08902047/adl/hw1/cache/slot/",
    )
    parser.add_argument(
        "--icache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="/tmp2/b08902047/adl/hw1/cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    parser.add_argument("--v", type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
