import enum
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from re import S
from typing import Dict
import numpy as np
import torch
from tqdm import trange
from torch.utils.data import DataLoader, Dataset
from os import listdir
from os.path import isfile, join
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
import random
from sched import get_linear_schedule_with_warmup

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def calacc(pred, gt):
    amax = torch.argmax(pred, -1)
    return torch.eq(amax, gt).sum()

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    criterion = torch.nn.CrossEntropyLoss()
    loaders = {split: DataLoader(datasets[split], batch_size=args.batch_size, 
                num_workers=4, collate_fn=datasets[split].collate_fn, shuffle=True) 
                    for split in SPLITS}
    # TODO: crecate DataLoader for train / dev datasets

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, 
                            args.bidirectional, 150).to(args.device)
    # model = torch.nn.DataParallel(model, device_ids=[2, 3])
    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, len(loaders[TRAIN]) * args.num_epoch // 20,
                                                     len(loaders[TRAIN]) * args.num_epoch)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_acc = 0
    for epoch in epoch_pbar:
        tacc, tsum = 0, 0
        losses = []
        model.train()
        for i, batch in enumerate(loaders[TRAIN]):
            # print(batch)
            optimizer.zero_grad()
            btext = batch['text'].to(args.device)
            bint = batch['intent'].to(args.device)
            logits = model(btext)
            loss = criterion(logits, bint)
            loss.backward()
            optimizer.step()
            scheduler.step()
            acc = calacc(logits, bint)
            tacc += acc.detach().cpu()
            tsum += bint.shape[0]
            losses.append(loss.detach().cpu().item())
        
        print('Train | Epoch: {:3d}, Acc: {:.5f}, Loss: {:.5f}'.format(epoch, tacc / tsum, np.mean(losses)))

        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        tacc, tsum = 0, 0
        losses = []
        model.eval()
        for i, batch in enumerate(loaders[DEV]):
            btext = batch['text'].to(args.device)
            bint = batch['intent'].to(args.device)
            with torch.no_grad():
                logits = model(btext)
                loss = criterion(logits, bint)
                acc = calacc(logits, bint)
                tacc += acc.detach().cpu()
                tsum += bint.shape[0]
                losses.append(loss.detach().cpu().item())

        if (tacc / tsum) > best_acc:
            torch.save(model.state_dict(), join(args.ckpt_dir, 'i1.ckpt'))
            best_acc = tacc / tsum
        print('Valid | Epoch: {:3d}, Acc: {:.5f}, Loss: {:.5f}'.format(epoch, tacc / tsum, np.mean(losses)))
        pass


    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
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
        default="./ckpt/intent/",
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
