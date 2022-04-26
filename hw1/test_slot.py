import enum
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset
from os import listdir
from os.path import isfile, join
from dataset import SeqClsDataset, SlotDataset
from model import SeqClassifier, Slottagger
from utils import Vocab

TASK = 'slot'
def main(args):
    
    slot_idx_path = args.scache_dir / "tag2idx.json"
    slot2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())
    intent_idx_path = args.icache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    sembeddings = torch.load(args.scache_dir / "embeddings.pt")
    iembeddings = torch.load(args.icache_dir / "embeddings.pt")
    tembed = {'slot': sembeddings.to(args.device), 'intent': iembeddings.to(args.device)}
    # print(sembeddings.shape, iembeddings.shape)
    model = Slottagger(
        tembed,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        {'intent': len(intent2idx.keys()), 'slot': len(slot2idx.keys())}
    ).to(args.device)
    

    
    # TODO: predict dataset
    gt, preds, slen = [], [], []
    model.eval()
    if args.sckpt_path != None:
        with open(args.scache_dir / "vocab.pkl", "rb") as f:
            svocab: Vocab = pickle.load(f)

        
        sdata = json.loads(args.stest_file.read_text())
        sdataset = SlotDataset(sdata, svocab, slot2idx, args.max_len)
        # TODO: crecate DataLoader for test dataset
        sloader = DataLoader(sdataset, batch_size=args.batch_size, num_workers=4, collate_fn=sdataset.collate_fn)
        sckpt = torch.load(args.sckpt_path, map_location=args.device)
        # load weights into model
        model.load_state_dict(sckpt)
        for i, batch in enumerate(sloader):
            btext = batch['text'].to(args.device)
            bid = batch['id']
            with torch.no_grad():
                logits = model(btext, TASK)[-1]
                preds.extend(torch.argmax(logits, -1).cpu().tolist())
                gt.extend(bid.tolist())
                slen.extend(batch['len'].tolist())

        with open(args.pred_file, "w") as f:
            # The first row must be "Id, Category"
            f.write("id,tags\n")
            for i, e in enumerate(preds):
                s = ' '.join([sdataset.idx2label(w) for w in e[:slen[i]]])
                f.write('test-{},{}\n'.format(gt[i], s))
    # TODO: write prediction to file (args.pred_file)
    if args.ickpt_path != None:
        with open(args.icache_dir / "vocab.pkl", "rb") as f:
            ivocab: Vocab = pickle.load(f)
        
        idata = json.loads(args.itest_file.read_text())
        idataset = SeqClsDataset(idata, ivocab, intent2idx, args.max_len)
        # TODO: crecate DataLoader for test dataset
        iloader = DataLoader(idataset, batch_size=args.batch_size, num_workers=4, collate_fn=idataset.collate_fn)
        
        ickpt = torch.load(args.ickpt_path, map_location=args.device)
        # load weights into model
        model.load_state_dict(ickpt)
        # TODO: predict dataset
        gt, preds, slen = [], [], []
        model.eval()
        for i, batch in enumerate(iloader):
            btext = batch['text'].to(args.device)
            bid = batch['id']
            with torch.no_grad():
                logits = model(btext, 'intent')[0]
                preds.extend(torch.argmax(logits, -1).cpu().tolist())
                gt.extend(bid.tolist())


        with open(args.pred_file, "w") as f:
            # The first row must be "Id, Category"
            f.write("id,intent\n")
            for i, e in enumerate(preds):
                f.write('test-{},{}\n'.format(gt[i], idataset.idx2label(e)))
    # TODO: write prediction to file (args.pred_file)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--itest_file",
        type=Path,
        help="Path to the test file.",
    )
    parser.add_argument(
        "--stest_file",
        type=Path,
        help="Path to the test file.",
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
        "--ickpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default=None
    )
    parser.add_argument(
        "--sckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default=None
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
