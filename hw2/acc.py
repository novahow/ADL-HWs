import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path

parser = ArgumentParser()
parser.add_argument(
    "--pred",
    type=Path,
    help="Directory to the dataset.",
    default="./data/",
)
parser.add_argument(
    "--gt",
    type=Path,
    help="Directory to the dataset.",
    default="./data/",
)
args = parser.parse_args()
pred = pd.read_csv(args.pred)
gt = pd.read_json(args.gt)

pred = pred['answer']
gt = [e['text'] for e in gt['answer'].values]

acc = 0
for i, e in enumerate(pred):
    if e == gt[i]:
        acc += 1

print(acc / len(pred))
