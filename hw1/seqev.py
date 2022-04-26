from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
import pandas as pd
pred = pd.read_csv('./is.csv')
gt = pd.read_json('data/slot/eval.json')
lt = list(pred['tags'])
print(classification_report(list(gt['tags']), [e.split() for e in lt], mode='strict', scheme=IOB2))