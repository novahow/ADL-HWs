import jsonlines
from datasets import load_dataset
import json
output = []

ds = load_dataset("blended_skill_talk", split='train')
output = []
with jsonlines.open('botd/gd.jsonl') as f:
    for article, p in zip(f, ds):
        dialog = {'d': p['previous_utterance'] + article['dialog'], 'p': p['personas']}
        aid = article['id']

        output.append(dialog)
        

with open('botd/gd2.jsonl', "w") as f:
    for idx, dialog in enumerate(output):
        f.write(json.dumps({"id": idx, "dialog": dialog['d'], 'persona': dialog['p']}) + "\n")